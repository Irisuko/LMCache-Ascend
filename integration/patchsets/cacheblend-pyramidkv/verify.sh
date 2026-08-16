#!/usr/bin/env bash
set -euo pipefail

PATCH_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LMCACHE_ROOT=$(cd "$PATCH_ROOT/../../.." && pwd)
CORE_BASE=cd683d7f3bec0a8877c217f7e01c3812e1b98dd5
ASCEND_BASE=b0613602f502ffeb163ac5c4a6343f432880e38e
CORE_URL=${VLLM_REPO_URL:-https://github.com/vLLM-HUST/vllm-hust.git}
ASCEND_URL=${VLLM_ASCEND_REPO_URL:-https://github.com/vLLM-HUST/vllm-ascend-hust.git}
PYTHON_BIN=${PYTHON_BIN:-python3}
TEST_PYTHON_BIN=${TEST_PYTHON_BIN:-$PYTHON_BIN}
KEEP_VERIFY_WORKTREE=${KEEP_VERIFY_WORKTREE:-0}

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [EMPTY_WORK_DIR]" >&2
    exit 2
fi

cleanup=0
if [[ $# -eq 1 ]]; then
    VERIFY_ROOT=$1
    mkdir -p "$VERIFY_ROOT"
    [[ -z $(find "$VERIFY_ROOT" -mindepth 1 -maxdepth 1 -print -quit) ]] \
        || { echo "$VERIFY_ROOT must be empty" >&2; exit 2; }
else
    VERIFY_ROOT=$(mktemp -d)
    cleanup=1
fi

finish() {
    if [[ "$cleanup" == 1 && "$KEEP_VERIFY_WORKTREE" != 1 ]]; then
        rm -rf "$VERIFY_ROOT"
    else
        echo "Verification worktree: $VERIFY_ROOT" >&2
    fi
}
trap finish EXIT

CORE_DIR=$VERIFY_ROOT/vllm-hust
ASCEND_DIR=$VERIFY_ROOT/vllm-ascend-hust

git clone --no-checkout "$CORE_URL" "$CORE_DIR"
git -C "$CORE_DIR" checkout --detach "$CORE_BASE"
git clone --no-checkout "$ASCEND_URL" "$ASCEND_DIR"
git -C "$ASCEND_DIR" checkout --detach "$ASCEND_BASE"

PYTHON_BIN="$PYTHON_BIN" "$PATCH_ROOT/apply.sh" "$CORE_DIR" "$ASCEND_DIR"

export PYTHONPATH="$ASCEND_DIR:$CORE_DIR:$LMCACHE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$TEST_PYTHON_BIN" -m pytest \
    "$CORE_DIR/tests/v1/core/test_kv_cache_compression.py" -q
"$TEST_PYTHON_BIN" -m pytest \
    "$ASCEND_DIR/tests/ut/kv_cache_compression/test_lmcache_compat.py" \
    "$ASCEND_DIR/tests/ut/kv_cache_compression/test_pyramidkv.py" \
    "$ASCEND_DIR/tests/ut/kv_cache_compression/test_registry.py" -q
"$TEST_PYTHON_BIN" -m pytest \
    "$LMCACHE_ROOT/tests/integration/vllm/test_compression_coordination.py" \
    "$LMCACHE_ROOT/tests/integration/vllm/test_local_persist_skip.py" \
    "$LMCACHE_ROOT/tests/v1/test_cache_engine_retrieve.py" \
    "$LMCACHE_ROOT/benchmark/v1/rag/pyramidkv/test_tools.py" -q

if [[ ${RUN_NPU_TESTS:-0} == 1 ]]; then
    "$TEST_PYTHON_BIN" -m pytest \
        "$ASCEND_DIR/tests/ut/kv_cache_compression/a2/test_pyramidkv_npu.py" -q
else
    echo "NPU kernel tests skipped; set RUN_NPU_TESTS=1 to enable them."
fi

echo "Clean-checkout patch and targeted-test verification passed."
