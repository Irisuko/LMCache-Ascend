#!/usr/bin/env bash
set -euo pipefail

PATCH_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LMCACHE_ROOT=$(cd "$PATCH_ROOT/../../.." && pwd)
CORE_BASE=cd683d7f3bec0a8877c217f7e01c3812e1b98dd5
ASCEND_BASE=b0613602f502ffeb163ac5c4a6343f432880e38e
CORE_TREE=0dcfab0b172c783d01c3ef27301f6eba70f976e0
ASCEND_TREE=eeccfbf08be523b4f4d1678508b00fdb738a3617
PYTHON_BIN=${PYTHON_BIN:-python3}

usage() {
    echo "Usage: PYTHON_BIN=/path/to/python $0 VLLM_HUST_DIR VLLM_ASCEND_HUST_DIR" >&2
}

die() {
    echo "apply.sh: $*" >&2
    exit 1
}

require_clean_base() {
    local repo=$1
    local expected=$2
    local actual
    git -C "$repo" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
        || die "$repo is not a Git worktree"
    actual=$(git -C "$repo" rev-parse HEAD)
    [[ "$actual" == "$expected" ]] \
        || die "$repo HEAD is $actual, expected $expected"
    [[ -z $(git -C "$repo" status --porcelain --untracked-files=all) ]] \
        || die "$repo must be completely clean"
}

for_each_patch() {
    local repo=$1
    local patch_dir=$2
    local operation=$3
    local patch
    while IFS= read -r patch || [[ -n "$patch" ]]; do
        [[ -z "$patch" || "$patch" == \#* ]] && continue
        [[ "$patch" != /* && "$patch" != *..* ]] \
            || die "unsafe patch path in $patch_dir/series: $patch"
        "$operation" "$repo" "$PATCH_ROOT/$patch_dir/$patch"
    done < "$PATCH_ROOT/$patch_dir/series"
}

check_patch() {
    git -C "$1" apply --check "$2"
}

apply_patch_file() {
    git -C "$1" apply "$2"
}

worktree_tree() {
    local repo=$1
    local index_dir
    local index_file
    index_dir=$(mktemp -d)
    index_file=$index_dir/index
    GIT_INDEX_FILE="$index_file" git -C "$repo" read-tree HEAD
    GIT_INDEX_FILE="$index_file" git -C "$repo" add -A
    GIT_INDEX_FILE="$index_file" git -C "$repo" write-tree
    rm -rf "$index_dir"
}

runtime_check() {
    local core=$1
    local ascend=$2
    PYTHONPATH="$core:$ascend:$LMCACHE_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON_BIN" - "$core" "$ascend" "$LMCACHE_ROOT" <<'PY'
from importlib import metadata, util
from pathlib import Path
import sys

if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"Python 3.11 required, got {sys.version.split()[0]}")

required = {
    "lmcache": "0.4.4",
    "torch": "2.9.0",
    "torch-npu": "2.9.0",
    "vllm": "0.18.0",
    "vllm-ascend": "0.18.0",
}
for distribution, version_prefix in required.items():
    actual = metadata.version(distribution)
    if not (actual == version_prefix or actual.startswith(version_prefix + "+")):
        raise SystemExit(
            f"{distribution} version {actual!r} does not match {version_prefix!r}"
        )

expected_sources = {
    "vllm": Path(sys.argv[1]),
    "vllm_ascend": Path(sys.argv[2]),
    "lmcache_ascend": Path(sys.argv[3]),
}
for package, root in expected_sources.items():
    spec = util.find_spec(package)
    if spec is None or spec.origin is None:
        raise SystemExit(f"cannot resolve source for {package}")
    origin = Path(spec.origin).resolve()
    try:
        origin.relative_to(root.resolve())
    except ValueError as exc:
        raise SystemExit(
            f"{package} resolves to {origin}, expected a source under {root.resolve()}"
        ) from exc
    print(f"{package}_source={origin}")
PY

    local cann_info=/usr/local/Ascend/cann-9.0.0/aarch64-linux/ascend_toolkit_install.info
    [[ -f "$cann_info" ]] || die "CANN 9.0.0 manifest not found at $cann_info"
    grep -qx 'version=9.0.0' "$cann_info" \
        || die "CANN manifest does not report version 9.0.0"
}

[[ $# -eq 2 ]] || {
    usage
    exit 2
}

CORE_DIR=$(cd "$1" && pwd)
ASCEND_DIR=$(cd "$2" && pwd)

require_clean_base "$CORE_DIR" "$CORE_BASE"
require_clean_base "$ASCEND_DIR" "$ASCEND_BASE"
(cd "$PATCH_ROOT" && sha256sum -c SHA256SUMS)
runtime_check "$CORE_DIR" "$ASCEND_DIR"

# Validate every patch before changing either worktree.
for_each_patch "$CORE_DIR" vllm-hust check_patch
for_each_patch "$ASCEND_DIR" vllm-ascend-hust check_patch

for_each_patch "$CORE_DIR" vllm-hust apply_patch_file
for_each_patch "$ASCEND_DIR" vllm-ascend-hust apply_patch_file

[[ $(worktree_tree "$CORE_DIR") == "$CORE_TREE" ]] \
    || die "vllm-hust applied tree does not match $CORE_TREE"
[[ $(worktree_tree "$ASCEND_DIR") == "$ASCEND_TREE" ]] \
    || die "vllm-ascend-hust applied tree does not match $ASCEND_TREE"
runtime_check "$CORE_DIR" "$ASCEND_DIR"

echo "Applied CacheBlend + PyramidKV patchsets successfully."
