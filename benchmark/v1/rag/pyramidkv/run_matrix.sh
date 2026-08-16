#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LMCACHE_ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
PYTHON_EXEC=$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')
VLLM_BIN=${VLLM_BIN:-$(dirname "$PYTHON_EXEC")/vllm}
MODEL_PATH=${MODEL_PATH:-/workspace/models/Meta-Llama-3-8B-Instruct}
VLLM_DIR=${VLLM_DIR:-/workspace/vllm-hust}
VLLM_ASCEND_DIR=${VLLM_ASCEND_DIR:-/workspace/vllm-ascend-hust}
MUSIQUE_DATA=${MUSIQUE_DATA:-}
RESULT_ROOT=${1:-$SCRIPT_DIR/results/$(date -u +%Y%m%dT%H%M%SZ)}
PORT=${PORT:-8000}
NPU_DEVICE=${ASCEND_RT_VISIBLE_DEVICES:-0}
REPEATS=${REPEATS:-3}
MODE_LIST=${MODE_LIST:-"B C P CP"}
DECODE_CONCURRENCY=${DECODE_CONCURRENCY:-"1 8 16 32"}
BOUNDARY_CONCURRENCY=${BOUNDARY_CONCURRENCY:-8}
RAG_CONCURRENCY=${RAG_CONCURRENCY:-8}
BOUNDARY_REQUESTS=${BOUNDARY_REQUESTS:-48}
DECODE_REQUESTS=${DECODE_REQUESTS:-112}
RAG_REQUESTS=${RAG_REQUESTS:-200}
RAG_MIN_PROMPT_TOKENS=${RAG_MIN_PROMPT_TOKENS:-0}
PRECOMPUTE_SETTLE_SECONDS=${PRECOMPUTE_SETTLE_SECONDS:-}
MIXED_REQUESTS=${MIXED_REQUESTS:-16}
MIXED_CONCURRENCY=${MIXED_CONCURRENCY:-8}
RUN_BOUNDARY=${RUN_BOUNDARY:-1}
RUN_DECODE=${RUN_DECODE:-1}
RUN_RAG=${RUN_RAG:-1}
RUN_MIXED=${RUN_MIXED:-0}
CACHE_STATE=${CACHE_STATE:-full}
SERVER_TIMEOUT=${SERVER_TIMEOUT:-900}
ASYNC_SCHEDULING=${ASYNC_SCHEDULING:-1}
if [[ "$ASYNC_SCHEDULING" == 1 ]]; then
    SCHEDULER_MODE=async
else
    SCHEDULER_MODE=sync
fi

PYRAMID_CONFIG='{"schema_version":1,"provider":"pyramidkv_ascend","provider_config":{"max_capacity_prompt":512,"min_compression_prompt_tokens":4096,"window_size":8,"kernel_size":7,"pooling":"maxpool","beta":20,"kv_cache_granularity":"kv_head","gqa_score_aggregation":"mean","merge":null}}'
CONNECTOR_CONFIG='{"kv_connector":"LMCacheAscendConnectorV1Dynamic","kv_role":"kv_both","kv_connector_module_path":"lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"}'

mkdir -p "$RESULT_ROOT/raw"
RESULT_ROOT=$(cd "$RESULT_ROOT" && pwd)
export PYTHONPATH="$VLLM_ASCEND_DIR:$VLLM_DIR:$LMCACHE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES="$NPU_DEVICE"
export VLLM_KNORM_ENABLED=0
export VLLM_USE_V2_MODEL_RUNNER=0
export PYTHONHASHSEED=0
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
export -n VLLM_DIR VLLM_ASCEND_DIR

precompute_settle_args=()
if [[ -n "$PRECOMPUTE_SETTLE_SECONDS" ]]; then
    precompute_settle_args=(
        --precompute-settle-seconds "$PRECOMPUTE_SETTLE_SECONDS"
    )
fi

[[ -x "$VLLM_BIN" ]] || {
    echo "vLLM console entry point is not executable: $VLLM_BIN" >&2
    exit 2
}

if [[ "$RUN_RAG" == 1 && -z "$MUSIQUE_DATA" ]]; then
    echo "MUSIQUE_DATA is required when RUN_RAG=1" >&2
    exit 2
fi

server_pid=
metrics_pid=
stop_server() {
    if [[ -n "$metrics_pid" ]] && kill -0 "$metrics_pid" 2>/dev/null; then
        kill "$metrics_pid" 2>/dev/null || true
        wait "$metrics_pid" 2>/dev/null || true
    fi
    metrics_pid=
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid" 2>/dev/null || true
        for _ in $(seq 1 120); do
            kill -0 "$server_pid" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$server_pid" 2>/dev/null; then
            echo "server $server_pid did not stop within 120 seconds" >&2
            return 1
        fi
        wait "$server_pid" 2>/dev/null || true
    fi
    server_pid=
}
trap stop_server EXIT

assert_server_idle() {
    local stem=$1
    local snapshot=$RESULT_ROOT/raw/$stem.final.metrics
    for _ in $(seq 1 60); do
        if curl -fsS "http://127.0.0.1:$PORT/metrics" > "$snapshot"; then
            if "$PYTHON_BIN" - "$snapshot" <<'PY'
from pathlib import Path
import re
import sys

tracked = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
)
values = {name: [] for name in tracked}
for line in Path(sys.argv[1]).read_text().splitlines():
    for name in tracked:
        if line.startswith(name):
            match = re.search(r"\s([0-9.eE+-]+)$", line)
            if match:
                values[name].append(float(match.group(1)))
if not all(values.values()):
    raise SystemExit(1)
if any(value != 0 for samples in values.values() for value in samples):
    raise SystemExit(1)
PY
            then
                return 0
            fi
        fi
        sleep 1
    done
    echo "server did not return to zero waiting/running/KV usage for $stem" >&2
    return 1
}

record_environment() {
    local output=$RESULT_ROOT/environment.txt
    {
        date -u +%Y-%m-%dT%H:%M:%SZ
        echo "model=$MODEL_PATH"
        echo "npu_device=$NPU_DEVICE"
        echo "scheduler_mode=$SCHEDULER_MODE"
        "$PYTHON_BIN" --version
        "$PYTHON_BIN" - <<'PY'
from importlib import metadata

for name in ("lmcache", "lmcache-ascend", "torch", "torch-npu", "vllm", "vllm-ascend"):
    try:
        version = metadata.version(name)
    except metadata.PackageNotFoundError:
        version = "not-installed (source checkout on PYTHONPATH)"
    print(f"{name}={version}")
PY
        git -C "$LMCACHE_ROOT" rev-parse HEAD
        git -C "$VLLM_DIR" rev-parse HEAD
        git -C "$VLLM_ASCEND_DIR" rev-parse HEAD
        sha256sum "$MODEL_PATH/config.json" "$MODEL_PATH/model.safetensors.index.json" "$MODEL_PATH/tokenizer.json"
        if [[ -n "$MUSIQUE_DATA" ]]; then sha256sum "$MUSIQUE_DATA"; fi
        npu-smi info
    } > "$output"
}

start_server() {
    local mode=$1
    local stem=$2
    local log=$RESULT_ROOT/raw/$stem.server.log
    local metrics=$RESULT_ROOT/raw/$stem.metrics
    local -a command=(
        "$VLLM_BIN" serve "$MODEL_PATH"
        --host 127.0.0.1
        --port "$PORT"
        --served-model-name "$MODEL_PATH"
        --dtype bfloat16
        --tensor-parallel-size 1
        --pipeline-parallel-size 1
        --max-model-len 8192
        --block-size 128
        --max-num-seqs 32
        --max-num-batched-tokens 2048
        --gpu-memory-utilization 0.8
        --enable-chunked-prefill
        --no-enable-prefix-caching
        --enforce-eager
        --seed 0
    )
    if [[ "$ASYNC_SCHEDULING" == 1 ]]; then
        command+=(--async-scheduling)
    else
        command+=(--no-async-scheduling)
    fi
    if [[ "$mode" == C || "$mode" == CP ]]; then
        command+=(--kv-transfer-config "$CONNECTOR_CONFIG")
    fi
    if [[ "$mode" == P || "$mode" == CP ]]; then
        command+=(--kv-cache-compression-config "$PYRAMID_CONFIG")
    fi
    printf '%q ' "${command[@]}" > "$RESULT_ROOT/raw/$stem.command"
    printf '\n' >> "$RESULT_ROOT/raw/$stem.command"
    if [[ "$mode" == C || "$mode" == CP ]]; then
        LMCACHE_CONFIG_FILE="$SCRIPT_DIR/lmcache_blend.yaml" "${command[@]}" > "$log" 2>&1 &
    else
        "${command[@]}" > "$log" 2>&1 &
    fi
    server_pid=$!

    for second in $(seq 1 "$SERVER_TIMEOUT"); do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            wait "$server_pid" || true
            echo "server failed for $stem; inspect $log" >&2
            return 1
        fi
        if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            break
        fi
        if [[ "$second" == "$SERVER_TIMEOUT" ]]; then
            echo "server readiness timed out for $stem" >&2
            return 1
        fi
        sleep 1
    done

    (
        while kill -0 "$server_pid" 2>/dev/null; do
            echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ)"
            curl -fsS "http://127.0.0.1:$PORT/metrics" || true
            sleep 1
        done
    ) > "$metrics" 2>/dev/null &
    metrics_pid=$!
}

run_fixed_cell() {
    local input_length=$1
    local output_length=$2
    local requests=$3
    local concurrency=$4
    local mode=$5
    local repeat=$6
    local stem="fixed-in${input_length}-out${output_length}-c${concurrency}-${mode}-r${repeat}-${CACHE_STATE}-${SCHEDULER_MODE}"
    echo "Running $stem"
    start_server "$mode" "$stem"
    "$PYTHON_BIN" "$SCRIPT_DIR/run_workload.py" fixed \
        --base-url "http://127.0.0.1:$PORT/v1" \
        --model "$MODEL_PATH" --tokenizer "$MODEL_PATH" \
        --mode "$mode" --repeat "$repeat" --seed 0 \
        --input-length "$input_length" --max-tokens "$output_length" \
        --requests "$requests" --concurrency "$concurrency" \
        --scheduler-mode "$SCHEDULER_MODE" \
        "${precompute_settle_args[@]}" \
        --cache-state "$CACHE_STATE" \
        --output "$RESULT_ROOT/raw/$stem.json"
    assert_server_idle "$stem"
    stop_server
}

run_mixed_cell() {
    local mode=$1
    local repeat=$2
    local stem="mixed-in1024-7168-out32-c${MIXED_CONCURRENCY}-${mode}-r${repeat}-${CACHE_STATE}-${SCHEDULER_MODE}"
    echo "Running $stem"
    start_server "$mode" "$stem"
    "$PYTHON_BIN" "$SCRIPT_DIR/run_workload.py" mixed \
        --base-url "http://127.0.0.1:$PORT/v1" \
        --model "$MODEL_PATH" --tokenizer "$MODEL_PATH" \
        --mode "$mode" --repeat "$repeat" --seed 0 \
        --short-input-length 1024 --long-input-length 7168 --max-tokens 32 \
        --requests "$MIXED_REQUESTS" --concurrency "$MIXED_CONCURRENCY" \
        --scheduler-mode "$SCHEDULER_MODE" \
        "${precompute_settle_args[@]}" \
        --cache-state "$CACHE_STATE" \
        --output "$RESULT_ROOT/raw/$stem.json"
    assert_server_idle "$stem"
    stop_server
}

run_rag_cell() {
    local mode=$1
    local repeat=$2
    local sample_suffix=
    if [[ "$RAG_MIN_PROMPT_TOKENS" -gt 0 ]]; then
        sample_suffix="-min${RAG_MIN_PROMPT_TOKENS}"
    fi
    local stem="rag-musique-${RAG_REQUESTS}${sample_suffix}-out32-c${RAG_CONCURRENCY}-${mode}-r${repeat}-${CACHE_STATE}-${SCHEDULER_MODE}"
    echo "Running $stem"
    start_server "$mode" "$stem"
    "$PYTHON_BIN" "$SCRIPT_DIR/run_workload.py" rag \
        --base-url "http://127.0.0.1:$PORT/v1" \
        --model "$MODEL_PATH" --tokenizer "$MODEL_PATH" \
        --mode "$mode" --repeat "$repeat" --seed 0 \
        --dataset "$MUSIQUE_DATA" \
        --sample-list "$RESULT_ROOT/raw/sample_indices.json" \
        --min-prompt-tokens "$RAG_MIN_PROMPT_TOKENS" \
        "${precompute_settle_args[@]}" \
        --max-tokens 32 --requests "$RAG_REQUESTS" --concurrency "$RAG_CONCURRENCY" \
        --scheduler-mode "$SCHEDULER_MODE" \
        --cache-state "$CACHE_STATE" \
        --output "$RESULT_ROOT/raw/$stem.json"
    assert_server_idle "$stem"
    stop_server
}

record_environment
for repeat in $(seq 1 "$REPEATS"); do
    if [[ "$RUN_BOUNDARY" == 1 ]]; then
        for input_length in 1024 4096 4097; do
            for mode in $MODE_LIST; do
                run_fixed_cell "$input_length" 256 "$BOUNDARY_REQUESTS" "$BOUNDARY_CONCURRENCY" "$mode" "$repeat"
            done
        done
    fi
    if [[ "$RUN_DECODE" == 1 ]]; then
        for concurrency in $DECODE_CONCURRENCY; do
            for mode in $MODE_LIST; do
                run_fixed_cell 7168 512 "$DECODE_REQUESTS" "$concurrency" "$mode" "$repeat"
            done
        done
    fi
    if [[ "$RUN_MIXED" == 1 ]]; then
        for mode in $MODE_LIST; do
            run_mixed_cell "$mode" "$repeat"
        done
    fi
    if [[ "$RUN_RAG" == 1 ]]; then
        for mode in $MODE_LIST; do
            run_rag_cell "$mode" "$repeat"
        done
    fi
done

"$PYTHON_BIN" "$SCRIPT_DIR/summarize.py" "$RESULT_ROOT/raw" "$RESULT_ROOT/summary.csv"
if [[ "$REPEATS" -ge 3 && "$MODE_LIST" == "B C P CP" ]]; then
    "$PYTHON_BIN" "$SCRIPT_DIR/validate.py" "$RESULT_ROOT/summary.csv" \
        --output "$RESULT_ROOT/validation.json"
fi
sha256sum "$RESULT_ROOT"/raw/*.json "$RESULT_ROOT/summary.csv" \
    > "$RESULT_ROOT/result-SHA256SUMS"
echo "Results: $RESULT_ROOT"
