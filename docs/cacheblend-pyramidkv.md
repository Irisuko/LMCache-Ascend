# CacheBlend + PyramidKV on Ascend

## Scope

This integration loads reusable RAG document KV through LMCache, lets
CacheBlend recompute its selected tokens, preserves the final prefill window,
and then commits PyramidKV's compacted block table before decode. It is a
fail-closed integration with the following supported configuration:

- `/workspace/models/Meta-Llama-3-8B-Instruct`, BF16,
  `LlamaForCausalLM`, 32 layers, 32 query heads, 8 KV heads, head dimension 128.
- One Ascend 910B2 with CANN 9.0.0; TP, PP, PCP, and DCP are all 1.
- `LMCacheAscendConnectorV1Dynamic`, `kv_both`, `use_layerwise: true`, and
  `enable_blending: true`.
- Eager execution, chunked prefill, async scheduling, block size 128, and
  vLLM prefix caching disabled.

P/D disaggregation, remote decode, KV offload, speculative decoding, KNorm,
quantized KV, other connectors and roles, and non-Llama models are rejected.
Qwen support is intentionally absent. PIECEWISE graph execution is not part of
the first release claim.

## Data flow and ownership

For prompts at or below the 4096-token threshold, LMCache stores and loads KV
normally. For longer prompts, the vLLM scheduler applies one shared hit limit
to local and external hits and reserves at least one complete 128-token tail
block for final recomputation. The Ascend provider sees the final prefill,
selects PyramidKV destinations, compacts KV, and returns a transaction plan.

The scheduler commits the plan atomically, updates the block table, releases
source blocks, and sends a commit event with the next connector ACK. LMCache
marks only that transaction's `save_spec.can_save` false. Short document
precompute requests remain writable; a compacted request can never overwrite
their complete KV entries.

The model tracker is registered immediately after model loading and before KV
connector initialization. Startup logs include the connector module, role,
LMCache blend flags, PyramidKV provider configuration, and the final
`none`, `lmcache_local_blend`, or `unsupported` compatibility decision.

## Apply the pinned patchsets

Start with the exact revisions listed in the manifest and a Python environment
containing LMCache 0.4.4, torch/torch-npu 2.9.0, vLLM/vLLM-Ascend 0.18.0,
Python 3.11, and CANN 9.0.0:

```bash
git clone https://github.com/vLLM-HUST/vllm-hust.git /src/vllm-hust
git -C /src/vllm-hust checkout --detach cd683d7f3bec0a8877c217f7e01c3812e1b98dd5

git clone https://github.com/vLLM-HUST/vllm-ascend-hust.git /src/vllm-ascend-hust
git -C /src/vllm-ascend-hust checkout --detach b0613602f502ffeb163ac5c4a6343f432880e38e

PYTHON_BIN=/path/to/python3.11 \
  ./integration/patchsets/cacheblend-pyramidkv/apply.sh \
  /src/vllm-hust /src/vllm-ascend-hust
```

`apply.sh` rejects untracked or modified files and a mismatched `HEAD`. Before
changing either repository it verifies `SHA256SUMS`, every `series` entry,
both `git apply --check` results, the runtime versions, and Python package
origins. After application it builds each worktree in a temporary Git index
and compares the resulting tree object with `manifest.yaml`.

To recreate both repositories and run all targeted tests from scratch:

```bash
PYTHON_BIN=/path/to/python3.11 \
  ./integration/patchsets/cacheblend-pyramidkv/verify.sh

# Include the Ascend kernel tests on an available 910B2.
ASCEND_RT_VISIBLE_DEVICES=1 RUN_NPU_TESTS=1 \
PYTHON_BIN=/path/to/python3.11 \
  ./integration/patchsets/cacheblend-pyramidkv/verify.sh
```

If runtime and developer test dependencies are intentionally split between
two environments, keep `PYTHON_BIN` on the pinned runtime and set
`TEST_PYTHON_BIN` to the Python containing the vLLM pytest extras. Both test
processes still load the newly patched checkout through `PYTHONPATH`.

`VLLM_REPO_URL` and `VLLM_ASCEND_REPO_URL` may point to local mirrors. The
revisions remain fixed. This integration does not use the historical
site-packages `worker_v1.py` patcher; that patcher remains available only for
older CacheBlend installations.

## Launch configurations

All four experiment modes share these server arguments:

```text
--dtype bfloat16
--tensor-parallel-size 1
--pipeline-parallel-size 1
--max-model-len 8192
--block-size 128
--max-num-seqs 32
--max-num-batched-tokens 2048
--gpu-memory-utilization 0.8
--enable-chunked-prefill
--async-scheduling
--no-enable-prefix-caching
--enforce-eager
--seed 0
```

C and CP set `LMCACHE_CONFIG_FILE` to
`benchmark/v1/rag/pyramidkv/lmcache_blend.yaml` and add:

```bash
--kv-transfer-config '{
  "kv_connector": "LMCacheAscendConnectorV1Dynamic",
  "kv_role": "kv_both",
  "kv_connector_module_path":
    "lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"
}'
```

P and CP add:

```bash
--kv-cache-compression-config '{
  "schema_version": 1,
  "provider": "pyramidkv_ascend",
  "provider_config": {
    "max_capacity_prompt": 512,
    "min_compression_prompt_tokens": 4096,
    "window_size": 8,
    "kernel_size": 7,
    "pooling": "maxpool",
    "beta": 20,
    "kv_cache_granularity": "kv_head",
    "gqa_score_aggregation": "mean",
    "merge": null
  }
}'
```

Removing both feature arguments produces B. Adding only KV transfer produces
C; adding only compression produces P; adding both produces CP.

## Structured evidence

Every committed combined transaction emits one log record prefixed with
`LMCACHE_PYRAMIDKV_EVENT`. The JSON payload contains:

```json
{
  "cacheblend_recomputed_tokens": 1055,
  "compression_ms": 4.25,
  "destination_blocks": 8,
  "lmcache_hit_tokens": 7037,
  "physical_tokens": 991,
  "released_blocks": 48,
  "semantic_tokens": 7168,
  "source_blocks": 56
}
```

The scheduler transaction ID is used for correlation but is not included in
the stable public metric fields. A missing or mismatched ACK is an error rather
than a silent downgrade. Request abort, preemption, stale async output, and
normal completion all clear provider state.

PyramidKV-only mode emits `VLLM_PYRAMIDKV_EVENT` at the same successful core
commit point. It contains the semantic/physical lengths, block counts, and
compression time but omits the LMCache hit and CacheBlend recompute fields.
The benchmark summarizer prefers the richer LMCache event when both markers
refer to a combined transaction, avoiding double counting.

## NPU acceptance and performance

The complete command and result contract live in
`benchmark/v1/rag/pyramidkv/README.md`. The matrix alternates B/C/P/CP on one
NPU, independently restarts every cell, repeats it three times, and records
raw JSON, server logs, exact commands, environment versions, model/data/sample
hashes, Prometheus samples, `summary.csv`, and `validation.json`.

Use separate `cold`, `partial`, and `full` CP runs for cache-state E2E, and run
C alone for CacheBlend document precompute/hit and P alone for 7168/512
compression/decode. A mixed short/long batch and async-off comparison must use
the same generated token IDs and seed; retain both raw result files. The runner
fails if running requests, waiting requests, or KV usage do not return to zero.

Release gates are mechanical: 100% request success; no NaN, duplicate plan,
invalid block table, or cache pollution; no more than eight destination blocks
for a 7168-token prompt; average MuSiQue F1 drop at most 3 and per-group drop at
most 5; CP TTFT no more than 110% of C and lower than B. Report decode speedup
only when all three paired CP runs improve TPOT over C by at least 10%.
Otherwise report only the observed physical sequence length, released blocks,
and KV capacity reduction.
