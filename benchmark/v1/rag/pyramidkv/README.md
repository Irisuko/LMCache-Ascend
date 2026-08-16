# CacheBlend + PyramidKV benchmark

This directory is the executable experiment definition for the four modes in
`matrix.yaml`: baseline (`B`), CacheBlend (`C`), PyramidKV (`P`), and the
combined path (`CP`). Each cell restarts the server, uses one physical NPU, and
receives deterministic prompts generated with seed 0.

The fixed workload constructs every long prompt from independently cached
document segments. Each precompute request remains below 4096 tokens, while the
final 7168-token request crosses the PyramidKV threshold. Compressed requests
therefore cannot overwrite the full document cache. `cold`, `partial`, and
`full` cache states are selected with `CACHE_STATE`.

## Run

Apply and verify the external patchsets first. Then run from an environment
containing the versions in `integration/patchsets/cacheblend-pyramidkv/manifest.yaml`:

```bash
export PYTHON_BIN=/root/miniconda3/envs/kvcache-lmcache-ascend/bin/python
export MODEL_PATH=/workspace/models/Meta-Llama-3-8B-Instruct
export VLLM_DIR=/path/to/patched/vllm-hust
export VLLM_ASCEND_DIR=/path/to/patched/vllm-ascend-hust
export MUSIQUE_DATA=/path/to/musique_ans_v1.0_dev.jsonl
export ASCEND_RT_VISIBLE_DEVICES=1

./benchmark/v1/rag/pyramidkv/run_matrix.sh /path/to/results
```

`run_matrix.sh` executes boundary prompts (1024/4096/4097), the 7168/512
decode matrix at concurrency 1/8/16/32, and the first 200 valid MuSiQue
samples. Defaults are 48, 112, and 200 requests respectively, with three
repetitions in B/C/P/CP order. The sample index file and dataset hash are
recorded with the raw results.

The reader accepts the official MuSiQue JSONL format, CacheBlend's `ctxs` JSON
format, and LongBench MuSiQue JSONL. Use the official answerable dev split from
<https://github.com/StonyBrookNLP/musique> for the release matrix. A sample is
valid when its question, answer, and documents are present, its complete prompt
fits 8192 tokens, and every document precompute remains below the 4096-token
compression threshold. The resulting first-200 index list is persisted and
hashed, so every mode uses exactly the same examples.

The official `musique_v1.0.zip` archive used for validation has SHA256
`98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd`.
Its extracted `musique_ans_v1.0_dev.jsonl` has SHA256
`15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b`.
With the pinned Llama-3 tokenizer, the first-200 sample list has SHA256
`7cc6515ace561c77d870917b576e38eb6a58ba49fadc16915e3362931b344b9a`.
Those samples contain 3075 unique document prompts (474489 tokens, 3268
256-token cache chunks), requiring about 102.1 GiB of uncompressed BF16 KV. The
checked-in LMCache configuration reserves 120 GB so the `full` state does not
silently become an eviction workload. Its `blend_min_tokens: 64` makes every
official document in these selections eligible at the length gate (the
shortest is 80 tokens); the upstream default of 256 would exclude many of
them. Actual hit counts remain an observed result, not a configuration
guarantee.

The literal first 200 valid records range from 1261 to 3846 prompt tokens, so
they do not cross the 4096-token PyramidKV threshold. Across the full official
dev split only 14 records cross it (4136-4677 tokens). Treat the required
first-200 MuSiQue F1 comparison as the RAG/CacheBlend quality gate, not by
itself as evidence about compressed-request quality. The fixed 7168-token
matrix supplies the primary combined-path evidence. Use the executable
supplementary subset when publishing a PyramidKV quality claim:

The pinned 14-sample list has SHA256
`9e473a8e896a62a1ec0d64564fd1d4c36fb370008d27f0b748e546eb0956fe29`
and contains six `2hop`, five `3hop1`, and three `4hop1` records.

```bash
RUN_BOUNDARY=0 RUN_DECODE=0 RUN_RAG=1 MODE_LIST="B P CP" REPEATS=1 \
RAG_REQUESTS=14 RAG_MIN_PROMPT_TOKENS=4097 \
./benchmark/v1/rag/pyramidkv/run_matrix.sh /path/to/long-rag-results
```

For a short wiring check, override the matrix without changing its definition:

```bash
REPEATS=1 BOUNDARY_REQUESTS=4 DECODE_REQUESTS=4 RAG_REQUESTS=4 \
DECODE_CONCURRENCY=1 MODE_LIST="B C P CP" \
./benchmark/v1/rag/pyramidkv/run_matrix.sh /path/to/smoke-results
```

To exercise cache states separately, use three output directories:

```bash
for state in cold partial full; do
  CACHE_STATE=$state REPEATS=1 MODE_LIST=CP RUN_BOUNDARY=0 \
  DECODE_CONCURRENCY=1 RUN_RAG=0 \
  ./benchmark/v1/rag/pyramidkv/run_matrix.sh "/path/to/e2e-$state"
done
```

The mixed-batch check alternates 1024- and 7168-token requests in one batch:

```bash
RUN_BOUNDARY=0 RUN_DECODE=0 RUN_RAG=0 RUN_MIXED=1 \
REPEATS=1 MODE_LIST=CP MIXED_REQUESTS=16 MIXED_CONCURRENCY=8 \
./benchmark/v1/rag/pyramidkv/run_matrix.sh /path/to/mixed-results
```

Run identical CP inputs once with each scheduler, then compare the exact
greedy token IDs returned by the server:

```bash
ASYNC_SCHEDULING=1 REPEATS=1 MODE_LIST=CP RUN_BOUNDARY=0 RUN_RAG=0 \
DECODE_CONCURRENCY=1 DECODE_REQUESTS=4 \
./benchmark/v1/rag/pyramidkv/run_matrix.sh /path/to/async-results

ASYNC_SCHEDULING=0 REPEATS=1 MODE_LIST=CP RUN_BOUNDARY=0 RUN_RAG=0 \
DECODE_CONCURRENCY=1 DECODE_REQUESTS=4 \
./benchmark/v1/rag/pyramidkv/run_matrix.sh /path/to/sync-results

python benchmark/v1/rag/pyramidkv/compare_outputs.py \
  /path/to/async-results/raw/fixed-in7168-out512-c1-CP-r1-full-async.json \
  /path/to/sync-results/raw/fixed-in7168-out512-c1-CP-r1-full-sync.json
```

## Outputs

Every cell produces a client JSON, server log, exact launch command, sampled
Prometheus metrics, and a final idle snapshot. The runner fails unless waiting
requests, running requests, and KV usage all return to zero. `summarize.py`
joins those sources into `summary.csv`; `validate.py` writes
`validation.json` and enforces the release gates when the complete three-run
matrix is present. Client-supplied request IDs let the summarizer report
LMCache lookup hits and requested load tokens for C as well as CP while
excluding document-precompute traffic. In mixed batches,
`lmcache_hit_tokens_*` covers all benchmark requests and
`compression_lmcache_hit_tokens_*` covers only requests that committed a
PyramidKV transaction.

LMCache save completion trails the OpenAI response. Before steady-state
requests, the runner therefore waits `max(5 seconds, 0.25 seconds per
precomputed document)` after the last precompute response and includes that
barrier in `precompute.duration_s`. Set `PRECOMPUTE_SETTLE_SECONDS` to an
explicit measured value when storage profiling shows a longer device-specific
drain time; the chosen value is retained in `precompute.settle_s`.

The validator never infers decode acceleration from block release alone. It
sets `decode_speedup_publishable=true` only when every one of the three paired
CP runs improves TPOT by at least 10% over C. Kernel profiler output and HBM
samples remain supporting artifacts and must not replace the scheduler block
or structured transaction evidence.
