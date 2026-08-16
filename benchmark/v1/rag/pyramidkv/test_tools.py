# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from pathlib import Path
from types import SimpleNamespace
import asyncio
import json

# Third Party
import compare_outputs
import pytest
import run_workload
import summarize
import validate


class FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        tokens = [10 + index % 17 for index, _ in enumerate(text.split())]
        return ([1] if add_special_tokens else []) + tokens


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs

        async def chunks():
            yield SimpleNamespace(
                choices=[SimpleNamespace(text="first", token_ids=[501])]
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(text=" second", model_extra={"token_ids": [502]})
                ]
            )

        return chunks()


class FakeClient:
    def __init__(self):
        self.completions = FakeCompletions()


@pytest.mark.parametrize("input_length", [1024, 4096, 4097, 7168])
def test_fixed_workload_has_exact_length_and_short_precomputes(input_length):
    items, precompute = run_workload.fixed_items(FakeTokenizer(), input_length, 5, 0)

    assert len(items) == 5
    assert all(item.prompt_tokens == input_length for item in items)
    assert all(len(prompt) < 4096 for prompt in precompute)
    assert [item.prompt for item in items] == [
        item.prompt
        for item in run_workload.fixed_items(FakeTokenizer(), input_length, 5, 0)[0]
    ]


def test_mixed_workload_interleaves_exact_short_and_long_prompts():
    items, precompute = run_workload.mixed_items(FakeTokenizer(), 1024, 7168, 6, 0)

    assert [item.prompt_tokens for item in items] == [1024, 7168] * 3
    assert all(len(prompt) < 4096 for prompt in precompute)


def test_rag_reader_accepts_jsonl_longbench_records(tmp_path: Path):
    dataset = tmp_path / "musique.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "input": "Who wrote it?",
                "context": "Passage 1:\nTitle\nFirst text.\n\nPassage 2:\nSecond text.",
                "answers": ["A writer"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    items, documents, sample_sha = run_workload.rag_items(
        FakeTokenizer(), dataset, tmp_path / "samples.json", 1
    )

    assert len(items) == 1
    assert len(documents) == 2
    assert items[0].quality_group == "contexts_2"
    assert len(sample_sha) == 64


def test_rag_reader_filters_by_minimum_prompt_length(tmp_path: Path):
    dataset = tmp_path / "musique.jsonl"
    records = [
        {
            "input": "Short?",
            "context": "Passage 1:\nbrief",
            "answers": ["brief"],
        },
        {
            "input": "Long?",
            "context": "Passage 1:\n" + "evidence " * 80,
            "answers": ["evidence"],
        },
    ]
    dataset.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    items, _, _ = run_workload.rag_items(
        FakeTokenizer(),
        dataset,
        tmp_path / "samples.json",
        1,
        min_prompt_tokens=100,
    )

    assert items[0].sample_index == 1


def test_precompute_settle_time_is_recordable_and_scales_with_documents():
    assert run_workload.resolve_precompute_settle_seconds(8, None) == 5
    assert run_workload.resolve_precompute_settle_seconds(187, None) == 46.75
    assert run_workload.resolve_precompute_settle_seconds(187, 12.5) == 12.5


def test_execute_item_records_server_token_ids_and_request_header():
    client = FakeClient()
    item = run_workload.WorkItem("fixed-0000", [1, 2], 2, [])

    result = asyncio.run(
        run_workload.execute_item(
            client,
            FakeTokenizer(),
            "model",
            item,
            2,
            asyncio.Semaphore(1),
        )
    )

    assert result.generated_token_ids == [501, 502]
    assert result.token_ids_from_server is True
    assert client.completions.kwargs["extra_body"]["return_token_ids"] is True
    assert client.completions.kwargs["extra_headers"] == {"x-request-id": "fixed-0000"}


def test_compare_outputs_requires_identical_token_ids(tmp_path: Path):
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    payload = {
        "requests": [
            {"request_id": "request", "generated_token_ids": [1, 2], "error": None}
        ]
    }
    left.write_text(json.dumps(payload), encoding="utf-8")
    right.write_text(json.dumps(payload), encoding="utf-8")

    compare_outputs.compare(left, right)

    payload["requests"][0]["generated_token_ids"] = [1, 3]
    right.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="generated token IDs differ"):
        compare_outputs.compare(left, right)


def test_summary_merges_structured_event_and_latency(tmp_path: Path):
    result = tmp_path / "cell.json"
    result.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark_type": "fixed",
                "metadata": {
                    "mode": "CP",
                    "repeat": 1,
                    "concurrency": 1,
                    "input_length": 7168,
                    "output_length": 512,
                },
                "precompute": {"documents": 8, "duration_s": 1.0},
                "summary": {
                    "duration_s": 2.0,
                    "completed": 1,
                    "failed": 0,
                    "request_throughput": 0.5,
                    "output_throughput": 256,
                    "f1": None,
                    "em": None,
                    "f1_by_group": None,
                    "em_by_group": None,
                },
                "requests": [
                    {
                        "request_id": "request",
                        "error": None,
                        "ttft_s": 0.1,
                        "tpot_s": 0.01,
                        "itl_s": [0.01],
                        "e2e_s": 0.2,
                    }
                ],
            }
        )
    )
    result.with_suffix(".server.log").write_text(
        "Reqid: cmpl-request-0, Total tokens 7168, Inference Engine computed "
        "tokens: 0, LMCache hit tokens: 7040, need to load: 7040\n"
        "prefix LMCACHE_PYRAMIDKV_EVENT "
        + json.dumps(
            {
                "lmcache_hit_tokens": 7040,
                "cacheblend_recomputed_tokens": 1056,
                "semantic_tokens": 7168,
                "physical_tokens": 1024,
                "source_blocks": 56,
                "destination_blocks": 8,
                "released_blocks": 48,
                "compression_ms": 4.0,
            }
        )
        + " \x1b[3m(connector.py:117:lmcache_ascend)\x1b[0m\n"
    )
    result.with_suffix(".metrics").write_text("vllm:kv_cache_usage_perc 0.25\n")

    row = summarize.aggregate_result(result)

    assert row["success_rate"] == 1
    assert row["ttft_p50_ms"] == pytest.approx(100)
    assert row["destination_blocks_max"] == 8
    assert row["lmcache_hit_tokens_total"] == 7040
    assert row["compression_lmcache_hit_tokens_total"] == 7040
    assert row["lmcache_lookup_hit_tokens_total"] == 7040
    assert row["lmcache_load_tokens_total"] == 7040
    assert row["precompute_settle_s"] == 0
    assert row["peak_kv_usage"] == 0.25


def test_summary_excludes_precompute_lmcache_hits(tmp_path: Path):
    log = tmp_path / "requests.server.log"
    log.write_text(
        "Reqid: cmpl-precompute-00000-0, Total tokens 512, Inference Engine "
        "computed tokens: 0, LMCache hit tokens: 41, need to load: 41\n"
        "Reqid: cmpl-fixed-7168-0000-0-deadbeef, Total tokens 7168, Inference Engine "
        "computed tokens: 0, LMCache hit tokens: 7037, need to load: 7037\n",
        encoding="utf-8",
    )

    assert summarize.load_lmcache_request_stats(log, ["fixed-7168-0000"]) == {
        "fixed-7168-0000": {"hit_tokens": 7037, "load_tokens": 7037}
    }


def test_summary_uses_core_event_when_lmcache_is_disabled(tmp_path: Path):
    log = tmp_path / "pyramid-only.server.log"
    event = {
        "compression_ms": 5.0,
        "destination_blocks": 8,
        "physical_tokens": 1024,
        "released_blocks": 48,
        "request_id": "request",
        "semantic_tokens": 7168,
        "source_blocks": 56,
        "transaction_id": 1,
    }
    log.write_text(
        "prefix VLLM_PYRAMIDKV_EVENT " + json.dumps(event) + " (scheduler.py:1:vllm)\n",
        encoding="utf-8",
    )

    assert summarize.load_events(log) == [event]


def test_summary_prefers_lmcache_event_over_duplicate_core_event(tmp_path: Path):
    log = tmp_path / "combined.server.log"
    core_event = {"request_id": "request", "released_blocks": 48}
    lmcache_event = {
        **core_event,
        "lmcache_hit_tokens": 7037,
        "cacheblend_recomputed_tokens": 1055,
    }
    log.write_text(
        "VLLM_PYRAMIDKV_EVENT "
        + json.dumps(core_event)
        + "\nLMCACHE_PYRAMIDKV_EVENT "
        + json.dumps(lmcache_event)
        + "\n",
        encoding="utf-8",
    )

    assert summarize.load_events(log) == [lmcache_event]


@pytest.mark.parametrize(
    ("workload", "fields", "expected"),
    [
        ("fixed", {"input_length": "7168", "completed": "7"}, 7),
        ("rag", {"min_prompt_tokens": "4097", "completed": "14"}, 14),
        (
            "mixed",
            {"input_lengths": "[1024, 7168]", "requests": "7", "completed": "7"},
            3,
        ),
        ("fixed", {"input_length": "4096", "completed": "7"}, None),
        ("rag", {"min_prompt_tokens": "0", "completed": "14"}, None),
    ],
)
def test_validator_computes_expected_compression_events(workload, fields, expected):
    row = {"mode": "CP", "workload": workload, **fields}

    assert validate.expected_compression_events(row) == expected


def test_validator_cell_key_separates_execution_conditions():
    row = {
        "workload": "fixed",
        "input_length": "7168",
        "input_lengths": "",
        "output_length": "512",
        "concurrency": "8",
        "cache_state": "full",
        "scheduler_mode": "async",
        "min_prompt_tokens": "0",
    }

    assert validate.cell_key(row) != validate.cell_key(
        {**row, "scheduler_mode": "sync"}
    )
    assert validate.cell_key(row) != validate.cell_key(
        {**row, "cache_state": "partial"}
    )


def test_validator_averages_f1_groups_across_repeats():
    rows = [
        {"f1_by_group": '{"2hop":10,"3hop":20}'},
        {"f1_by_group": '{"2hop":30,"3hop":40}'},
    ]

    assert validate.mean_group_metrics(rows, "f1_by_group") == {
        "2hop": 20,
        "3hop": 30,
    }
