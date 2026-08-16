# SPDX-License-Identifier: Apache-2.0
"""Evaluate the release gates from a completed four-mode result matrix."""

# Future
from __future__ import annotations

# Standard
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any
import argparse
import csv
import json


def number(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    return float(value) if value not in {"", "None"} else None


CELL_FIELDS = (
    "workload",
    "input_length",
    "input_lengths",
    "output_length",
    "concurrency",
    "cache_state",
    "scheduler_mode",
    "min_prompt_tokens",
    "requests",
    "model",
    "seed",
    "dataset_sha256",
    "sample_list_sha256",
)


def cell_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(field, "") for field in CELL_FIELDS)


def expected_compression_events(row: dict[str, str]) -> int | None:
    if row.get("mode") not in {"P", "CP"}:
        return None

    completed = int(number(row, "completed") or 0)
    if row.get("workload") == "fixed":
        input_length = number(row, "input_length")
        return completed if input_length is not None and input_length > 4096 else None
    if row.get("workload") == "rag":
        minimum = number(row, "min_prompt_tokens")
        return completed if minimum is not None and minimum > 4096 else None
    if row.get("workload") == "mixed":
        lengths = json.loads(row.get("input_lengths") or "[]")
        requests = int(number(row, "requests") or completed)
        if not lengths or not any(length > 4096 for length in lengths):
            return None
        return sum(lengths[index % len(lengths)] > 4096 for index in range(requests))
    return None


def mean_group_metrics(rows: list[dict[str, str]], field: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        values = json.loads(row.get(field) or "{}")
        for group, value in values.items():
            grouped[group].append(float(value))
    return {group: mean(values) for group, values in grouped.items()}


def validate_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    failures = []
    for row in rows:
        if number(row, "success_rate") != 1.0:
            failures.append(f"{row['result_file']}: success rate is not 100%")
        expected_events = expected_compression_events(row)
        if expected_events is not None:
            if number(row, "compression_events") != expected_events:
                failures.append(
                    f"{row['result_file']}: compression event count mismatch"
                )
            destination_blocks = number(row, "destination_blocks_max")
            if destination_blocks is None or destination_blocks > 8:
                failures.append(
                    f"{row['result_file']}: destination block count exceeded 8"
                )

    grouped: dict[tuple[str, ...], dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[cell_key(row)][row["mode"]].append(row)

    comparisons: list[dict[str, Any]] = []
    decode_speedup_publishable = True
    decode_comparisons = 0
    for key, modes in grouped.items():
        if set(modes) != {"B", "C", "P", "CP"}:
            failures.append(f"cell {key} does not contain all B/C/P/CP modes")
            continue
        mode_metrics = {
            mode: {
                field: mean(
                    value
                    for row in mode_rows
                    if (value := number(row, field)) is not None
                )
                for field in ("ttft_mean_ms", "tpot_mean_ms", "output_throughput", "f1")
                if any(number(row, field) is not None for row in mode_rows)
            }
            for mode, mode_rows in modes.items()
        }
        comparison: dict[str, Any] = {"cell": key, "metrics": mode_metrics}
        if "ttft_mean_ms" in mode_metrics["C"] and mode_metrics["C"]["ttft_mean_ms"]:
            ratio = (
                mode_metrics["CP"]["ttft_mean_ms"] / mode_metrics["C"]["ttft_mean_ms"]
            )
            comparison["cp_to_c_ttft_ratio"] = ratio
            if ratio > 1.10:
                failures.append(f"cell {key}: CP TTFT is {ratio:.3f}x C")
            if mode_metrics["CP"]["ttft_mean_ms"] >= mode_metrics["B"]["ttft_mean_ms"]:
                failures.append(f"cell {key}: CP TTFT is not below B")
        if key[0] == "rag" and "f1" in mode_metrics["B"]:
            f1_drop = mode_metrics["B"]["f1"] - mode_metrics["CP"]["f1"]
            comparison["cp_f1_drop"] = f1_drop
            if f1_drop > 3:
                failures.append(f"cell {key}: CP F1 drop is {f1_drop:.3f}")
            baseline_groups = mean_group_metrics(modes["B"], "f1_by_group")
            cp_groups = mean_group_metrics(modes["CP"], "f1_by_group")
            for group in sorted(set(baseline_groups) & set(cp_groups)):
                group_drop = baseline_groups[group] - cp_groups[group]
                if group_drop > 5:
                    failures.append(
                        f"cell {key}: CP F1 drop for group {group} is {group_drop:.3f}"
                    )
        if key[0] == "fixed" and key[1] == "7168":
            decode_comparisons += 1
            repeat_ratios = []
            c_by_repeat = {row["repeat"]: row for row in modes["C"]}
            cp_by_repeat = {row["repeat"]: row for row in modes["CP"]}
            for repeat in sorted(set(c_by_repeat) & set(cp_by_repeat)):
                c_tpot = number(c_by_repeat[repeat], "tpot_mean_ms")
                cp_tpot = number(cp_by_repeat[repeat], "tpot_mean_ms")
                if c_tpot and cp_tpot is not None:
                    repeat_ratios.append(cp_tpot / c_tpot)
            stable = len(repeat_ratios) >= 3 and all(
                ratio <= 0.90 for ratio in repeat_ratios
            )
            comparison["decode_tpot_ratios"] = repeat_ratios
            comparison["stable_decode_improvement_10pct"] = stable
            decode_speedup_publishable &= stable
        comparisons.append(comparison)

    return {
        "passed": not failures,
        "failures": failures,
        "decode_speedup_publishable": bool(decode_comparisons)
        and decode_speedup_publishable,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.summary_csv.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    report = validate_rows(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["failures"]:
        raise SystemExit("result matrix failed release gates; see " + str(args.output))


if __name__ == "__main__":
    main()
