# SPDX-License-Identifier: Apache-2.0
"""Merge raw client results, server events, and sampled service metrics."""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
from statistics import mean
from typing import Any, Iterable
import argparse
import csv
import json
import math
import re

LMCACHE_EVENT_MARKER = "LMCACHE_PYRAMIDKV_EVENT "
CORE_EVENT_MARKER = "VLLM_PYRAMIDKV_EVENT "
METRIC_RE = re.compile(r"^vllm:kv_cache_usage_perc(?:\{[^}]*\})?\s+([0-9.eE+-]+)$")
LMCACHE_REQUEST_RE = re.compile(
    r"Reqid: (?P<request_id>[^,]+), .*?LMCache hit tokens: (?P<hit>\d+), "
    r"(?:but )?need to load: (?P<load>\d+)"
)
JSON_DECODER = json.JSONDecoder()


def finite(values: Iterable[float | int | None]) -> list[float]:
    return [
        float(value) for value in values if value is not None and math.isfinite(value)
    ]


def percentile(values: Iterable[float | int | None], quantile: int) -> float | None:
    samples = sorted(finite(values))
    if not samples:
        return None
    position = (len(samples) - 1) * quantile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return samples[lower]
    return samples[lower] + (samples[upper] - samples[lower]) * (position - lower)


def metric_columns(
    prefix: str, values_s: Iterable[float | int | None]
) -> dict[str, float | None]:
    milliseconds = [value * 1000 for value in finite(values_s)]
    return {
        f"{prefix}_mean_ms": mean(milliseconds) if milliseconds else None,
        **{
            f"{prefix}_p{quantile}_ms": percentile(milliseconds, quantile)
            for quantile in (50, 90, 95, 99)
        },
    }


def load_events(server_log: Path) -> list[dict[str, Any]]:
    lmcache_events = []
    core_events = []
    if not server_log.exists():
        return lmcache_events
    for line in server_log.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = next(
            (
                candidate
                for candidate in (LMCACHE_EVENT_MARKER, CORE_EVENT_MARKER)
                if candidate in line
            ),
            None,
        )
        if marker is None:
            continue
        payload = line.split(marker, 1)[1].lstrip()
        event, _ = JSON_DECODER.raw_decode(payload)
        if not isinstance(event, dict):
            raise ValueError(f"compression event must be an object: {server_log}")
        target = lmcache_events if marker == LMCACHE_EVENT_MARKER else core_events
        target.append(event)
    # Combined mode logs both records for the same commit. Prefer the LMCache
    # record because it adds hit and CacheBlend recompute evidence.
    return lmcache_events or core_events


def peak_kv_usage(metrics_path: Path) -> float | None:
    samples = []
    if not metrics_path.exists():
        return None
    for line in metrics_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = METRIC_RE.match(line)
        if match:
            samples.append(float(match.group(1)))
    return max(samples, default=None)


def load_lmcache_request_stats(
    server_log: Path, request_ids: Iterable[str]
) -> dict[str, dict[str, int]]:
    """Read lookup/load counts for benchmark requests, excluding precompute."""
    if not server_log.exists():
        return {}
    internal_ids = {f"cmpl-{request_id}-0": request_id for request_id in request_ids}
    stats = {}
    for line in server_log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LMCACHE_REQUEST_RE.search(line)
        if match is None:
            continue
        internal_id = match.group("request_id")
        canonical_id = internal_id
        if internal_id not in internal_ids:
            prefix, separator, suffix = internal_id.rpartition("-")
            if separator and re.fullmatch(r"[0-9a-f]{8}", suffix):
                canonical_id = prefix
        request_id = internal_ids.get(canonical_id)
        if request_id is None:
            continue
        if request_id in stats:
            raise ValueError(f"duplicate LMCache lookup record for {request_id}")
        stats[request_id] = {
            "hit_tokens": int(match.group("hit")),
            "load_tokens": int(match.group("load")),
        }
    return stats


def aggregate_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported result schema in {path}")
    metadata = payload["metadata"]
    summary = payload["summary"]
    requests = payload["requests"]
    successes = [request for request in requests if request["error"] is None]

    ttfts = [request["ttft_s"] for request in successes]
    tpots = [request["tpot_s"] for request in successes]
    itls = [itl for request in successes for itl in request["itl_s"]]
    e2es = [request["e2e_s"] for request in successes]

    server_log = path.with_suffix(".server.log")
    metrics_path = path.with_suffix(".metrics")
    events = load_events(server_log)
    lmcache_stats = load_lmcache_request_stats(
        server_log, (request["request_id"] for request in requests)
    )
    log_text = (
        server_log.read_text(encoding="utf-8", errors="replace")
        if server_log.exists()
        else ""
    )
    row: dict[str, Any] = {
        "result_file": str(path),
        "workload": payload["benchmark_type"],
        **metadata,
        **summary,
        "success_rate": len(successes) / len(requests) if requests else 0.0,
        "precompute_documents": payload["precompute"]["documents"],
        "precompute_s": payload["precompute"]["duration_s"],
        "precompute_settle_s": payload["precompute"].get("settle_s", 0.0),
        "preemptions": len(re.findall(r"preempt", log_text, flags=re.IGNORECASE)),
        "peak_kv_usage": peak_kv_usage(metrics_path),
        "compression_events": len(events),
    }
    for field in ("f1_by_group", "em_by_group"):
        if isinstance(row.get(field), dict):
            row[field] = json.dumps(row[field], sort_keys=True, separators=(",", ":"))
    for field in (
        "lmcache_hit_tokens",
        "cacheblend_recomputed_tokens",
        "semantic_tokens",
        "physical_tokens",
        "source_blocks",
        "destination_blocks",
        "released_blocks",
        "compression_ms",
    ):
        values = finite(event.get(field) for event in events)
        row[f"{field}_total"] = sum(values) if values else 0
        row[f"{field}_mean"] = mean(values) if values else None
        row[f"{field}_max"] = max(values) if values else None

    compression_hits = finite(event.get("lmcache_hit_tokens") for event in events)
    row["compression_lmcache_hit_tokens_total"] = (
        sum(compression_hits) if compression_hits else 0
    )
    row["compression_lmcache_hit_tokens_mean"] = (
        mean(compression_hits) if compression_hits else None
    )
    row["compression_lmcache_hit_tokens_max"] = (
        max(compression_hits) if compression_hits else None
    )

    lookup_hits = finite(stat["hit_tokens"] for stat in lmcache_stats.values())
    requested_loads = finite(stat["load_tokens"] for stat in lmcache_stats.values())
    for field, values in (
        ("lmcache_lookup_hit_tokens", lookup_hits),
        ("lmcache_load_tokens", requested_loads),
    ):
        row[f"{field}_total"] = sum(values) if values else 0
        row[f"{field}_mean"] = mean(values) if values else None
        row[f"{field}_max"] = max(values) if values else None

    # The request log covers every request, while combined events cover only
    # compressed requests. Prefer the former for a uniform B/C/P/CP total.
    if lookup_hits:
        row["lmcache_hit_tokens_total"] = sum(lookup_hits)
        row["lmcache_hit_tokens_mean"] = mean(lookup_hits)
        row["lmcache_hit_tokens_max"] = max(lookup_hits)

    row.update(metric_columns("ttft", ttfts))
    row.update(metric_columns("tpot", tpots))
    row.update(metric_columns("itl", itls))
    row.update(metric_columns("e2e", e2es))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()

    paths = sorted(args.raw_dir.glob("*.json"))
    paths = [path for path in paths if path.name != "sample_indices.json"]
    if not paths:
        raise SystemExit(f"no raw JSON results found in {args.raw_dir}")
    rows = [aggregate_result(path) for path in paths]
    fields = list(dict.fromkeys(field for row in rows for field in row))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
