# SPDX-License-Identifier: Apache-2.0
"""Structured CacheBlend and PyramidKV request evidence."""

# Standard
from dataclasses import asdict, dataclass
from typing import Any, Mapping
import json


@dataclass(frozen=True)
class PendingCompressionRequest:
    request_id: str
    transaction_id: int
    lmcache_hit_tokens: int
    cacheblend_recomputed_tokens: int


@dataclass(frozen=True)
class CompressionRequestEvent:
    request_id: str
    transaction_id: int
    lmcache_hit_tokens: int
    cacheblend_recomputed_tokens: int
    semantic_tokens: int
    physical_tokens: int
    source_blocks: int
    destination_blocks: int
    released_blocks: int
    compression_ms: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def cacheblend_recomputed_tokens(config: Any, lmcache_hit_tokens: int) -> int:
    """Return CacheBlend's configured top-k size for a loaded prefix."""
    if lmcache_hit_tokens <= 0:
        return 0
    ratios = getattr(config, "blend_recompute_ratios", None)
    if not ratios:
        return 0
    ratio = ratios[0] if isinstance(ratios, (list, tuple)) else ratios
    ratio = float(ratio)
    if ratio <= 0:
        return 0
    return max(int(lmcache_hit_tokens * ratio), 1)


def disable_compression_transaction_stores(
    metadata: Any,
    transactions: Mapping[str, int],
) -> dict[str, Any]:
    """Disable LMCache stores and return transaction request metadata."""
    requests = {}
    for request in metadata.requests:
        transaction_id = transactions.get(request.req_id)
        if transaction_id is None:
            continue
        if request.save_spec is not None:
            request.save_spec.can_save = False
        request.kv_cache_compression_transaction_id = transaction_id
        requests[request.req_id] = request
    return requests


def merge_commit_event(
    pending: PendingCompressionRequest,
    commit: Any,
) -> CompressionRequestEvent:
    if pending.request_id != commit.request_id:
        raise ValueError("compression request IDs do not match")
    if pending.transaction_id != commit.transaction_id:
        raise ValueError("compression transaction IDs do not match")
    return CompressionRequestEvent(
        request_id=pending.request_id,
        transaction_id=pending.transaction_id,
        lmcache_hit_tokens=pending.lmcache_hit_tokens,
        cacheblend_recomputed_tokens=pending.cacheblend_recomputed_tokens,
        semantic_tokens=commit.semantic_tokens,
        physical_tokens=commit.physical_tokens,
        source_blocks=len(commit.source_block_ids),
        destination_blocks=len(commit.destination_block_ids),
        released_blocks=len(commit.released_block_ids),
        compression_ms=float(commit.compression_ms),
    )
