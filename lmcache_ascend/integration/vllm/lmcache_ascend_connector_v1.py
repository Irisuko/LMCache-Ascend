# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any

# Third Party
from lmcache.logging import init_logger
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

# First Party
from lmcache_ascend import _build_info
from lmcache_ascend.integration.vllm.compression_events import (
    PendingCompressionRequest,
    cacheblend_recomputed_tokens,
    disable_compression_transaction_stores,
    merge_commit_event,
)

if _build_info.__framework_name__ == "pytorch":
    # First Party
    import lmcache_ascend  # noqa: F401
elif _build_info.__framework_name__ == "mindspore":
    # First Party
    import lmcache_ascend.mindspore  # noqa: F401
else:
    raise ValueError("Unsupported Framework")

# Third Party
from lmcache.integration.vllm.lmcache_connector_v1 import LMCacheConnectorV1Dynamic

logger = init_logger(__name__)


class LMCacheAscendConnectorV1Dynamic(LMCacheConnectorV1Dynamic):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Any | None = None,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._pending_compression_events: dict[str, PendingCompressionRequest] = {}
        logger.info("Enabled LMCache CacheBlend/PyramidKV transaction coordination")

    def build_connector_meta(self, scheduler_output):
        meta = super().build_connector_meta(scheduler_output)
        transactions = (
            getattr(
                scheduler_output,
                "kv_cache_compression_transaction_ids",
                None,
            )
            or {}
        )
        transaction_requests = disable_compression_transaction_stores(
            meta, transactions
        )
        request_trackers = (
            getattr(self._lmcache_engine, "_request_trackers", {})
            if transactions
            else {}
        )
        for request_id, transaction_id in transactions.items():
            request = transaction_requests.get(request_id)
            load_spec = request.load_spec if request is not None else None
            lmcache_hit_tokens = 0
            if load_spec is not None and load_spec.can_load:
                lmcache_hit_tokens = max(
                    0,
                    load_spec.lmcache_cached_tokens - load_spec.vllm_cached_tokens,
                )
            else:
                tracker = request_trackers.get(request_id)
                if tracker is not None:
                    lmcache_hit_tokens = max(0, int(tracker.num_lmcache_cached_tokens))
            self._pending_compression_events[request_id] = PendingCompressionRequest(
                request_id=request_id,
                transaction_id=transaction_id,
                lmcache_hit_tokens=lmcache_hit_tokens,
                cacheblend_recomputed_tokens=(
                    cacheblend_recomputed_tokens(
                        self._lmcache_engine.config,
                        lmcache_hit_tokens,
                    )
                ),
            )
            logger.info(
                "Disabled LMCache store for PyramidKV transaction: "
                "request_id=%s transaction_id=%d lmcache_hit_tokens=%d "
                "worker_metadata_present=%s",
                request_id,
                transaction_id,
                lmcache_hit_tokens,
                request is not None,
            )

        commit_events = (
            getattr(
                scheduler_output,
                "kv_cache_compression_commit_events",
                None,
            )
            or []
        )
        for commit in commit_events:
            pending = self._pending_compression_events.pop(commit.request_id, None)
            if pending is None:
                raise RuntimeError(
                    "received PyramidKV commit event without LMCache "
                    f"transaction metadata for request {commit.request_id!r}"
                )
            event = merge_commit_event(pending, commit)
            logger.info("LMCACHE_PYRAMIDKV_EVENT %s", event.to_json())

        for request_id in scheduler_output.finished_req_ids:
            self._pending_compression_events.pop(request_id, None)
        return meta
