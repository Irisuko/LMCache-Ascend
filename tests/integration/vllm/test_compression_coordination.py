# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest
import torch


def _events_module():
    return pytest.importorskip("lmcache_ascend.integration.vllm.compression_events")


def test_compression_transaction_disables_store_without_affecting_short_request():
    events = _events_module()
    compressed = SimpleNamespace(
        req_id="compressed",
        save_spec=SimpleNamespace(can_save=True),
    )
    short = SimpleNamespace(
        req_id="short",
        save_spec=SimpleNamespace(can_save=True),
    )
    metadata = SimpleNamespace(requests=[compressed, short])

    matched = events.disable_compression_transaction_stores(metadata, {"compressed": 7})

    assert matched == {"compressed": compressed}
    assert not compressed.save_spec.can_save
    assert compressed.kv_cache_compression_transaction_id == 7
    assert short.save_spec.can_save
    assert not hasattr(short, "kv_cache_compression_transaction_id")


def test_structured_event_contains_all_required_fields():
    events = _events_module()
    pending = events.PendingCompressionRequest(
        request_id="request",
        transaction_id=3,
        lmcache_hit_tokens=7040,
        cacheblend_recomputed_tokens=1056,
    )
    commit = SimpleNamespace(
        request_id="request",
        transaction_id=3,
        semantic_tokens=7168,
        physical_tokens=991,
        source_block_ids=tuple(range(56)),
        destination_block_ids=tuple(range(8)),
        released_block_ids=tuple(range(8, 56)),
        compression_ms=4.25,
    )

    event = events.merge_commit_event(pending, commit)
    payload = event.to_json()

    assert event.source_blocks == 56
    assert event.destination_blocks == 8
    assert event.released_blocks == 48

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
        assert f'"{field}"' in payload


def test_external_hit_cap_preserves_cacheblend_segment_boundary_load():
    pytest.importorskip("lmcache")
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")
    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    adapter.load_specs = {
        "request": SimpleNamespace(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=7168,
            can_load=False,
        )
    }
    request = SimpleNamespace(request_id="request", num_tokens=7168)

    base_update = MagicMock()
    parent = adapter_mod.LMCacheConnectorV1Impl
    original = parent.update_state_after_alloc
    parent.update_state_after_alloc = base_update
    try:
        adapter.update_state_after_alloc(request, 7040)
    finally:
        parent.update_state_after_alloc = original

    assert adapter.load_specs["request"].lmcache_cached_tokens == 7168
    base_update.assert_called_once_with(request, 7167)


def test_partial_prefill_metadata_keeps_lmcache_load_span():
    pytest.importorskip("lmcache")
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")
    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    adapter._block_size = 128
    adapter._request_trackers = {
        "request": SimpleNamespace(
            token_ids=list(range(124)),
            allocated_block_ids=[3],
        )
    }
    request = SimpleNamespace(
        req_id="request",
        token_ids=[],
        slot_mapping=torch.empty(0, dtype=torch.long),
        load_spec=SimpleNamespace(lmcache_cached_tokens=41, can_load=True),
    )
    metadata = adapter_mod.LMCacheConnectorMetadata(requests=[request])

    parent = adapter_mod.LMCacheConnectorV1Impl
    original = parent.build_connector_meta
    parent.build_connector_meta = MagicMock(return_value=metadata)
    try:
        result = adapter.build_connector_meta(SimpleNamespace())
    finally:
        parent.build_connector_meta = original

    assert result is metadata
    assert request.token_ids == list(range(41))
    assert request.slot_mapping.tolist() == list(range(3 * 128, 3 * 128 + 41))


def test_legacy_cacheblend_scheduler_output_needs_no_compression_fields():
    pytest.importorskip("lmcache")
    connector_mod = pytest.importorskip(
        "lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"
    )
    connector = object.__new__(connector_mod.LMCacheAscendConnectorV1Dynamic)
    connector._pending_compression_events = {}
    metadata = SimpleNamespace(requests=[])
    scheduler_output = SimpleNamespace(finished_req_ids=set())

    parent = connector_mod.LMCacheConnectorV1Dynamic
    original = parent.build_connector_meta
    parent.build_connector_meta = MagicMock(return_value=metadata)
    try:
        result = connector.build_connector_meta(scheduler_output)
    finally:
        parent.build_connector_meta = original

    assert result is metadata
    assert connector._pending_compression_events == {}


def test_compression_event_uses_tracker_when_worker_metadata_is_empty():
    pytest.importorskip("lmcache")
    connector_mod = pytest.importorskip(
        "lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"
    )
    connector = object.__new__(connector_mod.LMCacheAscendConnectorV1Dynamic)
    connector._pending_compression_events = {}
    connector._lmcache_engine = SimpleNamespace(
        config=SimpleNamespace(blend_recompute_ratios=[0.15]),
        _request_trackers={"request": SimpleNamespace(num_lmcache_cached_tokens=7037)},
    )
    metadata = SimpleNamespace(requests=[])
    transaction_output = SimpleNamespace(
        kv_cache_compression_transaction_ids={"request": 3},
        kv_cache_compression_commit_events=None,
        finished_req_ids=set(),
    )
    commit_output = SimpleNamespace(
        kv_cache_compression_transaction_ids=None,
        kv_cache_compression_commit_events=[
            SimpleNamespace(
                request_id="request",
                transaction_id=3,
                semantic_tokens=7168,
                physical_tokens=991,
                source_block_ids=tuple(range(56)),
                destination_block_ids=tuple(range(8)),
                released_block_ids=tuple(range(8, 56)),
                compression_ms=4.25,
            )
        ],
        finished_req_ids=set(),
    )

    parent = connector_mod.LMCacheConnectorV1Dynamic
    original = parent.build_connector_meta
    parent.build_connector_meta = MagicMock(return_value=metadata)
    try:
        connector.build_connector_meta(transaction_output)
        pending = connector._pending_compression_events["request"]
        assert pending.lmcache_hit_tokens == 7037
        connector.build_connector_meta(commit_output)
    finally:
        parent.build_connector_meta = original

    assert connector._pending_compression_events == {}


def test_connector_forwards_vllm_kv_cache_config():
    pytest.importorskip("lmcache")
    connector_mod = pytest.importorskip(
        "lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"
    )
    connector = object.__new__(connector_mod.LMCacheAscendConnectorV1Dynamic)
    vllm_config = object()
    role = object()
    kv_cache_config = object()

    parent = connector_mod.LMCacheConnectorV1Dynamic
    original = parent.__init__
    parent.__init__ = MagicMock()
    try:
        connector.__init__(vllm_config, role, kv_cache_config)
    finally:
        mocked_init = parent.__init__
        parent.__init__ = original

    mocked_init.assert_called_once_with(
        vllm_config=vllm_config,
        role=role,
        kv_cache_config=kv_cache_config,
    )


def test_config_patch_refreshes_preloaded_vllm_adapter_reference():
    pytest.importorskip("lmcache")
    # Third Party
    import lmcache.integration.vllm.utils as lmcache_utils
    import lmcache.integration.vllm.vllm_v1_adapter as upstream_adapter
    import lmcache.v1.config as config_module

    # First Party
    import lmcache_ascend

    stale_class = object()
    stale_instance = object()
    upstream_adapter.LMCacheEngineConfig = stale_class
    lmcache_utils.LMCacheEngineConfig = stale_class
    lmcache_utils._config_instance = stale_instance

    lmcache_ascend._refresh_config_class_references(config_module.LMCacheEngineConfig)

    assert upstream_adapter.LMCacheEngineConfig is config_module.LMCacheEngineConfig
    assert lmcache_utils.LMCacheEngineConfig is config_module.LMCacheEngineConfig
    assert lmcache_utils._config_instance is None


def test_blend_attention_imports_current_vllm_attention_layer():
    pytest.importorskip("lmcache")
    blend_attention = pytest.importorskip("lmcache_ascend.v1.blend.attention.attention")
    # Third Party
    from vllm.model_executor.layers.attention import Attention

    assert blend_attention.Attention is Attention


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        ([torch.tensor([1, 2]), torch.tensor([])], [(0, 2, (1, 2))]),
        (
            [torch.tensor([]), torch.tensor([3]), torch.tensor([])],
            [(2, 3, (3,))],
        ),
    ],
)
def test_blend_token_database_skips_empty_separator_segments(chunks, expected):
    token_database = pytest.importorskip("lmcache_ascend.v1.token_database")
    database = SimpleNamespace(
        sep_len=2,
        _fast_split_by_subtensor=lambda _tokens: iter(chunks),
        _hash_tokens=lambda tokens: tuple(int(token) for token in tokens),
    )

    result = list(
        token_database.TokenDatabase_process_tokens(
            database,
            tokens=[1, 2, 3, 4, 5],
            make_key=False,
        )
    )

    assert result == expected


def test_wait_for_save_accepts_already_exhausted_layerwise_generator():
    pytest.importorskip("lmcache")
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")
    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)

    def exhausted():
        if False:
            yield None

    request = SimpleNamespace(req_id="request")
    adapter._parent = SimpleNamespace(
        _get_connector_metadata=lambda: adapter_mod.LMCacheConnectorMetadata(
            requests=[request]
        )
    )
    adapter.kv_role = "kv_both"
    engine = SimpleNamespace(
        _is_passive=lambda: False,
        lookup_unpin=MagicMock(),
    )
    adapter._manager = SimpleNamespace(lmcache_engine=engine)
    adapter.use_layerwise = True
    adapter.store_async = False
    adapter._layerwise_save_storers = {"request": exhausted()}
    adapter._replay_finished_stores_after_save = MagicMock()

    adapter.wait_for_save()

    engine.lookup_unpin.assert_called_once_with("request")
    assert adapter._wait_for_save_done


@pytest.mark.parametrize("method_name", ["embed_input_ids", "get_input_embeddings"])
def test_llama_blend_supports_current_and_legacy_embedding_apis(method_name):
    llama = pytest.importorskip("lmcache_ascend.v1.blend.models.llama")
    expected = torch.tensor([[1.0]])
    model = SimpleNamespace(**{method_name: MagicMock(return_value=expected)})

    result = llama._embed_input_ids(model, torch.tensor([1]))

    assert result is expected
    getattr(model, method_name).assert_called_once()
