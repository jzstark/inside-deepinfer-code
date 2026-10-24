if dflash_target_layer_ids:
    in_consumer = [t >= cfg.n_prefill_layer for t in dflash_target_layer_ids]
    assert all(in_consumer) or not any(in_consumer), (
        "yoco consumer compaction requires DFlash target layers on the same side of "
        f"n_prefill_layer ({cfg.n_prefill_layer}), got {dflash_target_layer_ids}"
    )
    self.dflash_targets_in_consumer = all(in_consumer)
assert all(layer_id < cfg.n_prefill_layer for layer_id in engram_layer_ids), (
    "yoco consumer compaction does not support Engram in consumer layers yet."
)
