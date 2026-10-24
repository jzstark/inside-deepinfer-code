assert not use_cp
cast_args = CastArgs(
    "e4m3",
    self.swa_cache_num_per_channels,
    round_sf=True,
    sf_format="ue8m0",
)
out_args = KVCacheOutArgs(
    swa_cache,
    input_metadata.slot_mapping_store_swa,
)
