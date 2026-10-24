FP4_INDEX_KV_FORMAT = KVCacheFormat(
    quantized_dtype="e2m1",
    quantized_elements=128,
    elements_per_sf=32,
    bf16_elements=0,
    do_cast=False,
    sf_format="ue8m0",
    note="V4/V41, index kv.",
)
