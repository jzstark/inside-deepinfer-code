V41_FORMAT = KVCacheFormat(
    quantized_dtype="e4m3",
    quantized_elements=512,
    elements_per_sf=32,
    bf16_elements=0,
    do_cast=True,
    sf_format="ue8m0",
    note="V41, main/swa kv.",
    token_interleaved=True,
)
