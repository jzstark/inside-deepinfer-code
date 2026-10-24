assert start % 32 == 0
if blocks is not None:
    swa_cache = blocks[:, start : start + swa_size].as_strided(
        (blocks.shape[0], swa_block_size, 1, swa_size_per_token), (blocks.stride(0), swa_size_per_token, swa_size_per_token, 1)
    )
