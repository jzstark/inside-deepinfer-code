# topk outputs the logical index of the compressed token within the seq, which is converted in place into the kvcache
# physical slot through block_tables (-1 is kept)
convert_kv_index(
    topk_index.unsqueeze(1),
    compressed_block_size,
    topk_lens.unsqueeze(1),
    input_metadata.block_tables,
)
return topk_index
