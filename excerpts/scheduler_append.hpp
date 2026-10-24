void _append_slot(Sequence &seq) {
    int num_blocks = seq.get_num_blocks();
    if (static_cast<decltype(num_blocks)>(seq.cache.block_table.size()) < num_blocks) {
        seq.cache.block_table.emplace_back(block_manager->allocate_single_nolock());
    } else if (static_cast<decltype(num_blocks)>(seq.cache.block_table.size()) > num_blocks) {
        std::vector<int> redundant_blocks(seq.cache.block_table.begin() + num_blocks, seq.cache.block_table.end());
        block_manager->free_blocks_nolock(redundant_blocks);
        seq.cache.block_table.erase(seq.cache.block_table.begin() + num_blocks, seq.cache.block_table.end());
    }
    TORCH_CHECK(static_cast<decltype(num_blocks)>(seq.cache.block_table.size()) == num_blocks,
        "len(block_table) = ", seq.cache.block_table.size(), ", but should be ", num_blocks, ". seq_len = ", seq.get_len());
}
