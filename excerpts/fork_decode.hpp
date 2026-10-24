if (forked) {
    if ((state_manager && !state_manager->can_allocate_single_nolock()) || !block_manager->can_allocate_single_nolock()) {
        break;
    }
    _clone_state(*seq);
    int new_block = block_manager->allocate_single_nolock();
    int num_blocks = (seq->cache.num_tokens + seq->cache.block_size - 1) / seq->cache.block_size;
    TORCH_CHECK_LE(num_blocks, static_cast<int>(seq->cache.block_table.size()));
    clone_blocks.emplace_back(seq->cache.block_table[num_blocks - 1]);
    clone_blocks.emplace_back(new_block);
    for (size_t i = num_blocks; i < seq->cache.block_table.size(); ++i) {
        clone_blocks.emplace_back(seq->cache.block_table[i]);
        clone_blocks.emplace_back(0);
    }
    seq->cache.block_table[num_blocks - 1] = new_block;
    seq->cache.block_table.erase(seq->cache.block_table.begin() + num_blocks, seq->cache.block_table.end());
}
