block_manager->pin_blocks(fork_seq->cache.block_table);
if (state_manager) {
    fork_seq->cache.state_id = seq->cache.state_id;
    state_manager->pin_block(seq->cache.state_id);
}
