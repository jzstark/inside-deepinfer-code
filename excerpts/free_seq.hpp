void free_seq_nolock(Sequence &seq) {
    release_mtp_info_slot(seq);
    if (seq.cache.state_id != -1) {
        state_manager->free_blocks_nolock({seq.cache.state_id});
    }
    block_manager->free_blocks_nolock(seq.cache.block_table);
    seq.cache.block_table.clear();
}
