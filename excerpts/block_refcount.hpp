void pin_block(int block_id) {
    TORCH_CHECK(block_id > 0 && block_id < static_cast<int>(_blocks.size()));
    auto &block = _blocks[block_id];
    TORCH_CHECK(block.ref_count > 0, "try to pin free block");
    block.ref_count += 1;
}

void free(int block_id) {
    TORCH_CHECK(block_id > 0 && block_id < static_cast<int>(_blocks.size()));
    auto &block = _blocks[block_id];
    block.ref_count -= 1;
    TORCH_CHECK(block.ref_count >= 0, "ref_count corrupted: ref_count < 0");
    if (block.ref_count == 0) {
        _push_back(block_id);
    }
}
