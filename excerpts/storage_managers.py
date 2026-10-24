self.block_manager = BlockManager(cache_config.num_gpu_blocks)
self.state_manager = BlockManager(cache_config.num_state_seqs) if cache_config.bytes_of_state_per_seqs > 0 else None
