if self.clone_unpins_to_release is not None:
    states_to_unpin, blocks_to_unpin = self.clone_unpins_to_release
    self.clone_unpins_to_release = None
    self.clone_done_evt.synchronize()
    if states_to_unpin.size > 0:
        self.state_manager.unpin_blocks(states_to_unpin)
    if blocks_to_unpin.size > 0:
        self.block_manager.unpin_blocks(blocks_to_unpin)
