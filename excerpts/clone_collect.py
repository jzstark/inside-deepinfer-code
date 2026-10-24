def collect_clone_unpins(self, states_to_unpin: np.ndarray[np.int32], blocks_to_unpin: np.ndarray[np.int32]) -> None:
    assert self.clone_unpins_to_release is None
    self.clone_done_evt.record()
    self.clone_unpins_to_release = (states_to_unpin, blocks_to_unpin)
