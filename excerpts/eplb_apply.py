if self._intranode_shuffle:
    expert_model_parallel_shuffle_inplace_intranode(
        self._ipc_weight_buffers,
        self.physical_to_logical_map,
        self._expert_weights,
        next_physical_to_logical_map,
        ep_group,
    )
else:
    expert_model_parallel_shuffle_inplace(
        self.physical_to_logical_map,
        self._expert_weights,
        next_physical_to_logical_map,
        ep_group,
    )
self.physical_to_logical_map.copy_(next_physical_to_logical_map)
self.logical_to_physical_map.copy_(next_logical_to_physical_map)
self.logical_count.copy_(next_logical_count)
