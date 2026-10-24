window_size, num_moe_layers, num_local_physical_experts = self._expert_count_window.shape
num_physical_experts = self.physical_to_logical_map.shape[1]
num_logical_experts = self.logical_count.shape[1]
local_physical_to_logical_map = self.physical_to_logical_map[
    :,
    ep_group.rank() * num_local_physical_experts : (ep_group.rank() + 1) * num_local_physical_experts,
]
logical_expert_token_counts = torch.zeros(
    window_size,
    num_moe_layers,
    num_logical_experts,
    dtype=torch.int,
    device=local_physical_to_logical_map.device,
)
logical_expert_token_counts.scatter_add_(
    -1,
    local_physical_to_logical_map.to(torch.int64).expand_as(self._expert_count_window),
    self._expert_count_window,
)
dist.all_reduce(logical_expert_token_counts, group=ep_group)

next_physical_to_logical_map, next_logical_to_physical_map, next_logical_count = rebalance_experts_global(
    logical_expert_token_counts,
    num_physical_experts,
    num_local_physical_experts,
)
