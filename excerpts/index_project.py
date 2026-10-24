if not self.store_index_k:
    return
assert index_k_input is not None
assert state_cache is None
index_k = self.index_k_module.forward_pre_norm(index_k_input)
norm = self.index_k_module.norm
fused_norm_rope_hadamard_cast_store_kvcache(
    index_k,
    norm_args=NormArgs(norm.weight, norm.variance_epsilon),
