    to_physical_map=eplb_layer.logical_to_physical_map,
    logical_count=eplb_layer.logical_count,
    scoring_func=self.scoring_func,
    out=out,
)
if self.use_mega_gate:
    topk_idx, topk_weights = deep_gemm.bf16_mega_gate(hidden_states, self.gate.weight, **gate_kwargs)
else:
    topk_idx, topk_weights = moe_topk_gate(self.gate(hidden_states), **gate_kwargs)
