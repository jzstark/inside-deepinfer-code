down_in_hidden = swiglu_forward_and_per_token_cast(
    gate_up_proj_out,
    **_get_swiglu_kwargs(self.use_fp8_act),
    psum_num_tokens_per_expert=psum_num_recv_tokens_per_expert,
    topk_weights=topk_weights,
    clamp_value=self.swiglu_clamp_value,
)
