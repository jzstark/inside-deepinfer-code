logits.mul_((1.0 / transform_inputs.temperatures).unsqueeze(1))
apply_top_p_top_k_ref(logits, transform_inputs.top_ps, transform_inputs.top_ks)
logprobs = torch.log_softmax(logits, dim=-1)
