q_a_kv_swa = self.q_a_kv_swa_proj(hidden_states)
if self.yoco_swa_share_hidden:
    assert shared_hidden_bf16 is not None, f"layer {self.layer_idx}: yoco_swa_share_hidden requires shared_hidden_bf16"
    return q_a_kv_swa, self.kv_swa_proj(shared_hidden_bf16)
if self.use_swa:
    return q_a_kv_swa.split([self.q_lora_rank, self.qk_head_dim], dim=-1)
return q_a_kv_swa, None
