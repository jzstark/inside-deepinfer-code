    compact_inputs, full_inputs = compact_yoco_boundary(
        hidden_states, residual, input_metadata, cross_layer_runtime, hc_enabled=self.hc_enabled
    )
    cross_layer_runtime.consumer_full_hidden = self.attn_norm_forward(*full_inputs)[1]
    hidden_states, residual = compact_inputs
    del compact_inputs, full_inputs
hidden_states, hidden_states_bf16, residual = self.attn_norm_forward(hidden_states, residual, hc_post_intervention)
