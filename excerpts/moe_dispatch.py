if self.use_mega_moe:
    return self.mega_moe_forward(hidden_states, hidden_states_bf16, input_metadata)

topk_idx, topk_weights = self.gate_forward(hidden_states_bf16, input_metadata)

try:
    # Dispatch
    event_overlap = self.dispatch(
        hidden_states,
        topk_idx,
        topk_weights,
        input_metadata,
        None,
    )
    moe_in_hidden, _dispatched_topk_idx, dispatched_topk_weights, handle = event_overlap.current_stream_wait()
