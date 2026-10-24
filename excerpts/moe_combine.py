# Combine
combine_pre_evt = self.ep_buffer.capture()
with limit_num_sms(envs.DEEPINFER_USERBUFFER_COMM_SMS, enabled=PLATFORM.is_gpu):
    shared_out = self.shared_forward(hidden_states)
event_overlap = self.combine(
    moe_out_hidden,
    shared_out,
    handle,
    combine_pre_evt,
)
moe_combined_hidden, _combined_topk_weights = event_overlap.current_stream_wait()
