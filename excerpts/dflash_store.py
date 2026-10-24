self.mtp_info_pool.store(
    sampling_input.mtp_info_write_slots,
    prediction,
    next_n_token_logprobs,
    next_n_token_logprobs_index,
)
return prediction
