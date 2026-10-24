(
    logprobs,
    token_index_cur,
    candidate_token_ids,
    accept_result_verify,
    draft_selected_logprobs,
    accept_probs,
) = process_sampling(
    logits,
    driver_inputs.kernel,
    self.force_topk,
    row_to_req=row_to_req,
    local_step=local_step,
    is_verify=is_verify,
    last_nextn_token_logprobs=mtp_info_pool.logprobs if speculative_inputs is not None else None,
    last_nextn_token_logprobs_index=mtp_info_pool.logprobs_index if speculative_inputs is not None else None,
    draft_token_ids_pool=mtp_info_pool.token_ids if mtp_info_pool is not None else None,
    mtp_info_read_rows=speculative_inputs.mtp_info_read_rows if speculative_inputs is not None else None,
)
