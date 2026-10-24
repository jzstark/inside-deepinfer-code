if confidence_rates is not None:
    self.confidence_head(
        hidden_states_before_norm[step_i :: self.max_MTP_value],
        w1,
        out=confidence_rates[:, step_i],
        step_i=step_i,
    )
_, token_index_next, draft_token_ids, _, _, _ = process_sampling(
    logits,
    SamplingKernelInputs(
        sampling_input.kernel_inputs.logits, sampling_input.kernel_inputs.seq_seeds, sample_positions_table[step_i]
    ),
    self.force_topk,
    logprobs_out=next_n_token_logprobs[:, step_i],
    tag="mtp",
)
