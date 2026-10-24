ftp = decode_inputs.mtp_first_token_positions
seqlen_q = decode_inputs.mtp_seqlens_q
row_to_req, local_step, is_verify = prepare_verification_metadata(ftp, seqlen_q, total_q, self.max_MTP_value)
speculative_inputs = driver_inputs.speculative

# Apply structured-output masks before sampling and draft verification.
if so_indices is not None:
    for step, indices in enumerate(so_indices):
        if len(indices):
            apply_token_bitmask_by_step_inplace(logits, so_bitmasks[step], indices, ftp, step)
