row_to_req = torch.repeat_interleave(
    torch.arange(ftp.shape[0], dtype=ftp.dtype, device=ftp.device),
    seqlen_q,
    output_size=total_q,
)
local_step = torch.arange(total_q, dtype=ftp.dtype, device=ftp.device) - ftp[row_to_req.long()]
is_verify = local_step != seqlen_q[row_to_req.long()] - 1
return row_to_req, local_step, is_verify
