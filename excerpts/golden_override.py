accept = raw_accept
tokens = candidate_token_ids
if golden is not None:
    use_golden = golden != -1
    accept = torch.where(use_golden, golden == draft_token_ids, accept)
    tokens = torch.where(use_golden, golden, tokens)
blocked = ~(is_verify & accept)
prefix = torch.cat([blocked.new_zeros(1, dtype=torch.int32), blocked.cumsum(0, dtype=torch.int32)])
alive = prefix[:-1] == prefix[ftp[row_to_req.long()].long()]
effective_accept = alive & is_verify & accept
