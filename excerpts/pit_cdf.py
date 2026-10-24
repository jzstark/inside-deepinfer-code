pivot = logprobs.gather(1, sample_results.unsqueeze(1))
mask_less = logprobs < pivot
col_idx = torch.arange(logprobs.shape[1], device=logprobs.device).unsqueeze(0)
mask_equal_before = (logprobs == pivot) & (col_idx <= sample_results.unsqueeze(1))
out = (logprobs.exp() * (mask_less | mask_equal_before)).sum(dim=1)
out.masked_fill_(invalid, 0.0)
return out
