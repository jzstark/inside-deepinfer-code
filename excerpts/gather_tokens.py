"""Gather input_ids from fresh_tokens or prev sampled/predict tokens.

For each (seq i, token j) with j < seqlens_q[i], let q_base = cu_seqlens_q[i-1]
(0 for i == 0):
    if remap[i] >= 0 (old seq):
        j == 0: out[q_base+j] = prev_sampled_token_ids[remap[i]]
        j >  0: out[q_base+j] = prev_predict_token_ids[remap[i], j-1]
    else (fresh seq):
        out[q_base+j] = fresh_tokens[i, j]
"""
