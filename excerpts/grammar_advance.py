num_accepted_steps = 0
for k, draft_token in enumerate(seq_metadata.last_next_n_token):
    if matcher.accept_token(draft_token):
        if matcher.constrained():
            mask_filled = matcher.fill_vocab_mask(self.bitmasks_cpu[k + 1], idx)
