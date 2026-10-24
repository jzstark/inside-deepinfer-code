score = tokens_per_expert_all_diff / logical_count  # NOTE: Values in score must be different from each other
score_added = tokens_per_expert / (logical_count + 1)
values, indices = score.max(-1, keepdim=True)
values = values.expand_as(score).contiguous()
score.scatter_(-1, indices, score_added.gather(-1, indices))
values.scatter_(-1, indices, score.max(-1, keepdim=True).values)
