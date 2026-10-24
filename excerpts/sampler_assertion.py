if len(all_quantiles) == 0:
    errors.append(f"DP {dp_rank} {config_name}: no valid trajectories for all, skip={skip_counts}")
elif all_pv is None:
    errors.append(f"DP {dp_rank} {config_name}: computing the all p_value failed")
elif all_pv <= 1e-5:
    errors.append(f"DP {dp_rank} {config_name}: all p_value={all_pv} is too small; the sampler may have a problem")
