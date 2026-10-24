ssp_arr = np.array(ssp, dtype=np.float64)
p_arr = np.exp(np.array(log_probs, dtype=np.float64))
if not (np.all(np.isfinite(ssp_arr)) and np.all(np.isfinite(p_arr)) and np.all(p_arr > 0)):
    return None
if (ssp_arr - p_arr < -0.01).sum() > 0:
    return None
rng = np.random.RandomState(seed)
u = np.clip(ssp_arr - p_arr * rng.uniform(0, 1, size=n), 0, 1)
ks_stat, ks_quantile = ks_test_uniform(u.tolist())
if ks_stat is None:
    return None
return ks_stat, ks_quantile, n
