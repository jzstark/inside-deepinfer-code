q_a, kv_swa = self._project_q_and_swa(
    hidden_states,
    shared_hidden_bf16,
)
del hidden_states, hidden_states_bf16, shared_hidden_bf16
kv_swa, wait_kv_swa = self._prepare_swa_kv(
    kv_swa,
    kv_cache.swa_cache,
    input_metadata,
    use_cp,
)

q_b, index_q_b = self._project_q_b(q_a, input_metadata.positions)
del q_a
