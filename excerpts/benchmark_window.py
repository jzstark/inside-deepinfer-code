for _ in range(decode_warmup_steps):
    if not llm.llm_engine.has_unfinished_requests():
        break
    run_decode_step(measure=False)
set_engine_log_interval(20)
for _ in range(decode_measure_steps):
    if not llm.llm_engine.has_unfinished_requests():
        break
    run_decode_step(measure=False)
set_engine_log_interval(0)
while len(stable_decode) < stable_decode.maxlen and llm.llm_engine.has_unfinished_requests():
    run_decode_step(measure=True)
