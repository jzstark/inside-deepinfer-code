def run_decode_step(*, measure: bool) -> None:
    countdown_before = lb_flush_countdown() if measure else None
    llm.llm_engine.step()
    if measure and len(llm.llm_engine.scheduler.running) == args.decode_num_requests:
        flushed = countdown_before is not None and lb_flush_countdown() > countdown_before
        stable_decode.append((time.perf_counter(), flushed))
