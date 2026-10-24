with cpu_timer.record("worker.post_process.sync"):
    step_state.sample_end_evt.synchronize()
with cpu_timer.record("worker.post_process.graph_timer_flush"):
    # flush the time of completed graph, no effect on the running graph.
    cuda_graph_timer.flush_out()
