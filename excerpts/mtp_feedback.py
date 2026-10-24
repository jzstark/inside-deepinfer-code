if step_state.forward_events is not None and step_state.sample_events is not None:
    # Worker has synchronized this step's sample_end_evt before calling post_process.
    self.scheduler.update_mtp_runtime(
        len(seq_metadata_list),
        step_state.graph_size,
        step_state.forward_events[0].elapsed_time(step_state.forward_events[1]),
        step_state.sample_events[0].elapsed_time(step_state.sample_events[1]),
    )
