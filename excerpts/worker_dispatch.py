def _execute_model_async_step(
    self, seq_metadata_list: list[SequenceMetadata], scheduler_outputs: SchedulerOutputs
) -> np.ndarray[np.uint64]:
    with cpu_timer.record("run_workers"):
        request_outputs = self._run_workers(
            "execute_model_step",
            driver_kwargs=dict(
                seq_metadata_list=seq_metadata_list,
                scheduler_outputs=scheduler_outputs,
            ),
            is_prompt=scheduler_outputs.prompt_run,
            is_placeholder=scheduler_outputs.is_empty(),
        )
    self.deferred_release.collect_request_outputs(request_outputs)
    return request_outputs
