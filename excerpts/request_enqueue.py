def _enqueue(s: Sequence) -> None:
    if s.so_matcher is not None and s.so_matcher.is_aborted():
        s.status = SequenceStatus.FINISHED_SO_FAILED
    self.scheduler.add_sequence(s)
    if self.multi_modal_processor is not None:
        self.multi_modal_processor.add(request_id, multi_modal_data)

if so_pending:
    self.so_matcher_builder.submit(seq, sampling_params.so_args, _enqueue)
else:
    _enqueue(seq)
