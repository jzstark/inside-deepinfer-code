if not scheduler_outputs.is_empty() and self.engine_type == EngineType.PROMPT:
    assert scheduler_outputs.prompt_run, "Prompt engines should not perform decoding"
    assert seq_metadata_list[0].seqlen_q > 0, "seqlen_q should be > 0"
