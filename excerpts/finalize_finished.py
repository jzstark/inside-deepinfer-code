if self.engine_type in [EngineType.COMPLETION, EngineType.NORMAL]:
    for seq in scheduler_outputs.finished_seqs:
        self.finalize_seq(seq)
