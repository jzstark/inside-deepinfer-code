seq_metadata_list, scheduler_outputs = self.scheduler.schedule(
    self.sequence_metadata_pool
)  # Will also free the finished sequence groups.
