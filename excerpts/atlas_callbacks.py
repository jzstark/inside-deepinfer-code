driver_kwargs = {
    "engine_callbacks": EngineCallbacks(post_process_fn=self.post_process, pre_sampler_sync_fn=self.pre_sampler_sync),
}
