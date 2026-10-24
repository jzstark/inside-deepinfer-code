if run_cuda_graph:
    model_output = self.cuda_graph_runner.replay(cuda_graph_used)
else:
    model_output = self.model(kv_caches=kv_caches, input_metadata=input_metadata)
