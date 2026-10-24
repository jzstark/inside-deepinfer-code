sampling_params = copy.deepcopy(sampling_params)
request_id = str(next(self.request_counter))
if prompt_token_ids is None:
    if prompt is None:
        raise ValueError("prompt and prompt_token_ids cannot both be None")
    prompt_token_ids = self.llm_engine.tokenizer.encode(prompt)
prompt_token_ids = np.asarray(prompt_token_ids, dtype=np.int32)
