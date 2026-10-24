outputs: dict[str, list[RequestOutput]] = defaultdict(list)
while True:
    unfinished = self.llm_engine.has_unfinished_requests()
    step_outputs = self.llm_engine.step() if unfinished else self.llm_engine.flush_inflight_reqs()
    for raw_output in step_outputs:
        # clone step_outputs since step_outputs will be released in the next step
        output = RequestOutput.clone(raw_output)
        outputs[output.request_id].append(output)
        if output.finished and use_tqdm:
            pbar.update(1)
    if not unfinished:
        break
request_ids = sorted(outputs, key=int)
