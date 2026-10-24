for token_ids, output in zip(prompt_token_ids, outputs, strict=True):
    sp = copy.copy(sampling_params)
    sp.golden_token_ids = [x.token_id for x in output]
    output_single = llm.generate(None, sp, [token_ids], use_tqdm=False)[0]
    result_batched = _get_result(output)
    result_single = _get_result(output_single)
    assert result_batched == result_single, f"{dp_rank=}, {result_batched=}, {result_single=}"
