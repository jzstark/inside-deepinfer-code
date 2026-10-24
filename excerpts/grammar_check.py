if schema_type == "regex":
    assert re.fullmatch(so_args["schema"], text), f"regex: {text!r}"
elif schema_type == "json":
    value = json.loads(text)
    assert isinstance(value, dict) and set(value) == {"color", "count"}, f"json: {text!r}"
    assert value["color"] in {"red", "green", "blue"}, f"json: {text!r}"
    assert type(value["count"]) is int and value["count"] in {1, 2, 3}, f"json: {text!r}"
else:
    assert schema_type == "ebnf"
    assert re.fullmatch(r"answer=(yes|no);level=[1-3]", text), f"ebnf: {text!r}"
