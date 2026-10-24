"""One causal attention layer on fixed two-number inputs; no learned weights.

Compare recomputing every prefix with reusing projected keys and values.
Counts describe this Python algorithm, not accelerator instructions or time.
"""

from math import exp, isclose, sqrt

INPUTS = [(1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]


def project(x):
    a, b = x
    return (a, b), (a - b, a + b), (a + 2 * b, 2 * a - b)


def attend(query, keys, values):
    scores = [sum(a * b for a, b in zip(query, key)) / sqrt(len(query))
              for key in keys]
    peak = max(scores)
    masses = [exp(score - peak) for score in scores]
    total = sum(masses)
    weights = [mass / total for mass in masses]
    return tuple(
        sum(weight * value[d] for weight, value in zip(weights, values))
        for d in range(len(values[0])))


def recompute(xs, window=None):
    rows = [project(x) for x in xs]
    outputs, pairs = [], 0
    for i, (query, _, _) in enumerate(rows):
        start = 0 if window is None else max(0, i - window + 1)
        visible = rows[start:i + 1]
        outputs.append(attend(query, [r[1] for r in visible],
                              [r[2] for r in visible]))
        pairs += len(visible)
    return outputs, pairs


def incremental(xs, window=None):
    keys, values, outputs = [], [], []
    pairs = 0
    for x in xs:
        query, key, value = project(x)
        keys.append(key)
        values.append(value)
        if window is not None:
            keys, values = keys[-window:], values[-window:]
        outputs.append(attend(query, keys, values))
        pairs += len(keys)
    return outputs, pairs, len(keys)


def main():
    print("position | input = query | key     | value")
    for i, x in enumerate(INPUTS):
        q, k, v = project(x)
        print(f"{i:8} | {str(q):13} | {str(k):7} | {v}")
    for window in (None, 3):
        outputs, pairs, retained = incremental(INPUTS, window)
        rebuilt_rows = rebuilt_pairs = 0
        for end in range(1, len(INPUTS) + 1):
            reference, work = recompute(INPUTS[:end], window)
            rebuilt_rows += end
            rebuilt_pairs += work
            assert all(isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                       for actual, expected in zip(outputs[:end], reference)
                       for a, b in zip(actual, expected))
        name = "full" if window is None else f"window={window}"
        print(f"\n{name}: cached outputs match every recomputed prefix")
        print(f"projected rows: recompute={rebuilt_rows}, cache={len(INPUTS)}")
        print(f"query/key pairs: recompute={rebuilt_pairs}, cache={pairs}")
        print(f"retained KV rows: {retained}")
        print("last output: " + ", ".join(f"{value:.6f}" for value in outputs[-1]))
    full, _, _ = incremental(INPUTS)
    windowed, _, _ = incremental(INPUTS, 3)
    assert full[:3] == windowed[:3]
    assert any(not isclose(a, b) for a, b in zip(full[-1], windowed[-1]))


if __name__ == "__main__":
    main()
