"""Exact arithmetic for a teaching model of bounded SWA replay.

Each layer caches its input (identity KV projection), takes the uniform mean
of a causal window including the current row, and adds a row-local residual.
The shared variant projects KV from the unchanged boundary input instead.
This isolates cross-position dependencies; it does not run DeepInfer attention.
"""

from fractions import Fraction
from itertools import product


def prefill(inputs, window, depth, shared):
    boundary = list(map(Fraction, inputs))
    hidden = boundary
    caches = []
    for _ in range(depth):
        kv = boundary if shared else hidden
        caches.append(kv[-window:])
        output = []
        for i, value in enumerate(hidden):
            visible = kv[max(0, i - window + 1):i + 1]
            output.append(value + sum(visible) / len(visible))
        hidden = output
    return hidden, caches


def decode(value, caches, window, shared):
    boundary = hidden = Fraction(value)
    updated = []
    for cache in caches:
        kv = boundary if shared else hidden
        visible = (cache + [kv])[-window:]
        updated.append(visible)
        hidden += sum(visible) / len(visible)
    return hidden, updated


def dependency_span(window, depth):
    return window + (depth - 1) * (window - 1)


def check_cases():
    """Windows 2-4, depths 1-3; every 0/1 input two positions longer than the span."""
    checks = 0
    for window in (2, 3, 4):
        for depth in (1, 2, 3):
            span = dependency_span(window, depth)
            short_differs = False
            for inputs in product(range(2), repeat=span + 2):
                full = prefill(inputs, window, depth, True)
                suffix = prefill(inputs[-window:], window, depth, True)
                assert full[0][-1] == suffix[0][-1] and full[1] == suffix[1]
                assert decode(3, full[1], window, True) == decode(3, suffix[1], window, True)
                full = prefill(inputs, window, depth, False)
                suffix = prefill(inputs[-span:], window, depth, False)
                assert full[0][-1] == suffix[0][-1] and full[1] == suffix[1]
                assert decode(3, full[1], window, False) == decode(3, suffix[1], window, False)
                short = prefill(inputs[-(span - 1):], window, depth, False)
                short_differs |= (full[0][-1] != short[0][-1]
                                  or decode(3, full[1], window, False)[0]
                                  != decode(3, short[1], window, False)[0])
                checks += 1
            assert short_differs, (window, depth)  # one position fewer is not enough
    return checks


def main():
    inputs = list(range(1, 9))
    window, depth, keep = 3, 2, 3
    print(f"input: {inputs}; window={window}, layers={depth}, keep={keep}")
    print("KV source    last row (full / suffix)  next row (full / suffix)")
    for shared in (True, False):
        full = prefill(inputs, window, depth, shared)
        suffix = prefill(inputs[-keep:], window, depth, shared)
        next_full = decode(9, full[1], window, shared)[0]
        next_suffix = decode(9, suffix[1], window, shared)[0]
        name = "shared" if shared else "per-layer"
        print(f"{name:12} {str(full[0][-1]):>5} / {str(suffix[0][-1]):<7}"
              f"              {next_full} / {next_suffix}")
        if shared:
            assert full[0][-1] == suffix[0][-1] and full[1] == suffix[1]
        else:
            assert full[0][-1] != suffix[0][-1] and next_full != next_suffix
            for layer, (left, right) in enumerate(zip(full[1], suffix[1]), 1):
                differences = [i for i, (a, b) in enumerate(zip(left, right)) if a != b]
                if differences:
                    i = differences[0]
                    position = len(inputs) - window + i
                    print(f"first changed cache: layer {layer}, position {position}"
                          f" (zero-based), full={left[i]}, suffix={right[i]}")
                    break
    span = dependency_span(window, depth)
    full = prefill(inputs, window, depth, False)
    longer = prefill(inputs[-span:], window, depth, False)
    assert full[1] == longer[1] and full[0][-1] == longer[0][-1]
    print(f"keep={span}: per-layer final row and retained KV are exact")
    print(f"20 layers, window 128: dependency span = {dependency_span(128, 20)}")
    print(f"PASS {check_cases()} cases: shared and span replays exact,"
          " span - 1 fails on some input; one decode step each")


if __name__ == "__main__":
    main()
