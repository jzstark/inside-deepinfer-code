"""Teaching reductions and numerical checks; Python floats, no engine kernels.

The metric expression follows deepinfer/testing.py:11 at 774ce244.
The two inputs are adjacent binary32 values, represented exactly as Python
binary64 floats. This reproduces cancellation, not PyTorch execution.
"""

import itertools
import math


def left_sum(values):
    total = 0.0
    for value in values:
        total += value
    return total


def reduce_rows(rows, active, fixed=False):
    lanes = 1 if fixed or len(rows) == 1 else 2
    result = {}
    for name, values in rows:
        if name in active:
            partials = [left_sum(values[lane::lanes]) for lane in range(lanes)]
            result[name] = left_sum(partials)
    return result


def main():
    rows = [('A', [1e16, -1e16, 1.0]), ('B', [2.0, 3.0, 4.0])]
    single = reduce_rows(rows[:1], {'A'})['A']
    batched = reduce_rows(rows, {'A', 'B'})['A']
    masked = reduce_rows(rows, {'A'})['A']
    compact = reduce_rows(rows[:1], {'A'})['A']
    assert single == compact == 1.0 and batched == masked == 0.0
    print(f'row-count-dependent: single={single:g}, batched={batched:g}')
    print(f'row-count-dependent: masked={masked:g}, compact={compact:g}')

    # Compare every nonempty retained subset in both batch orders.
    checks = 0
    for order in itertools.permutations(rows):
        for count in range(1, len(rows) + 1):
            for names in itertools.combinations(('A', 'B'), count):
                active = set(names)
                compact_rows = [row for row in order if row[0] in active]
                expected = {name: left_sum(values) for name, values in compact_rows}
                assert reduce_rows(order, active, fixed=True) == expected
                assert reduce_rows(compact_rows, active, fixed=True) == expected
                checks += 1
    print(f'fixed order: A=1 in every layout; {checks} subset/order checks')

    x = -0.0010000000474974513
    y = -0.0009999999310821295
    d = x - y
    metric = (math.exp(d) - 1) - d
    assert x != y and abs(d) == 2**-33 and metric == 0.0
    print(f'adjacent float32 inputs: equal={x == y}, abs diff={abs(d):.12g}')
    print(f'exp(d)-1-d metric={metric:g}; zero does not establish equality')

    a, b = [-1.0, -3.0], [-2.0, -2.0]
    ppl_a = math.exp(-left_sum(a) / len(a))
    ppl_b = math.exp(-left_sum(b) / len(b))
    assert a != b and ppl_a == ppl_b
    print(f'logprob lists differ; both perplexities={ppl_a:.9f}')


if __name__ == '__main__':
    main()
