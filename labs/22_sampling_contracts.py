"""Teaching random keys, verification budgets and exact PIT interval masses.

SHA-256 is a specified teaching generator, not DeepInfer's device RNG. Roles
0/1/2 and the draft seed offset are inspired by tilelang_ops/fused_sample.py
and cuda/fused_sample.py at 774ce244. Positions here are absolute toy positions;
DeepInfer additionally mixes local verification offsets into its position keys.
Budget examples use target=draft=(1/2,1/2), so acceptance is certain. No model,
XGrammar or released statistical test runs here. PIT calculations use Fraction.
"""

from fractions import Fraction as F
from hashlib import sha256
from itertools import permutations
from random import Random

DRAFT_OFFSET = 1000000007


def keyed_draw(seed, position, role, draft=False):
    effective_seed = seed + (DRAFT_OFFSET if draft else 0)
    message = f'{effective_seed}:{3 * position + role}'.encode('ascii')
    bits = int.from_bytes(sha256(message).digest()[:8], 'big')
    return F(bits, 2**64)


def token(draw):
    return int(draw >= F(1, 2))


def shared_batch(order):
    rng = Random(1)
    return {name: token(rng.random()) for name in order}


def keyed_batch(order, seeds):
    return {name: token(keyed_draw(seeds[name], 7, 0)) for name in order}


def budget_output(seed, start, q):
    # q-1 verified drafts, then one fresh target draw, as in the metadata rule.
    output, roles = {}, {}
    for local in range(q):
        position = start + local
        verify = local != q - 1
        if verify:
            draft = token(keyed_draw(seed, position, 0, draft=True))
            assert keyed_draw(seed, position, 2) < 1  # target/draft ratio = 1
            output[position], roles[position] = draft, 'accepted draft'
        else:
            output[position] = token(keyed_draw(seed, position, 0))
            roles[position] = 'fresh target'
    return output, roles


def pit_bin_masses(target, actual, edges):
    # Inclusive CDF: ascending probability, then token index, as in the source.
    order = sorted(range(len(target)), key=lambda i: (target[i], i))
    intervals, upper = {}, F(0)
    for i in order:
        intervals[i] = (upper, upper + target[i])
        upper += target[i]
    assert upper == 1 and sum(actual) == 1 and all(p > 0 for p in target)
    masses = []
    for left, right in zip(edges, edges[1:]):
        mass = F(0)
        for i, (lo, hi) in intervals.items():
            overlap = max(F(0), min(right, hi) - max(left, lo))
            mass += actual[i] * overlap / target[i]
        masses.append(mass)
    assert sum(masses) == 1
    return masses


def main():
    seeds = {'A': 11, 'B': 29, 'C': 43}
    ab, ba = shared_batch('AB'), shared_batch('BA')
    assert ab != ba
    print(f'shared stream AB: A={ab["A"]}, B={ab["B"]}')
    print(f'shared stream BA: A={ba["A"]}, B={ba["B"]}')
    reference = keyed_batch('ABC', seeds)
    for order in permutations('ABC'):
        assert keyed_batch(order, seeds) == reference
    for name in seeds:
        assert keyed_batch([name], seeds)[name] == reference[name]
    for position in range(8):
        for role in range(3):
            draws = {name: keyed_draw(seeds[name], position, role) for name in 'ABC'}
            assert {name: keyed_draw(seeds[name], position, role) for name in 'CBA'} == draws
    print('keyed draws: all 6 batch orders and single requests agree')

    short, short_roles = budget_output(11, 7, 1)
    long, long_roles = budget_output(11, 7, 3)
    print(f'q=1, position 7: {short_roles[7]}, token={short[7]}')
    print(f'q=3, position 7: {long_roles[7]}, token={long[7]}')
    assert short[7] != long[7]
    # Both rules have the same exact marginal under uniform draws.
    target = (F(1, 2), F(1, 2))
    accepted_marginal = tuple(p * min(F(1), t / p) for p, t in zip(target, target))
    assert accepted_marginal == target
    print('ideal marginal in both budgets: [1/2, 1/2]')

    target = [F(1, 2), F(1, 4), F(1, 4)]
    edges = [F(i, 4) for i in range(5)]
    correct = pit_bin_masses(target, target, edges)
    biased = pit_bin_masses(target, [F(3, 4), F(1, 8), F(1, 8)], edges)
    assert correct == [F(1, 4)] * 4
    assert biased == [F(1, 8), F(1, 8), F(3, 8), F(3, 8)]
    print('PIT quarter masses, correct: ' + ', '.join(map(str, correct)))
    print('PIT quarter masses, biased:  ' + ', '.join(map(str, biased)))


if __name__ == '__main__':
    main()
