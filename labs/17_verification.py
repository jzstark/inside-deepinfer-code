"""Enumerate standard and penalised speculative sampling with exact fractions.

Run: python3 labs/17_verification.py
P is the target after sampling transforms; Q is the distribution used to draw
proposals. The modified residual follows tilelang_ops/common.py:3-21,53 and
cuda/fused_sample.py:389-394,435-445. Rational alpha is illustrative; no kernel,
floating-point transform, fixed-point accumulation or fallback is simulated.
"""

from fractions import Fraction as F
from itertools import product
from math import expm1


def distribution(target, draft, alpha=F(1)):
    """Enumerate each proposal's accept/reject branches, then each replacement."""
    assert len(target) == len(draft) and 0 <= alpha <= 1
    assert sum(target) == sum(draft) == 1 and min(*target, *draft) >= 0
    residual = [max(p - alpha * q, F(0)) for p, q in zip(target, draft)]
    mass = sum(residual)
    output = [F(0)] * len(target)
    for token, q in enumerate(draft):
        if not q:
            continue
        accept = min(F(1), target[token] / q)
        output[token] += q * accept
        rejected = q * (1 - accept)
        if rejected:
            assert mass > 0
            for replacement, weight in enumerate(residual):
                output[replacement] += rejected * weight / mass
    assert sum(output) == 1
    return output


def check_bound(target, output, alpha):
    epsilon = 1 - alpha
    for p, a in zip(target, output):
        assert p - epsilon / (1 + epsilon) <= a <= (1 + epsilon) * p


def simplex(denominator):
    return [[F(a, denominator), F(b, denominator), F(denominator-a-b, denominator)]
            for a in range(denominator + 1) for b in range(denominator - a + 1)]


if __name__ == "__main__":
    p, q = [F(1, 2), F(1, 3), F(1, 6)], [F(1, 3), F(1, 2), F(1, 6)]
    alpha = F(9, 10)
    standard, modified = distribution(p, q), distribution(p, q, alpha)
    assert standard == p and modified != p
    check_bound(p, modified, alpha)
    print("Exact probabilities; illustrative alpha=9/10")
    print("token  target  draft  standard  penalised  difference")
    for name, target, draft, exact, changed in zip("ABC", p, q, standard, modified):
        print(f"  {name}    {str(target):>4}   {str(draft):>4}      {str(exact):>4}"
              f"     {str(changed):>6}   {str(changed-target):>8}")
    print("Standard output equals target; penalised output obeys both bounds.")
    tests = 0
    for target, draft in product(simplex(6), repeat=2):
        assert distribution(target, draft) == target
        for a in (F(1), F(9, 10), F(1, 2), F(0)):
            check_bound(target, distribution(target, draft, a), a)
            tests += 1
    print(f"Exact grid: {tests} bound checks (includes equal and disjoint supports).")
    print(f"Configured penalty=5e-5: epsilon approximately {-expm1(-5e-5):.10f}")

    mean_accept = sum(min(x, y) for x, y in zip(p, q))
    residual_mass = sum(max(x - alpha * y, F(0)) for x, y in zip(p, q))
    proxy = max(1 - residual_mass, F(0))
    assert 0 <= mean_accept - proxy <= 1 - alpha
    print("\nIdeal arithmetic, before fixed-point rounding:")
    print(f"  mean acceptance of standard test: {mean_accept}")
    print(f"  1 - penalised residual mass:      {proxy}")
    print(f"  difference:                      {mean_accept-proxy}")
