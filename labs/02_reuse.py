"""Reuse a prefix sum in an integer generator.

Run: python3 labs/02_reuse.py
The next integer is sum(history) modulo 10. Count additions, not wall time.
Both paths produce three integers; neither processes the final emitted integer.
"""


def recompute(prompt, steps):
    history = list(prompt)
    output = []
    additions = 0
    for _ in range(steps):
        total = 0
        for token in history:
            total += token
            additions += 1
        token = total % 10
        history.append(token)
        output.append(token)
    return output, additions


def reuse(prompt, steps):
    total = 0
    additions = 0
    processed = 0
    for token in prompt:
        total += token
        additions += 1
        processed += 1
    output = []
    for step in range(steps):
        if step > 0:
            total += output[-1]
            additions += 1
            processed += 1
        output.append(total % 10)
    return output, additions, processed


if __name__ == "__main__":
    prompt = [1, 2, 3, 5]
    plain, plain_count = recompute(prompt, 3)
    cached, cached_count, processed = reuse(prompt, 3)
    assert plain == cached == [1, 2, 4]
    assert (plain_count, cached_count) == (15, 6)
    assert processed == len(prompt) + len(cached) - 1
    print("prompt:", prompt)
    print("recompute:", plain, "| additions:", plain_count)
    print("reuse:    ", cached, "| additions:", cached_count)
    print("same output:", plain == cached)
    print(f"history entries: {len(prompt) + len(cached)}"
          f" | entries covered by final cache: {processed}")
