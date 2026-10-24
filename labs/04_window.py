"""Rebuild a three-integer window from the processed suffix.

Run: python3 labs/04_window.py
Prediction is sum(last three processed integers) modulo ten. The newest output
has not yet been processed. Try another prompt or a different window width.
"""

from collections import deque


def generate(prompt, steps, width=3):
    history = list(prompt)
    window = deque(prompt, maxlen=width)
    processed = len(prompt)
    output = []
    for step in range(steps):
        if step > 0:
            window.append(output[-1])
            processed += 1
        token = sum(window) % 10
        assert list(window) == history[max(0, processed - width):processed]
        assert token == sum(history[max(0, processed - width):processed]) % 10
        history.append(token)
        output.append(token)
    return history, output, window, processed


def rebuild(history, processed, width=3):
    return deque(history[max(0, processed - width):processed], maxlen=width)


if __name__ == "__main__":
    history, output, window, processed = generate([1, 2, 3, 5], 4)
    original = list(window)
    window.clear()  # Lose the cache; keep the token history and its boundary.
    window = rebuild(history, processed)
    assert list(window) == original == [0, 8, 3]
    assert output == [0, 8, 3, 1]
    wrong = list(history[-3:])  # Includes the emitted but unprocessed token.
    assert wrong != list(window)
    print("generated:", output)
    print(f"history entries: {len(history)} | processed entries: {processed}")
    print("window before loss:", original)
    print("rebuilt from processed suffix:", list(window))
    print("same window:", list(window) == original)
    print("wrong suffix (includes latest output):", wrong)
    window.append(output[-1])  # Process the latest output before predicting.
    next_token = sum(window) % 10
    assert next_token == sum(history[-3:]) % 10 == 2
    print("next token after processing latest output:", next_token)
