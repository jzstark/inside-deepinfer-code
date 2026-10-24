"""Generate tokens with a fixed score table, not a trained language model.

Run: python3 labs/01_generate.py
Change the scores or the output limit. Greedy selection uses the largest score.
The invented vocabulary and scores have no connection to DeepSeek's tokenizer.
"""

VOCAB = ["<start>", "Small", "steps", "help", "<end>"]
SCORES = [
    [0, 9, 1, 0, 0],  # <start> -> Small
    [0, 0, 8, 1, 0],  # Small   -> steps
    [0, 0, 0, 7, 1],  # steps   -> help
    [0, 0, 0, 0, 9],  # help    -> <end>
    [0, 0, 0, 0, 9],  # <end>   -> <end>
]
END = VOCAB.index("<end>")


def generate(max_new_tokens=8):
    history = [0]
    output = []
    for step in range(1, max_new_tokens + 1):
        scores = SCORES[history[-1]]
        token = scores.index(max(scores))
        print(f"step {step}: {VOCAB[history[-1]]} -> {VOCAB[token]}")
        history.append(token)
        if token == END:
            print("stop: end token")
            break
        output.append(VOCAB[token])
    else:
        print("stop: output limit")
    print("text:", " ".join(output))
    return output


if __name__ == "__main__":
    output = generate()
    assert output == ["Small", "steps", "help"]
