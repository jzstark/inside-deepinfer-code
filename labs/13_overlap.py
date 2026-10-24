"""Simulate host and device timelines for synchronous and ZOS decode steps.

Run: python3 labs/13_overlap.py
Durations are assigned abstract units, not measurements. The host performs one
operation at a time; the device runs queued work in submission order. The ZOS
order follows Worker.execute_model_step in DeepInfer's worker.py, lines 113-189.
Writes results/13_timeline.tex, the drawing data for the chapter's figure.
"""

from pathlib import Path

BASE = {"S": 3, "L": 1, "P": 4, "C": 1, "F": 10, "M": 2, "X": 0}
NAMES = {"S": "schedule", "P": "post", "C": "callbacks"}


def run(costs, steps, zos):
    host, device, sample_end = [], [], {}
    t, free = 0, 0

    def work(name, step, duration):
        nonlocal t
        host.append((name, step, t, t + duration))
        t += duration

    def enqueue(name, step, duration):
        nonlocal free
        start = max(t, free)
        free = start + duration
        device.append((name, step, start, free))
        return free

    def wait(done, step):
        nonlocal t
        if done > t:
            host.append(("wait", step, t, done))
            t = done

    for n in range(1, steps + 1):
        work("schedule", n, costs["S"])
        work("launch", n, costs["L"])
        d2h_done = enqueue("correct", n, costs["X"]) if zos and costs["X"] and n > 1 else 0
        enqueue("forward", n, costs["F"])
        if zos and n > 1:
            wait(sample_end[n - 1], n - 1)  # sample_end_evt(N).synchronize()
            work("post", n - 1, costs["P"])
            wait(d2h_done, n)  # d2h_evt.synchronize() before sampler metadata
        work("launch", n, costs["L"])
        enqueue("sample", n, costs["M"])
        work("callbacks", n, costs["C"])
        sample_end[n] = max(free, t)  # the event is recorded after callbacks
        if not zos:
            wait(sample_end[n], n)
            work("post", n, costs["P"])
    if zos:
        wait(sample_end[steps], steps)  # final flush of the pending step
        work("post", steps, costs["P"])
    return host, device, t


def summary(costs, zos, steps=6):
    host, device, finish = run(costs, steps, zos)
    ends = [end for name, _, _, end in device if name == "sample"]
    interval = ends[3] - ends[2]  # a repeating middle step
    busy = costs["F"] + costs["M"] + costs["X"] * zos
    forward = {step: end for name, step, _, end in device if name == "forward"}
    sample = {step: start for name, step, start, _ in device if name == "sample"}
    gap = sample[3] - forward[3]
    waited = sum(end - start for name, step, start, end in host if name == "wait" and step == 3)
    if zos:
        post = {step: start for name, step, start, _ in host if name == "post"}
        launch = [start for name, step, start, _ in host if name == "launch" and step == 4]
        assert launch[0] < post[3]  # forward 4 was enqueued before post(3)
    return interval, interval - busy, gap, waited, ends[0], finish


def draw(costs, path, steps=3, scale=0.17):
    lines = []
    for row, zos in ((0, False), (2, True)):
        host, device, _ = run(costs, steps, zos)
        for lane, items in ((row, host), (row + 1, device)):
            for name, step, start, end in items:
                if end > start:
                    style = "wait" if name == "wait" else name
                    wide = (end - start) * scale >= 0.45
                    label = str(step) if wide and name not in ("wait", "launch") else ""
                    lines.append(f"\\draw[{style}] ({start * scale:.2f},{-lane * 0.7:.2f}) "
                                 f"rectangle ({end * scale:.2f},{-lane * 0.7 - 0.5:.2f}) "
                                 f"node[midway,font=\\tiny]{{{label}}};")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    cases = [("baseline", BASE),
             ("slow scheduling", {**BASE, "S": 12}),
             ("long post-processing", {**BASE, "P": 14}),
             ("correction before forward", {**BASE, "X": 1})]
    print("Assigned units. interval = repeating step; idle = device idle per step;")
    print("gap = device wait between forward and sample; wait = host wait for sample 3;")
    print("first = first sample end; finish = end of final post-processing.")
    print("case                       mode  interval idle gap wait first finish")
    results = {}
    for label, costs in cases:
        for zos in (False, True):
            if label.startswith("correction") and not zos:
                continue
            row = summary(costs, zos)
            results[label, zos] = row
            mode = "ZOS " if zos else "sync"
            print(f"{label:26s} {mode} " + " ".join(f"{v:6d}" for v in row))
    assert results["baseline", False][:3] == (20, 8, 0)
    assert results["baseline", True][:3] == (12, 0, 0)
    assert results["slow scheduling", True][1] > 0
    assert results["long post-processing", True][2] > 0
    assert results["correction before forward", True][:3] == (13, 0, 0)
    draw(BASE, Path(__file__).resolve().parents[1] / "results" / "13_timeline.tex")
    print("\nTimer view of the baseline: a host timer around 'launch' records",
          f"{BASE['L']} unit; the device forward it queued lasts {BASE['F']}.")
