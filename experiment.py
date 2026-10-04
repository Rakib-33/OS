"""Reproducible Track 2 experiment. Run: python experiment.py"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from disk_schedulers import MAX_CYLINDER, SCHEDULERS, START_HEAD, schedule

SEED = 3072026
BATCH_SIZE = 20
KINDS = ("sequential", "random", "bursty")
SPLIT_SIZES = {"train": 900, "validation": 300, "test": 300}


def make_batch(rng: np.random.Generator, kind: str) -> list[int]:
    if kind == "sequential":
        start = int(rng.integers(15, 145))
        step = int(rng.choice((-1, 1)))
        gaps = rng.integers(1, 5, size=BATCH_SIZE)
        values = np.clip(start + step * np.cumsum(gaps), 0, MAX_CYLINDER)
    elif kind == "random":
        values = rng.integers(0, MAX_CYLINDER + 1, size=BATCH_SIZE)
    elif kind == "bursty":
        centers = rng.choice(np.arange(15, 185), size=2, replace=False)
        labels = rng.integers(0, 2, size=BATCH_SIZE)
        values = np.clip(np.rint(centers[labels] + rng.normal(0, 5, BATCH_SIZE)),
                         0, MAX_CYLINDER)
        rng.shuffle(values)
    else:
        raise ValueError(kind)
    return [int(v) for v in values]


def features(requests: list[int]) -> np.ndarray:
    a = np.asarray(requests, dtype=float)
    ordered_gaps = np.diff(a)
    sorted_gaps = np.diff(np.sort(a))
    return np.array([
        a.mean() / 199, a.std() / 100, a.min() / 199, a.max() / 199,
        np.mean(a >= START_HEAD), np.mean(np.abs(a - START_HEAD)) / 100,
        np.mean(np.abs(ordered_gaps)) / 100,
        np.mean(ordered_gaps > 0),
        np.mean(sorted_gaps <= 5),
        np.quantile(a, 0.25) / 199, np.quantile(a, 0.75) / 199,
    ], dtype=float)


def seek_values(requests: list[int]) -> np.ndarray:
    return np.array([schedule(name, requests).seek for name in SCHEDULERS], dtype=int)


def make_split(seed: int, count: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    items = []
    for i in range(count):
        kind = KINDS[i % len(KINDS)]
        items.append({"id": i, "kind": kind, "requests": make_batch(rng, kind)})
    rng.shuffle(items)
    return items


def make_timeline(seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [{"id": i, "kind": "sequential" if i < 30 else "bursty",
             "requests": make_batch(rng, "sequential" if i < 30 else "bursty")}
            for i in range(60)]


def dataset(items: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.vstack([features(item["requests"]) for item in items])
    seeks = np.vstack([seek_values(item["requests"]) for item in items])
    labels = np.argmin(seeks, axis=1)  # Fixed tie order: FCFS, SCAN, C-SCAN, SSTF.
    return x, seeks, labels


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train(x: np.ndarray, y: np.ndarray, mean: np.ndarray, std: np.ndarray):
    z = (x - mean) / std
    z = np.column_stack((z, np.ones(len(z))))
    weights = np.zeros((z.shape[1], len(SCHEDULERS)), dtype=float)
    targets = np.eye(len(SCHEDULERS))[y]
    for _ in range(1200):
        probabilities = softmax(z @ weights)
        gradient = z.T @ (probabilities - targets) / len(z)
        gradient[:-1] += 0.01 * weights[:-1]
        weights -= 0.15 * gradient
    return weights


def probabilities(x: np.ndarray, mean: np.ndarray, std: np.ndarray,
                  weights: np.ndarray, temperature: float) -> np.ndarray:
    z = np.column_stack(((x - mean) / std, np.ones(len(x))))
    return softmax((z @ weights) / temperature)


def fit_temperature(x: np.ndarray, y: np.ndarray, mean: np.ndarray,
                    std: np.ndarray, weights: np.ndarray) -> float:
    # Validation-only temperature scaling; the test/timeline remain untouched.
    candidates = np.linspace(0.5, 3.0, 51)
    losses = []
    for t in candidates:
        p = probabilities(x, mean, std, weights, float(t))
        losses.append(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean())
    return float(candidates[int(np.argmin(losses))])


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(name: str, items: list[dict], mean: np.ndarray, std: np.ndarray,
             weights: np.ndarray, temperature: float, out: Path) -> dict:
    x, seeks, labels = dataset(items)
    p = probabilities(x, mean, std, weights, temperature)
    picks = np.argmax(p, axis=1)
    confidence = np.max(p, axis=1)
    best = np.min(seeks, axis=1)
    correct = seeks[np.arange(len(items)), picks] == best
    calibration_error = 0.0
    for left, right in zip(np.linspace(0, 1, 6)[:-1], np.linspace(0, 1, 6)[1:]):
        in_bin = (confidence >= left) & (confidence < right if right < 1 else confidence <= right)
        if in_bin.any():
            calibration_error += float(in_bin.mean()) * abs(float(confidence[in_bin].mean()) -
                                                        float(correct[in_bin].mean()))
    rows = []
    for i, item in enumerate(items):
        row = {"batch": item["id"], "pattern": item["kind"],
               "requests": " ".join(map(str, item["requests"])),
               "oracle": SCHEDULERS[labels[i]], "selected": SCHEDULERS[picks[i]],
               "confidence": round(float(confidence[i]), 6), "correct": int(correct[i]),
               "selector_seek": int(seeks[i, picks[i]]), "oracle_seek": int(best[i])}
        row.update({f"{scheduler.lower().replace('-', '')}_seek": int(seeks[i, j])
                    for j, scheduler in enumerate(SCHEDULERS)})
        rows.append(row)
    write_csv(out / f"{name}_batches.csv", rows)
    summary = {
        "batches": len(items), "accuracy_equal_winners": round(float(correct.mean()), 4),
        "mean_confidence_correct": round(float(confidence[correct].mean()), 4) if correct.any() else None,
        "mean_confidence_wrong": round(float(confidence[~correct].mean()), 4) if (~correct).any() else None,
        "expected_calibration_error_5_bins": round(calibration_error, 4),
        "seek_totals": {**{s: int(seeks[:, j].sum()) for j, s in enumerate(SCHEDULERS)},
                        "selector": int(seeks[np.arange(len(items)), picks].sum()),
                        "oracle": int(best.sum())},
        "winner_label_counts": {s: int(np.sum(labels == j)) for j, s in enumerate(SCHEDULERS)},
    }
    if name == "timeline":
        summary["phases"] = {}
        for phase, sl in (("before", slice(0, 30)), ("after", slice(30, 60))):
            ps = seeks[sl]
            chosen = picks[sl]
            summary["phases"][phase] = {
                "accuracy_equal_winners": round(float(correct[sl].mean()), 4),
                "seek_totals": {**{s: int(ps[:, j].sum()) for j, s in enumerate(SCHEDULERS)},
                                "selector": int(ps[np.arange(len(ps)), chosen].sum()),
                                "oracle": int(ps.min(axis=1).sum())},
            }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    splits = {name: make_split(SEED + i, size)
              for i, (name, size) in enumerate(SPLIT_SIZES.items())}
    timeline = make_timeline(SEED + 10)
    train_x, _, train_y = dataset(splits["train"])
    val_x, _, val_y = dataset(splits["validation"])
    mean = train_x.mean(axis=0)
    std = np.maximum(train_x.std(axis=0), 1e-8)
    weights = train(train_x, train_y, mean, std)
    temperature = fit_temperature(val_x, val_y, mean, std, weights)
    np.savez(args.output / "selector_model.npz", weights=weights, mean=mean, std=std,
             temperature=np.array(temperature))
    summary = {"seed": SEED, "batch_size": BATCH_SIZE, "cylinders": 200,
               "initial_head": START_HEAD, "initial_direction": "up",
               "training_batches": len(splits["train"]),
               "validation_batches": len(splits["validation"]),
               "temperature": round(temperature, 3)}
    for name in ("test", "timeline"):
        summary[name] = evaluate(name, splits[name] if name == "test" else timeline,
                                 mean, std, weights, temperature, args.output)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
