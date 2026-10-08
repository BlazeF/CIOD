"""Apply causal history/feedback rules to saved frame predictions.

Usage:
    python history_feedback.py
    python history_feedback.py --input predictions.json --model mobilenet_v3_large \
        --output-csv history_feedback_comparison.csv \
        --output-summary history_feedback_summary.json

For each frame i, adjusted labels use only predictions from frames before i.
This script does not run or modify a model; it post-processes saved outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent


def count_switches(labels: list[str]) -> int:
    return sum(left != right for left, right in zip(labels, labels[1:]))


def prior_majority(labels: list[str], window: int) -> list[str]:
    """Predict from the preceding window only; first frame uses its raw label."""
    adjusted = [labels[0]] if labels else []
    for index in range(1, len(labels)):
        prior = labels[max(0, index - window):index]
        counts = Counter(prior)
        # Resolve ties in favor of the most recent label among tied classes.
        recent_rank = {label: offset for offset, label in enumerate(reversed(prior))}
        adjusted.append(max(counts, key=lambda label: (counts[label], -recent_rank[label])))
    return adjusted


def prior_confidence_latch(
    labels: list[str],
    confidences: list[float],
    target: str = "News",
    enter_threshold: float = 0.70,
    enter_run: int = 2,
    exit_threshold: float = 0.80,
    exit_run: int = 3,
) -> list[str]:
    """Causal target-label latch using only earlier frame top-1 outputs.

    Enter `target` after `enter_run` consecutive prior target predictions at or
    above enter_threshold. Exit after exit_run consecutive prior non-target
    predictions at or above exit_threshold. The initial state is frame 1's
    raw top-1 label. While latched to a non-target label, a sustained strong
    alternate class can update that label after exit_run frames.
    """
    if not labels:
        return []
    result = [labels[0]]
    state = labels[0]
    target_run = 0
    alternate_run = 0
    alternate_candidate: str | None = None

    for index in range(1, len(labels)):
        previous_label = labels[index - 1]
        previous_confidence = confidences[index - 1]

        if previous_label == target and previous_confidence >= enter_threshold:
            target_run += 1
        else:
            target_run = 0

        if previous_label != target and previous_confidence >= exit_threshold:
            if previous_label == alternate_candidate:
                alternate_run += 1
            else:
                alternate_candidate = previous_label
                alternate_run = 1
        else:
            alternate_run = 0
            alternate_candidate = None

        if state == target:
            if alternate_run >= exit_run and alternate_candidate is not None:
                state = alternate_candidate
                target_run = 0
                alternate_run = 0
                alternate_candidate = None
        elif target_run >= enter_run:
            state = target
            target_run = 0
        elif alternate_run >= exit_run and alternate_candidate is not None:
            state = alternate_candidate
            alternate_run = 0
            alternate_candidate = None

        result.append(state)

    return result


def first_top1(record: dict[str, Any], model: str, output_key: str, label_key: str) -> tuple[str, float]:
    values = record["models"][model][output_key]
    if not values:
        raise ValueError(f"Empty {output_key} for model {model}")
    best = values[0]
    return str(best[label_key]), float(best["probability"])


def summarize(semantic: list[str], fine: list[str]) -> dict[str, Any]:
    return {
        "semantic_top1_counts": dict(Counter(semantic).most_common()),
        "fine_top1_counts": dict(Counter(fine).most_common()),
        "semantic_switches": count_switches(semantic),
        "fine_switches": count_switches(fine),
        "semantic_news_frames": semantic.count("News"),
        "fine_news_channel_frames": fine.count("News Channel-News"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "predictions.json")
    parser.add_argument("--model", default="mobilenet_v3_large")
    parser.add_argument("--output-csv", type=Path, default=HERE / "history_feedback_comparison.csv")
    parser.add_argument("--output-summary", type=Path, default=HERE / "history_feedback_summary.json")
    parser.add_argument("--window", type=int, default=5, help="Number of prior predictions for majority vote")
    parser.add_argument("--news-enter-confidence", type=float, default=0.70)
    parser.add_argument("--news-enter-run", type=int, default=2)
    parser.add_argument("--other-exit-confidence", type=float, default=0.80)
    parser.add_argument("--other-exit-run", type=int, default=3)
    args = parser.parse_args()

    if args.window < 1 or args.news_enter_run < 1 or args.other_exit_run < 1:
        parser.error("window and run lengths must be positive")

    records = json.loads(args.input.read_text(encoding="utf-8"))
    if not records:
        parser.error("input JSON contains no frame records")

    semantic_pairs = [first_top1(r, args.model, "semantic_top2", "label") for r in records]
    fine_pairs = [first_top1(r, args.model, "fine_top3", "label") for r in records]
    semantic_raw, semantic_conf = map(list, zip(*semantic_pairs))
    fine_raw, fine_conf = map(list, zip(*fine_pairs))

    # "previous-frame hold" is explicitly the previous raw output on each step,
    # matching the earlier experiment's implementation.
    previous_hold_semantic = [semantic_raw[0], *semantic_raw[:-1]]
    previous_hold_fine = [fine_raw[0], *fine_raw[:-1]]
    majority_semantic = prior_majority(semantic_raw, args.window)
    majority_fine = prior_majority(fine_raw, args.window)
    latch_semantic = prior_confidence_latch(
        semantic_raw,
        semantic_conf,
        enter_threshold=args.news_enter_confidence,
        enter_run=args.news_enter_run,
        exit_threshold=args.other_exit_confidence,
        exit_run=args.other_exit_run,
    )

    methods = {
        "raw": (semantic_raw, fine_raw),
        "previous_frame_hold": (previous_hold_semantic, previous_hold_fine),
        "previous_prior_window_majority": (majority_semantic, majority_fine),
        "news_confidence_latch": (latch_semantic, majority_fine),
    }
    summary = {
        "model": args.model,
        "frame_count": len(records),
        "causal": True,
        "method_settings": {
            "majority_window_prior_frames": args.window,
            "news_enter_confidence": args.news_enter_confidence,
            "news_enter_consecutive_prior_frames": args.news_enter_run,
            "other_exit_confidence": args.other_exit_confidence,
            "other_exit_consecutive_same_other_class_prior_frames": args.other_exit_run,
        },
        "methods": {name: summarize(*pair) for name, pair in methods.items()},
        "limitation": "Video-level labels do not establish frame-level accuracy. Counts and switches measure output distribution/stability, not accuracy, unless frame labels are annotated.",
    }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "frame", "timestamp_seconds", "model", "raw_semantic", "raw_semantic_confidence",
            "previous_raw_semantic", "prior_majority_semantic", "news_latch_semantic",
            "raw_fine", "raw_fine_confidence", "previous_raw_fine", "prior_majority_fine",
        ])
        for index, record in enumerate(records):
            writer.writerow([
                record.get("frame", index + 1), record.get("timestamp_seconds", ""), args.model,
                semantic_raw[index], semantic_conf[index], previous_hold_semantic[index],
                majority_semantic[index], latch_semantic[index], fine_raw[index], fine_conf[index],
                previous_hold_fine[index], majority_fine[index],
            ])

    args.output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
