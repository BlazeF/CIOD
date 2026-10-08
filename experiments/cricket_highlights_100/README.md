# Cricket highlight frame experiment

**Source:** [ICC — Rohit Hundred Seals Win | South Africa vs India, 2019 Cricket World Cup](https://www.youtube.com/watch?v=amMJfaB5dXo) (5:08).

100 frames were sampled evenly from 20% through 80% of the video (61.576–246.304 seconds), leaving out the opening and closing sections. The frames are in `frames/`.

## Results

| Model | `Sports` semantic top-1 | `Cricket-Sports` fine top-1 | Mean top-1 confidence for fine class |
|---|---:|---:|---:|
| MobileNetV3 Large | 82 / 100 | 82 / 100 | 77.5% |
| ViT Tiny | 70 / 100 | 61 / 100 | 57.5% |

ViT predicted `Game` as its semantic top-1 on 23 frames and `Sports-Game` as its fine top-1 on 16. MobileNet predicted `Game` on 7 frames. Both models often returned ImageNet concepts such as `ballplayer` and `scoreboard`.

These are prediction counts, not accuracy: the sampled frames were not manually annotated with ground-truth labels. This is a single video and only an initial out-of-sample behavior check.

## Causal history-feedback test (MobileNet)

Applied prior-only rules to the saved MobileNet predictions, targeting semantic `Sports` and fine `Cricket-Sports`. Raw predictions gave 82/100 for both target labels. A majority vote over the previous five frames gave 97/100 semantic Sports and 96/100 Cricket-Sports. Confidence latches (enter after two prior target predictions at >=0.70; exit after three same-class alternatives at >=0.80) gave 100/100 Sports and 100/100 Cricket-Sports when both latches were combined. These are smoothed label counts, not accuracy gains; the clip has a video-level cricket label but no manual frame-level annotations.

- `history_feedback_comparison.csv`: raw and history-adjusted label for every frame.
- `history_feedback_summary.json`: counts, switches, settings, and limitation.
- The reusable script is `../aajtak_news_anchor_100/history_feedback.py`.

## Files

- `frames/`: 100 sampled JPG frames.
- `predictions.csv`: compact top-k outputs for each frame and model.
- `predictions.json`: complete per-frame top-k labels and probabilities.
- `summary.json`: prediction counts, mean confidences, and sampling details.
- `source.json`: source and sample metadata.
- `run_experiment.py`: script to reproduce frame extraction and inference from a local video file.
