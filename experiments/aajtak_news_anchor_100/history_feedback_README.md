# Causal history-feedback experiment (MobileNet)

Compares saved model predictions with causal sequence rules. For frame i, adjustments use only frames before i. Methods include previous-frame hold, prior-window majority vote, and a configurable target-label confidence latch. Defaults target semantic `News` and fine `News Channel-News`; for example, pass `--semantic-target Sports --fine-target Cricket-Sports` for a cricket clip. The latch enters after two consecutive prior target predictions at or above 0.70 confidence and exits after three consecutive prior predictions of the same other class at or above 0.80.

Video-level labels do not establish frame-level accuracy. Stability and class counts are not accuracy metrics without frame-level annotations. Tune thresholds on labeled validation clips to avoid choosing a rule that merely forces the target class.

See `history_feedback_comparison.csv` and `history_feedback_summary.json`.

Reusable reference script: `history_feedback.py`. From this folder, run
`python history_feedback.py`; it reads the saved `predictions.json` and writes
the comparison CSV and summary JSON. Options and thresholds can be changed with
command-line arguments; see `python history_feedback.py --help`.
