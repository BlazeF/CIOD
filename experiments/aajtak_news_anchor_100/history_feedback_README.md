# Causal history-feedback experiment (MobileNet)

Compares independent MobileNet predictions with causal sequence rules. For frame i, adjustments use only frames before i. Methods: previous-frame hold; five-frame prior majority vote; and News confidence latch (two preceding News predictions >=0.70 to enter News; three preceding non-News predictions >=0.80 to exit). Fine labels also include a five-prior-frame majority vote.

No human frame labels were created, so stability and class counts do not establish accuracy. Tune thresholds only on labeled validation clips to avoid choosing a rule that merely forces News.

See `history_feedback_comparison.csv` and `history_feedback_summary.json`.

Reusable reference script: `history_feedback.py`. From this folder, run
`python history_feedback.py`; it reads the saved `predictions.json` and writes
the comparison CSV and summary JSON. Options and thresholds can be changed with
command-line arguments; see `python history_feedback.py --help`.
