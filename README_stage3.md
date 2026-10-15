# Stage 3: tied-Gaussian head, epsilon table, confusability atlas

NEW FILES ONLY. Copy them into your existing project folder (stage0_pkg), keeping the same folder structure.
Nothing in this package replaces LOG.md, configs/poc.yaml or any Stage 0-2 file.

    src/head.py                  the stage
    tests/test_stage3.py         synthetic-data tests (needs scikit-learn ONLY for one test; it is skipped if missing)
    scripts/stage3_pair_ranks.py descriptive check: where do real confusions fall in the eps ranking
    LOG_stage3_entry.md          paste into LOG.md after the run

## Run
1. `py -3.12 -m pytest tests -q`            (everything should pass)
2. `py -3.12 -m src.head --run runs/20261003_stage1`
3. Output in `runs/20261003_stage1/head_emb/`: head.npz, scores_train_release.npz, scores_ft_train.npz,
   head_summary.json, pairs_903.csv, top10_confusable_pairs.csv, atlas.png, top10_pairs.png
4. Optional, descriptive only: `py -3.12 scripts/stage3_pair_ranks.py --dir runs/20261003_stage1/head_emb`
   (writes pair_ranks.csv and prints where the real confusions sit in the eps ranking).
5. Send me the printed summary, head_summary.json, atlas.png and top10_pairs.png.

## What it refuses to do
Stop unless the embeddings come from the frozen checkpoint and the Stage 2 files still match their recorded SHA-256.

## Reading the result
- eps is stored as log eps and used ONLY as a ranking (stratification). With a head fitted on tight, 99.96%-accurate frames the absolute values are far too small (log eps about -124 for a pair with a real 4.4% confusion rate). Never quote eps as an error probability.
- Spearman and the pair overlap are DESCRIPTIVE ONLY: nothing is tuned on them.

## If the head is more than 2 pp below softmax
The script prints the plan's mitigations in order. Do NOT try them on your own first: send me the output.
`--features raw` is plan mitigation 2 (un-normalised features); if you ever use it, log which version you used.
(In the real run the head was 0.70 pp BETTER than softmax, so no mitigation was needed.)

## Files the later stages read
Stage 4 verifies and reads `head.npz` and `scores_train_release.npz` by their SHA-256 (recorded in LOG_stage3_entry.md and head_summary.json). If you re-run Stage 3 the hashes must be re-checked.
