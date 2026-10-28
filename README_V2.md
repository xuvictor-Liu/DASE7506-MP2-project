# Second research round: current frozen submission

**Official test: 84.36830835117773% (394/467).** Target 90% is not achieved (at least 421 correct would be needed). First-round result was 382/467; the second-round method was selected on validation, frozen, then scored, and improves by 12 questions. No test-answer error analysis or test-driven configuration selection was performed. The previously observed aggregate first-round score was known during this round and is disclosed here.

Validation: 750/798 = 93.984962%; disagreement subset: 307/355 = 86.478873%. These are not test scores.

## Frozen inference

From `code/`, Python 3.12:

```powershell
python -m pip install numpy==2.2.6
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:NUMEXPR_NUM_THREADS='4'
python predict.py --method selector_refined.py --checkpoint runs_v2/final_bundle --split test --output runs_v2/reproduced-test-predictions.json
python score_predictions.py --split test --predictions runs_v2/reproduced-test-predictions.json --output runs_v2/reproduced-test-score.json
```

Linux uses the same Python arguments and `export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4`. Final inference requires only NumPy. `evaluate_v2.py` verifies frozen hashes, runs batch/permutation checks on validation, and times fresh official validation/test processes. The original runner, parser, scorer, and dataset remain unchanged.

## Reproduce the entire search

Install `requirements-method.txt`, then run in order from `code/`:

```text
python train.py
python train_neural.py
python train_v2.py
python train_v2.py --projected
python train_text.py
python train_stack.py
python refine_v2.py
python train_correction.py
python finalize_v2.py
```

These commands retrain and overwrite development model bundles; preserve the distributed checkpoints when reproducing. All fitting uses supplied training references. All model choices use validation references. `finalize_v2.py` writes hashes before any second-round test scoring. Do not repeatedly change configurations based on test results.

## Selected method

The final score combines 75% of the first-round 64-unit neural scorer with 25% of the projected pairwise ensemble. The latter averages three 64-unit networks (seeds 41/42/43, epoch 60), using separate training-only 96-component PCA transformations for raw and centered embeddings plus expanded structural features. It adds a correction-evidence weight of 1: a candidate's erroneous final numeric equation supports another supplied candidate if the independently computed left side matches that candidate's final answer. Scores do not generate new answers. No text TF-IDF, tree model, five-fold stack, or answer-group aggregation is included in the final checkpoint; those were evaluated and rejected or tied without improving the selected validation score.

The uncompressed checkpoint occupies 1,803,170 bytes (1.7196 MiB). Fresh official test prediction and scoring total 1.0539 seconds on Intel Core Ultra 7 251HX, Windows 11, four library threads. Reference-CPU eligibility remains determined by the course reference machine. All statistics, PCA transforms, ensemble members, and configuration are inside `code/runs_v2/final_bundle/model.pkl`.

## Relevant files

- `report_v2.pdf`: two-page second-round report followed by the four-page initial report as historical appendix; six pages total.
- `code/runs_v2/test-score.json`: current official result.
- `code/runs_v2/frozen_manifest.json`: current code/checkpoint SHA-256 hashes and validation selection record.
- `code/runs_v2/resources.json`: fresh-process timing and compatibility checks.
- `code/runs_v2/experiments.json`, `refinement.json`, `correction_experiments.json`, `final_comparison.json`: complete second-round comparisons.
- `code/runs_projected/experiments.json`, `code/runs_text/experiments.json`, `code/runs_stack/experiments.json`: additional experiments, including failed approaches.
- `AI_DISCLOSURE.md`: substantive AI assistance and attribution.

Review this version and report before publishing. Use the v2 checkpoint and accuracy together; do not mix the first-round `runs/model_bundle` with the second-round inference command. Online submission has not been made. A GitHub code version and public code/checkpoint links still need to be supplied by the student. The full package includes unused research checkpoints; only `runs_v2/final_bundle` is the designated inference asset directory and the asset count above applies to it.

The search evaluated many alternatives on a small validation disagreement subset. Its selected 86.48% is subject to development selection bias. The improvement over the first-round validation result is only five questions; neither this nor the test result justifies claiming 90% generalization. The initial report is retained for experimental provenance and is superseded by the second-round summary for the current submitted method.
