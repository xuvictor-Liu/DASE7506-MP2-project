# Result and checkpoint status — 26 October 2026

Read `report_consolidated.pdf` for the six-page consolidated report and audit. Earlier reports/READMEs are historical records; their wording about a "current" method is superseded by this version map. No online submission or publication has been made.

| Version | Entrypoint | Checkpoint directory | Official test result |
|---|---|---|---|
| V2 | `selector_refined.py` | `runs_v2/final_bundle` | 394/467 = **84.36830835117773%** |
| V3 | `selector_semantic_blend.py` | `runs_v3/final_bundle` | 391/467 = **83.72591006423983%** |

V3 was frozen on the basis of its higher hard-validation score (309/355 = 87.0423%). That validation number is **not** a submission accuracy. Its final test result was lower than V2's historical result. Scores must remain paired with their corresponding code and checkpoint. Neither version demonstrates the requested 87% test target. This document does not use test scores to select a replacement method.

## Reproduce an already frozen result

Work from `code/` with Python 3.12 and the unchanged course `data/` directory. V2 inference needs NumPy 2.2.6; install `requirements-method.txt` for V3 inference. Later neural research additionally uses `requirements-round5.txt`.

PowerShell thread settings:

```powershell
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:NUMEXPR_NUM_THREADS='4'
```

V2 reproduction:

```text
python predict.py --method selector_refined.py --checkpoint runs_v2/final_bundle --split test --output runs_v2/reproduced-predictions.json
python score_predictions.py --predictions runs_v2/reproduced-predictions.json --split test --output runs_v2/reproduced-score.json
```

V3 reproduction:

```text
python predict.py --method selector_semantic_blend.py --checkpoint runs_v3/final_bundle --split test --output runs_v3/reproduced-predictions.json
python score_predictions.py --predictions runs_v3/reproduced-predictions.json --split test --output runs_v3/reproduced-score.json
```

These commands reproduce previously frozen results. Do not use repeated test results to tune or select models. Source/checkpoint hashes and fresh-process timing records are retained in the respective `frozen_manifest.json` and `resources.json`. Local timings are 1.0539 seconds for V2 and 4.5102 seconds for V3; the course reference CPU determines eligibility.

## Audit and focused correction

```text
python audit_pipeline.py
python train_aligned.py
```

The audit reads train/validation references only. Label counts match the official scorer; positive labels are never masked as invalid. Sampled inference is batch-independent and candidate-permutation equivariant. Frozen source/checkpoint hashes match.

A real methodological mismatch was found: the V3 role-text classifier trains on raw five-fold teacher scores but receives raw V2 scores at inference. The matched-family, normalized-prior correction reaches 306/355 hard validation, below 309/355, so it was not substituted into a scored checkpoint. The experiment does not establish that this mismatch caused the test decline.

`train_aligned.py` requires preserved `runs_stack`, `runs_role_text`, and fifth-round development checkpoints. Search and training commands should be run in a copy: many scripts overwrite development outputs. Full earlier reproduction instructions remain in `README_V2.md` and the ROUND*_RESEARCH.md files. The original course data, parser and scorer remain unchanged. AI assistance is disclosed in the report and `AI_DISCLOSURE.md`.

## Reference-solution auxiliary supervision

`python train_reference_aux.py` tests auxiliary targets extracted only from training reference solutions: operation counts, step count, and answer-scale properties. Training-holdout selection chooses weight 1 and 20 epochs. Three-seed refitting reaches 305/355 (85.9155%) on hard validation; a 25% blend ties the earlier 309/355 by fixing one and regressing on one. No new test evaluation is performed. Logs and the research checkpoint are in `runs_reference_aux/`; this is not a replacement frozen submission. Its auxiliary prediction head is omitted from inference assets.

The audit/report package contains code, evidence and selected research/frozen checkpoints. It excludes the course data and the Python environment: restore the original `code/data/` before reproduction. Asset limits apply to the specified inference checkpoint directory and dependencies used by that method, not to the aggregate archive of unrelated experiments.

## Final experiment and stopping point

`train_final_focal.py` compares marginal-positive focal-loss exponents 0, 1 and 2 using a training-only holdout. Selection retained the no-focusing control at epoch 50 (81.5603% holdout). Three-seed refitting reached 299/355 (84.2254%) hard validation. The best fixed blend tied 309/355, with eight fixes and eight regressions. No additional official test was run, and neither frozen checkpoint was replaced. Research has stopped at the user's requested final attempt. Evidence is in `code/runs_final_focal/`.

Submission scores must match the exact frozen artifact: V2 is 394/467 = 84.3683083512%; V3 is 391/467 = 83.7259100642%. The 87.0423% development result is not an official test score.


## Follow-up experiment, 28 October 2026
Input-feature dropout probabilities 0, 0.1 and 0.3 were compared at epochs 20/50/100 on the training-only holdout. The control (probability 0, epoch 50) won at 81.5603%, versus 81.3239% for the best masked variants. Three-seed refitting scored 299/355 on hard validation; fixed blends reached at most the existing 309/355. No improvement and no new official test evaluation. See code/runs_final_masking/result.json and holdout_search.json. The six-page PDF records experiments through 26 October; this section documents the final follow-up.
