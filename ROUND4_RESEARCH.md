# Fourth research round: local reasoning evidence

The 90% official-test target has not been achieved. The frozen submission remains V2, with the previously measured official score of 394/467 (84.3683%). This round did not load test references or perform another test evaluation.

## Completed experiments

We investigated evidence beyond global candidate embeddings:

1. A local-evidence calibrator checks small linear equation systems for consistency, tracks whether intermediate values support the final answer, compares question wording with the candidate ending, and detects self-correction language. CatBoost depth and tree count were selected on the fixed training development holdout.
2. An answer-conditioned text classifier replaces numeric literals with their roles (question quantity, proposed final value, or derived value). Word n-grams and question/solution overlap features are learned from training data. A regularized logistic classifier combines these with an out-of-fold neural-verifier prior.

Both use training-fold checkpoints from round two for their prior. Final fitting uses training data only. The official validation set is used for development evaluation and fixed 25%/50% blend comparisons.

| Method | Hard validation correct / 355 | Hard accuracy |
|---|---:|---:|
| Frozen V2 | 307 | 86.4789% |
| Local-evidence calibrator | 302 | 85.0704% |
| 25% local-evidence / 75% V2 score blend | 308 | 86.7606% |
| Answer-conditioned text classifier with prior | 308 | 86.7606% |

The best development result is one additional correct question, with no regressions among the 355 disagreement questions. Overall validation accuracy is 751/798 (94.1103%); this includes many agreement questions and must not be presented as the official test accuracy. The exploratory paired-bootstrap interval for the hard-validation gain is [0, 0.8451] percentage points. It does not account for repeated selection and is not independent evidence of generalization. No new official test score is justified by this small gain.

## Development limitations

The same 423-question training holdout and official validation set have been reused across research rounds. The OOF prior is out-of-fold for each example, but the subsequent calibration holdout is not fully nested: teacher models supplying calibration-training scores can have seen other questions in that holdout. Its selected holdout numbers are therefore only development diagnostics. They are not clean generalization estimates. The known aggregate V2 test score was not used to choose these configurations, and test examples/errors were not inspected.

## Prepared but NOT trained

`neural_local.py`, `local_text.py`, `train_local.py`, and `selector_local.py` implement a randomly initialized text CNN and a permutation-equivariant set-attention verifier. They do not use pretrained weights. Syntax checks passed, but the neural models have **not** run and have no measured accuracy, checkpoint, CPU timing, or eligibility claim.

Installing PyTorch was attempted using the official wheel index: CUDA 2.8 (3.46 GB), CPU 2.8 (619 MB), then CPU 2.6 (206 MB). Downloads failed to make sufficient progress during this session and were explicitly interrupted. The GPU was not used for training. These pending models are included only as unvalidated research source. No download or training process was intentionally left running.

## Checks and reproduction

An implicit-multiplication parser bug spanning a newline was discovered and fixed before the final calibrator run. Checks distinguish consistent `x+108=160; x=52` from contradictory `x+108=160; x=53`, recognize `2x=10`, and verify role tokenization. All frozen V2 inference-file and checkpoint hashes still match `code/runs_v2/frozen_manifest.json`.

With the existing environment, run from `code/`:

```text
python train_trace.py
python train_role_text.py
python compare_round4.py
python check_round4.py
```

The two training scripts require the existing `runs_stack/model_bundle/model.pkl`; comparison requires the frozen V2 bundle. Logs and research checkpoints are saved under `runs_trace`, `runs_role_text`, and `runs_round4`. They are supplementary development artifacts, not replacement submission assets. Current submission instructions remain in `README_V2.md`.

To execute the pending neural research after successfully installing a compatible PyTorch version, run `python train_local.py --kind set` and `python train_local.py --kind cnn`, then repeat the comparison. Any resulting model still needs inference resource checks and a justified evaluation decision before replacing the frozen submission.

OpenAI Codex substantially assisted with the methods, implementation, experiments, checks, and documentation in this round. Existing course attribution and AI disclosure continue to apply.
