# Fifth research round: small development improvement

The best development model now selects **309/355 hard validation questions correctly (87.0423%)**, compared with round four's 308/355 (86.7606%) and frozen V2's 307/355 (86.4789%). This is a small development improvement, not a demonstrated improvement in official test accuracy. The 90% official-test target remains unmet. The frozen official submission remains V2: **394/467 = 84.3683%**. No test scoring or test-error inspection was performed in this round.

## Completed experiments

| Method | Training hard holdout | Hard validation | Final training/search seconds |
|---|---:|---:|---:|
| Nonlinear role-text semantic fusion | 83.4515% | 304/355 = 85.6338% | 173.36 |
| Small learned attention, NumPy implementation | 82.9787% | 299/355 = 84.2254% | 65.20 |
| Two-layer set-attention network, PyTorch | 81.0875% | 298/355 = 83.9437% | 179.27 |
| Role-conditioned text CNN, PyTorch | 82.0331% | 304/355 = 85.6338% | 380.12 |
| **75% prior role-text model + 25% semantic model** | Selected on validation | **309/355 = 87.0423%** | See components above |

Search times exclude dependency installation and an interrupted preliminary 48-channel CNN run. Some independent training jobs overlapped, with four numerical-library threads per job. All training in this round used CPU on the Intel Core Ultra 7 251HX; no GPU training occurred. The final CNN uses 24-dimensional token embeddings and three 24-channel convolutions with widths 3, 5, and 7. No pretrained weights or external training examples were used.

The semantic model fits a 12,000-feature role-normalized word n-gram vocabulary and a 64-dimensional truncated SVD on training text. Its projection and within-question differences are concatenated with supplied embeddings and existing structural features, then passed through a 96-unit neural verifier. It averages seeds 41, 42, and 43. Training-only holdout selection chose 50 epochs and auxiliary correctness-loss weight 0.1.

The small attention model learns which candidates should contribute to a shared context and scores interactions with that context. Its analytic gradients were checked against finite differences, and candidate-permutation equivariance was checked. The larger set model also passed a permutation check. The CNN learns local text patterns jointly with the existing supplied numerical representation. PyTorch models average seeds 41 and 42; selected dropout/epochs were 0.4/15 for set attention and 0.15/30 for CNN.

## What improved and what did not

The final development blend combines the round-four role-text classifier with the new semantic model after normalizing scores within each question. Relative to V2, it fixes three hard validation questions and regresses on one, for a net gain of two. Compared with the previous round's best count, the gain is one question. CNN and attention variants did not improve the best blend. All fixed 25% and 50% comparisons are retained in `code/runs_round5/comparison.json`.

Overall validation accuracy is **752/798 = 94.2356%**. That bank contains easier agreement questions, so this number must not be substituted for the leaderboard metric. The relevant disagreement subset is 309/355. The exploratory paired-bootstrap gain interval against V2 is **[-0.5634, 1.6901] percentage points**, which includes zero and does not account for repeated model selection.

The same training development holdout and official validation set have been reused across rounds. New semantic/attention/CNN model preprocessing and fitting use only the appropriate training portion, but the reused role-text prior has the non-nested calibration-development limitation documented in ROUND4_RESEARCH.md. Consequently, the small gain is provisional. This round does not justify claiming 87.04% test accuracy or running another test solely to select between near-tied models.

## Runnable development checkpoint

Entrypoint: `code/selector_semantic_blend.py`.

Checkpoint: `code/runs_round5/development_bundle/model.pkl`.

It contains all role-text, V2, and semantic inference components. Asset size is **9,724,630 bytes (9.274 MiB)**. An independently launched prediction process and scoring process reproduced 752/798 on validation in **6.650 seconds** on the local CPU with four inference-library threads. Other training was active during this timing, so it is a local development measurement, not a reference-CPU certification. Test-bank eligibility and official scoring have not been newly measured for this checkpoint.

From `code/`, with the existing Python environment:

```text
python predict.py --method selector_semantic_blend.py --checkpoint runs_round5/development_bundle --split validation --output runs_round5/reproduced-validation.json
python score_predictions.py --predictions runs_round5/reproduced-validation.json --split validation
```

Use `requirements-method.txt` for the selected development blend. The additional neural research requires `requirements-round5.txt`; exact environment versions are saved in `requirements-round5-lock.txt`.

To repeat the new training and comparison:

```text
python train_semantic.py
python train_attention.py
python train_local.py --kind set
python train_local.py --kind cnn
python compare_round5.py
python benchmark_round5.py
python check_round4.py
```

The comparison additionally requires preserved V2 and round-four role-text checkpoints. Train/validation data are the unchanged course files. The research archive includes inference source dependencies and the development checkpoints, but not course data or PyTorch installation binaries.

## Dependency and preservation checks

The previously blocked PyTorch installation was resolved by downloading the official PyPI wheel in small HTTP ranges and verifying the full SHA-256 against PyPI metadata. Installed version: **torch 2.6.0+cpu**. The official Windows/Python 3.12 wheel digest was `7e1448426d0ba3620408218b50aa6ada88aeae34f7a239ba5431f6c8774b1239`. This is a runtime library, not pretrained model weights. Both previously pending neural experiments are now completed.

All frozen V2 source and checkpoint hashes still match its original manifest. The standalone development selector's text encoding matches the training implementation. The original data, parser, runner, and scorer were not edited. Development artifacts are supplementary and do not replace the existing V2 submission ZIP or its reported official score.

OpenAI Codex substantially assisted with the experiment design, implementation, dependency repair, training, evaluation, and this research record. Existing course attribution and AI disclosure continue to apply.
