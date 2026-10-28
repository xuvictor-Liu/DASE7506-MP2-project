# Eighth round and final-score record

The requested official test target of at least 87% has **not** been achieved. On 467 test questions, at least **407 correct (87.1520%)** are required; 406/467 is only 86.9379%.

## Completed final evaluation of the previously selected model

Before this research round, the fifth-round model was selected and frozen using its development result of 309/355 hard validation questions. Its checkpoint and source hashes were written before test scoring. One fresh prediction process and an independent official scoring process produced:

- **V3: 391/467 = 83.7259100642%**.
- Fresh prediction plus scoring: **4.5102 seconds**, four numerical-library threads, Intel Core Ultra 7 251HX.
- Additional inference assets: **9,724,630 bytes**, below 16 MiB.
- Frozen hashes unchanged after evaluation. Reference-CPU performance remains unverified.

The older V2 historical result is **394/467 = 84.3683083512%**. These are scores for different checkpoints and must not be interchanged. V3's better development result did not transfer to the test set. No per-question test errors were inspected, and no test scores were used as the criterion for the new training searches below. The earlier self-imposed 90% validation gate was explicitly replaced before the V3 test score was observed, as recorded in `code/runs_v3/frozen_manifest.json`. The course restriction reserving test references for final scoring remained in force.

V3 inference uses `selector_semantic_blend.py` and `runs_v3/final_bundle/model.pkl`. Its official result and timing are in `runs_v3/test-score.json` and `runs_v3/resources.json`.

## New experiments

| Method | Training hard holdout | Hard validation |
|---|---:|---:|
| Previous best development model | Not comparable: already fit on all train | 309/355 = 87.0423% |
| Eight-model seed/subsample ensemble | 83.2151% | 301/355 = 84.7887% |
| Learned context/candidate interaction | 83.4515% | 305/355 = 85.9155% |

The ensemble search compared 90% and 100% question-level subsamples, epochs 20/50/100, and four seeds on the existing 423-question training holdout. The selected setting was full training fraction and 50 epochs. The final ensemble used eight prespecified seeds 51–58. These seeds were not selected on validation. Each member is a 64-unit pairwise ranker with its own training statistics.

The context model adds a learned multiplicative interaction between the candidate-group mean embedding and each candidate's centered embedding. Ranks 16/64 and epochs 20/50/100 were compared on the training holdout; rank 16 and epoch 100 were selected. The final model averages seeds 41, 42, and 43. NumPy-export predictions were checked against the PyTorch network during training.

Both methods were compared with fixed 25% and 50% blends against V2 and the previous development model. None exceeded the previous 309/355 result. The full comparisons are in `code/runs_bagged/result.json` and `code/runs_round8/comparison.json`.

## Limits and reproducibility

The training holdout and validation set have been reused many times. Their scores are development evidence, not independent generalization estimates. No fresh test evaluation was justified by these unsuccessful new experiments. Repeatedly scoring candidate methods on test until one reaches 87% would use the test set for selection, contrary to the course guide, and was not performed.

Run from `code/` with the existing environment and course data:

```text
python train_bagged.py
python train_context.py
python compare_round8.py
```

The new scripts require preserved V2 and fifth-round checkpoints for comparisons. All training used CPU with four numerical-library threads. The new model sources, search settings, trained checkpoints, timings, and results are saved under `runs_bagged`, `runs_context`, and `runs_round8`. No new models replaced the previously frozen checkpoints. All jobs in this round finished.

The goal remains unresolved; these experiments do not establish that 87% is impossible, and they do not supply a legitimate 87% test result. OpenAI Codex assisted substantially with implementation, training, checks, and documentation. Original course attribution and AI disclosure continue to apply.
