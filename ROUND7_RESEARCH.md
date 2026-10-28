# Seventh research round — 26 October 2026

This round did not improve the previous best. The retained development model remains **309/355 = 87.0423% hard-validation accuracy** (`runs_round5/development_bundle`). The previously measured official test score remains **394/467 = 84.3683%**, associated with frozen V2. No additional test evaluation was performed. The requested 90% official-test target remains unmet.

## Experiments completed

| Method | Hard validation correct | Hard validation accuracy |
|---|---:|---:|
| Previous best development blend | **309/355** | **87.0423%** |
| Pairwise ranker selected from perturbation-training search | 306/355 | 86.1972% |
| Role-text semantic features + hard-only pairwise ranking | 305/355 | 85.9155% |
| Calculation-quantity coverage calibrator | 304/355 | 85.6338% |

The new rankers were also compared with fixed 25% and 50% normalized-score blends against V2 and the previous best development model. None exceeded 309/355. Quantity-calibrator blends both scored 308/355. Existing model/checkpoint files were not replaced.

## Perturbation training and focused ranking

`train_robust.py` compares parameter-perturbation radii 0, 0.05, and 0.15 with snapshots at epochs 20, 50, and 100. It trains a 96-unit ReLU network on hard training questions using a pairwise logistic ranking loss. The second gradient evaluation uses the same dropout realization as the first. Weights are exported as NumPy arrays for inference; PyTorch is needed for training only.

Configuration selection uses the existing 423-question training-only hard holdout (seed 731). Radius 0 and radius 0.15 both reached 83.6879% at epoch 50. The deterministic first-best rule selected the **unperturbed control**, so the final result must not be described as evidence that parameter perturbation improved accuracy. The selected configuration was refit on all hard training questions with seeds 41, 42, and 43, then evaluated on validation. Total search/refit time was 42.80 seconds on CPU.

`train_semantic_rank.py` applies that training-selected configuration to the earlier role-text/SVD features, while fitting the ranking network only on hard questions. Text preprocessing uses training inputs, including easy training questions; no validation inputs are used to fit it. Its training holdout score was 82.5059%, and its final validation score was 305/355. Total time was 57.50 seconds. This tests a different supervision objective from the previous semantic model's hard/easy multitask loss.

## Training-error inspection and quantity usage

Eight early training examples where both cross-fitted semantic and CNN experts were wrong were inspected. Examples included omitted multiplicative factors, incorrect interpretation of a percentage increase, and a wrong aggregation scope. These observations motivated input-only features measuring whether question quantities appear in actual calculation lines and on left-hand sides, rather than merely being repeated in the stated givens. Common small number words are converted to numeric values. No sample IDs or question-specific answers are used by the features.

`train_quantity_usage.py` trains a fixed C=0.1 standardized logistic calibrator on hard training candidates, using the existing two-fold out-of-fold expert scores plus quantity-use features. There is no hyperparameter search for this calibrator. At validation time it uses the existing full-training semantic/CNN ensembles, which differ in training size and ensemble count from the OOF experts. This distribution mismatch is a limitation.

The calibrator fixed seven previous-best errors but regressed on twelve correct choices. A 25% blend fixed two and regressed on three; a 50% blend fixed five and regressed on six. Therefore none was retained. Quantity coverage is only a heuristic: some numbers are irrelevant, some are used implicitly, and a candidate can contain faulty reasoning yet have the correct final answer under the official metric. The checks do not prove correctness or incorrectness.

## Checks, interpretation, and reproduction

Pairwise loss direction, invalid-candidate masking, candidate permutation, calculation-only quantity coverage, and Python syntax checks passed. The selected previous model's zero bootstrap difference against itself is a mechanical comparison, not evidence of a generalization improvement. All development sets have been reused across many rounds; small validation differences are not independent evidence.

Source and logs:

```text
python train_robust.py
python train_semantic_rank.py
python compare_round7.py
python train_quantity_usage.py
```

Run from `code/` with `requirements-round5.txt` and the unchanged course data. The quantity experiment needs the prior `runs_gate/training_oof_scores.npz`, `runs_local_cnn/model_bundle/model.pt`, and fifth-round development bundle. The comparison needs preserved V2 and fifth-round checkpoints. Results are under `runs_robust`, `runs_semantic_rank`, `runs_quantity_usage`, and `runs_round7`. The seventh-round archive supplements earlier research packages and is not a new submission checkpoint.

All runs used CPU with four numerical-library threads. No external training data, pretrained weights, API reasoning calls, test-error analysis, or test-driven parameter selection was used. All processes for this round finished. The prior V2 submission package and fifth-round development package remain the designated artifacts for their respective measured results.

OpenAI Codex substantially assisted with design, implementation, experiments, training-error inspection, and documentation. Original course attribution and AI disclosure continue to apply.
