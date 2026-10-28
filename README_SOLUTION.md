# DASE7506 MP2: compact candidate-comparison ranker

Frozen official test accuracy: **81.79871520342613% (382/467)**.
Validation accuracy: **93.3584% (745/798)**; validation disagreement subset: **85.0704% (302/355)**.
The test set was scored only after selecting and hashing the method using validation.

## Reproduce the submitted checkpoint

Use Python 3.12. From `code/`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy==2.2.6
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:NUMEXPR_NUM_THREADS='4'
.\.venv\Scripts\python.exe check_data.py
.\.venv\Scripts\python.exe predict.py --method selector.py --checkpoint runs/model_bundle --split test --output runs/reproduced-test-predictions.json
.\.venv\Scripts\python.exe score_predictions.py --split test --predictions runs/reproduced-test-predictions.json --output runs/reproduced-test-score.json
```

On Linux, use `.venv/bin/python` and `export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4`. The final neural checkpoint requires only NumPy; do not install training dependencies for inference unless needed. Load the bundled pickle only from this trusted project.

`selector.py` accepts any aligned bank with four candidates and the supplied embedding dimension, including the official audit interface. It does not read labels or use question IDs. The original `submission.py` remains an untouched interface template; use `--method selector.py`.

## Reproduce training and development

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-method.txt
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe train_neural.py
.\.venv\Scripts\python.exe verify_method.py
```

Run those two training scripts in order: stage one writes the linear winner, stage two compares neural models and blends against it. Training intentionally overwrites the model bundle. Preserve the distributed frozen bundle before retraining if you want to compare hashes. The final selected model is seed 41, epoch 40, one 64-unit ReLU hidden layer; Adam learning rate 0.0005, dropout 0.15, L2 coefficient 0.002. Seeds 41/42/43 and epochs 10/20/40/60 were evaluated. Stage one uses seed 42 and the grids recorded in `train.py`.

`evaluate_final.py` hashes the chosen assets before launching fresh official prediction and scoring processes. Run only after freezing your method; it scores both validation and test. `analyze_validation.py` generates descriptive validation error and confidence analysis without tuning. `verify_method.py` checks batch independence, permutation answer consistency on 24 validation groups, and the 16 MiB asset bound.

## Artifacts and measured resources

- `code/runs/model_bundle/model.pkl`: all inference parameters and normalization statistics, 608,055 bytes (0.5799 MiB).
- `code/runs/test-score.json`: official submitted accuracy field.
- `code/runs/frozen_manifest.json`: SHA-256 hashes of inference code and model.
- `code/runs/experiments.json`, `neural_experiments.json`: every development configuration and validation result.
- `code/runs/resources.json`: fresh process prediction plus scoring timing and hardware.
- `code/requirements-lock.txt`: exact installed development environment.
- `report.pdf`: experiment report, below the ten-page limit.
- `AI_DISCLOSURE.md`: substantive AI assistance and reused assets.

Measured on Windows 11, Python 3.12.10, Intel Core Ultra 7 251HX, four library threads; no GPU training. Full test prediction plus independent scoring took 0.6585 seconds locally. The course reference Xeon determines official eligibility. Stage-one search took 27.82 seconds and stage-two search 27.92 seconds after imports, including data preparation; installation, process startup, and artifact authoring are excluded from these training timer fields.

The final architecture has 2,300 input features: raw 1,024-dimensional embeddings, within-question centered embeddings, and 252 structural features. Normalization is fit on training groups containing at least one incorrect candidate. Inputs are clipped to [-10, 10]. Correct candidates share a normalized target distribution for a four-way softmax training objective. At prediction, invalid numeric answers are masked and the highest-scoring valid candidate is returned.

## Submission still requiring the student's action

Review the report and implementation, add your name/student ID if desired, and publish this project with its matching frozen checkpoint in a publicly accessible repository. Record the submitted Git commit. Supply the code URL, checkpoint URL, and **81.79871520342613** as the accuracy percentage on the course submission website. The website prepares a GitHub issue; the issue must actually be created to complete submission. No online submission has been made by this project setup.

The results do not establish a global optimum. The selected configuration was compared on 355 hard validation questions, so differences of a few questions can reflect selection noise. Test references must not be used for subsequent tuning or error-driven changes.
