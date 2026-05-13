# PatchTST Re-implementation Run Log

## Environment

- Conda env: `patchtst`
- Python: 3.9
- Main packages: `torch 2.8.0+cu128`, `numpy 1.26.4`, `pandas 2.3.3`, `scikit-learn 1.6.1`
- GPU used for experiments: `NVIDIA RTX A6000`

## Data

- Dataset: Weather benchmark from the official PatchTST/Autoformer data source.
- Final repository path: `data/weather.csv`
- Shape: 52,696 rows, 21 weather variables plus the `date` column.

## Official Baseline Smoke Test

Before reorganizing the repository, the original authors' supervised code was cloned and used only to validate the official data/training/testing flow.

Result for Weather, `seq_len=336`, `pred_len=96`, 1 epoch:

- MSE: `0.22151042520999908`
- MAE: `0.2715323567390442`
- Prediction shape: `(10368, 96, 21)`

## Re-implementation Smoke Test

The first re-implementation smoke test used the same Weather setting for 1 epoch.

- MSE: `0.4007863402366638`
- MAE: `0.38288021087646484`
- Prediction shape: `(10368, 96, 21)`
- Checkpoint-load inference matched the post-training test metrics.

## Full Weather Re-implementation Run

Final supervised multivariate Weather experiments:

- Code path: `code/`
- Data path: `data/weather.csv`
- Model implementation: `code/models/patchtst_core.py`
- `seq_len=336`
- `pred_len in {96, 192, 336, 720}`
- `patch_len=16`
- `stride=8`
- `train_epochs=10`
- `patience=3`
- separate checkpoint-load inference pass after each training run

Logs are in `results/full_weather_reimpl/`.

| pred_len | MSE | MAE | RSE | pred.npy shape |
| ---: | ---: | ---: | ---: | --- |
| 96 | 0.1577966362 | 0.2082100362 | 0.5233098865 | `(10368, 96, 21)` |
| 192 | 0.2024025321 | 0.2485375553 | 0.5918664932 | `(10240, 192, 21)` |
| 336 | 0.2531976104 | 0.2862838805 | 0.6609854102 | `(10112, 336, 21)` |
| 720 | 0.3253938854 | 0.3391868472 | 0.7525272369 | `(9728, 720, 21)` |

All checkpoint-load inference metrics matched the post-training test metrics. Prediction arrays were generated and checked during the run; the final compact repository keeps logs rather than large `.npy` outputs.
