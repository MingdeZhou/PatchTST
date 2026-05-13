# PatchTST Re-implementation

This repository is a CS 4782 final project re-implementation of **A Time Series is Worth 64 Words: Long-term Forecasting with Transformers** by Nie, Nguyen, Sinthong, and Kalagnanam (ICLR 2023). It focuses on the paper's supervised multivariate long-term forecasting setting and re-implements the PatchTST model used for Weather forecasting.

## Chosen Result

The reproduced result is the supervised multivariate Weather result from Table 3 of the paper. The experiment uses look-back length `336` and prediction horizons `96`, `192`, `336`, and `720`, matching the PatchTST/42 Weather setting.

## GitHub Contents

- `code/`: re-implementation code, training runner, data loaders, metrics, and model implementation.
- `data/`: Weather benchmark data used for training and evaluation.
- `results/`: experiment logs and run summaries.
- `report/`: two-page project report source and PDF.
- `poster/`: poster placeholder for the course deliverable structure.
- `LICENSE`: project license.

## Re-implementation Details

The model keeps the official runner interface: `Model(configs).forward(x)`, where `x` has shape `[batch, seq_len, channels]` and the output has shape `[batch, pred_len, channels]`.

Implemented components:

- RevIN-style instance normalization and denormalization.
- End padding by repeating the last observed value.
- Patch extraction with `patch_len=16` and `stride=8`.
- Channel-independent processing by reshaping channels into the batch dimension.
- Linear patch embedding plus learnable positional embedding.
- Vanilla Transformer encoder blocks with multi-head attention, GELU feed-forward layers, residual connections, dropout, and BatchNorm.
- Flatten prediction head shared across channels.

Self-supervised pretraining and decomposition are outside the scope of this re-implementation.

## Setup

The experiments were run with Python 3.9, PyTorch 2.8.0+cu128, NumPy 1.26.4, pandas 2.3.3, scikit-learn 1.6.1, and an NVIDIA RTX A6000 GPU.

```bash
conda create -y -n patchtst python=3.9
conda activate patchtst
pip install -r code/requirements.txt
```

## Reproduction Steps

Run the unit smoke test:

```bash
cd code
python test_patchtst_core.py
```

Train and test one Weather horizon:

```bash
cd code
CUDA_VISIBLE_DEVICES=0 python -u run_longExp.py \
  --random_seed 2021 --is_training 1 \
  --root_path ../data/ --data_path weather.csv \
  --model_id weather_full_336_96 --model PatchTST --data custom --features M \
  --seq_len 336 --pred_len 96 --enc_in 21 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --dropout 0.2 --fc_dropout 0.2 --head_dropout 0 \
  --patch_len 16 --stride 8 --des FullReimpl \
  --train_epochs 10 --patience 3 --itr 1 \
  --batch_size 128 --learning_rate 0.0001 --num_workers 0
```

Run checkpoint-load inference for the same setting:

```bash
CUDA_VISIBLE_DEVICES=0 python -u run_longExp.py \
  --random_seed 2021 --is_training 0 \
  --root_path ../data/ --data_path weather.csv \
  --model_id weather_full_336_96 --model PatchTST --data custom --features M \
  --seq_len 336 --pred_len 96 --enc_in 21 \
  --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
  --dropout 0.2 --fc_dropout 0.2 --head_dropout 0 \
  --patch_len 16 --stride 8 --des FullReimpl \
  --train_epochs 10 --patience 3 --itr 1 \
  --batch_size 128 --learning_rate 0.0001 --num_workers 0
```

To reproduce the remaining horizons, change `--pred_len` and the matching `--model_id` suffix to `192`, `336`, or `720`.

## Results

Full Weather runs used real benchmark data, `seq_len=336`, `patch_len=16`, `stride=8`, `train_epochs=10`, and `patience=3`. The paper values are PatchTST/42 results from Table 3.

| Prediction Length | Paper MSE | Paper MAE | Reimplementation MSE | Reimplementation MAE |
| ---: | ---: | ---: | ---: | ---: |
| 96 | 0.152 | 0.199 | 0.1578 | 0.2082 |
| 192 | 0.197 | 0.243 | 0.2024 | 0.2485 |
| 336 | 0.249 | 0.283 | 0.2532 | 0.2863 |
| 720 | 0.320 | 0.335 | 0.3254 | 0.3392 |

All checkpoint-load inference runs matched the post-training test metrics. Prediction arrays were generated and checked during the run; the compact final repository keeps the logs and result summaries in `results/full_weather_reimpl/`.

## Conclusion

The re-implementation closely reproduces the Weather results from the original paper while using a shorter training budget. The project supports the paper's main insight: patching gives time-series tokens meaningful local context, and channel independence makes a vanilla Transformer effective for multivariate forecasting.

## References

- Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." ICLR 2023.
- Official PatchTST repository: https://github.com/yuqinie98/PatchTST
- Autoformer/PatchTST benchmark datasets linked by the official repository.
