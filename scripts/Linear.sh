export CUDA_VISIBLE_DEVICES=1

model=Linear_BatchNorm

# ETTh1
python -u run.py \
  --root_path ./data/ETT-small/ \
  --data_path ETTh1.csv \
  --model $model \
  --data ETTh1 \
  --features M \
  --enc_in 7 \
  --seq_len 96 \
  --pred_len 96

# ETTh2
python -u run.py \
  --root_path ./data/ETT-small/ \
  --data_path ETTh2.csv \
  --model $model \
  --data ETTh2 \
  --features M \
  --enc_in 7 \
  --seq_len 96 \
  --pred_len 96

# ETTm1
python -u run.py \
  --root_path ./data/ETT-small/ \
  --data_path ETTm1.csv \
  --model $model \
  --data ETTm1 \
  --features M \
  --enc_in 7 \
  --seq_len 96 \
  --pred_len 96

# ETTm2
python -u run.py \
  --root_path ./data/ETT-small/ \
  --data_path ETTm2.csv \
  --model $model \
  --data ETTm2 \
  --features M \
  --enc_in 7 \
  --seq_len 96 \
  --pred_len 96

# ECL
python -u run.py \
  --root_path ./data/electricity/ \
  --data_path electricity.csv \
  --model $model \
  --data Electricity \
  --features M \
  --enc_in 321 \
  --seq_len 96 \
  --pred_len 96

# Exchange
python -u run.py \
  --root_path ./data/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model $model \
  --data Exchange \
  --features M \
  --enc_in 8 \
  --seq_len 96 \
  --pred_len 96

# Traffic
python -u run.py \
  --root_path ./data/traffic/ \
  --data_path traffic.csv \
  --model $model \
  --data Traffic \
  --features M \
  --enc_in 862 \
  --seq_len 96 \
  --pred_len 96

# Weather
python -u run.py \
  --root_path ./data/weather/ \
  --data_path weather.csv \
  --model $model \
  --data Weather \
  --features M \
  --enc_in 21 \
  --seq_len 96 \
  --pred_len 96

# ILI
python -u run.py \
  --root_path ./data/illness/ \
  --data_path national_illness.csv \
  --model $model \
  --data ILI \
  --features M \
  --enc_in 7 \
  --seq_len 36 \
  --pred_len 24
