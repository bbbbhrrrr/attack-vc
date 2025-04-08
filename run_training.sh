#!/bin/bash

# 安装必要的依赖项
pip install torch torchaudio librosa matplotlib tqdm

# 创建数据目录
mkdir -p ./data

# 运行训练脚本
python train_predictive_model.py \
    --data_path "./data/LibriSpeech" \
    --dataset_type "librispeech" \
    --output_dir "./models" \
    --n_mels 512 \
    --input_length 100 \
    --output_length 32 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --epochs 50 \
    --seed 42 \
    --num_workers 4
