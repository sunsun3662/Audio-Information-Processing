#!/bin/bash
# Lab6 语音识别训练脚本

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 创建必要的目录
mkdir -p checkpoints logs results

# 开始训练
echo "=========================================="
echo "开始训练语音识别模型"
echo "=========================================="

python train.py \
    --train_wav_scp dataset/split/train/wav.scp \
    --train_text dataset/split/train/pinyin \
    --dev_wav_scp dataset/split/dev/wav.scp \
    --dev_text dataset/split/dev/pinyin \
    --batch_size 16 \
    --num_epochs 50 \
    --learning_rate 1e-4 \
    --d_model 512 \
    --nhead 8 \
    --num_encoder_layers 6 \
    --dim_feedforward 2048 \
    --dropout 0.1 \
    --save_dir checkpoints \
    --log_dir logs \
    --seed 42

echo "=========================================="
echo "训练完成！"
echo "=========================================="

# 查看最佳模型
if [ -f "checkpoints/best_model.pth" ]; then
    echo "最佳模型已保存到: checkpoints/best_model.pth"
fi
