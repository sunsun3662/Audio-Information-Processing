#!/bin/bash
# Lab6 语音识别推理脚本

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 创建结果目录
mkdir -p results

# 检查模型文件是否存在
MODEL_PATH="checkpoints/best_model.pth"
if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件不存在: $MODEL_PATH"
    echo "请先运行训练脚本: bash run_train.sh"
    exit 1
fi

echo "=========================================="
echo "开始推理测试"
echo "=========================================="

python inference.py \
    --test_wav_scp dataset/split/test/wav.scp \
    --test_text dataset/split/test/pinyin \
    --train_text dataset/split/train/pinyin \
    --model_path $MODEL_PATH \
    --batch_size 32 \
    --output_dir results

echo "=========================================="
echo "推理完成！"
echo "=========================================="

# 显示结果统计
if [ -f "results/stats.txt" ]; then
    echo ""
    echo "结果统计:"
    cat results/stats.txt
fi
