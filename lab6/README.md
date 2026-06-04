# Lab6: 基于AI Coding的语音识别

## 实验概述

本实验使用 **CTC + Transformer Encoder** 构建端到端语音识别系统，将语音信号转换为拼音序列。

## 目录结构

```
lab6/
├── README.md                    # 本文件
├── lab6 - 基于AI Coding的语音识别.pdf  # 实验文档
├── experiment_report.md         # 实验报告模板
│
├── train.py                     # 训练脚本
├── inference.py                 # 推理脚本
│
├── dataprocess/                 # 数据处理模块
│   ├── __init__.py
│   ├── extract.py              # Fbank特征提取
│   ├── vocab.py                # 词表管理
│   └── dataset.py              # 数据集加载
│
├── model/                       # 模型定义
│   └── __init__.py             # Transformer + CTC 模型
│
├── tokenizer/                   # 分词器
│   ├── __init__.py
│   ├── tokenizer.py
│   └── vocab.txt
│
├── dataset/                     # 数据集
│   └── split/
│       ├── train/              # 训练集
│       │   ├── wav.scp
│       │   └── pinyin
│       ├── dev/                # 验证集
│       └── test/               # 测试集
│
├── checkpoints/                 # 模型检查点（训练后生成）
└── logs/                        # TensorBoard日志（训练后生成）
```

## 快速开始

### 环境准备

```bash
# 安装依赖
pip install torch torchaudio tensorboard jiwer tqdm numpy

# 或使用requirements.txt（如有）
pip install -r requirements.txt
```

### 模型训练

```bash
# 使用默认参数训练
python train.py

# 自定义参数训练
python train.py \
    --batch_size 16 \
    --num_epochs 50 \
    --learning_rate 1e-4 \
    --d_model 512 \
    --nhead 8 \
    --num_encoder_layers 6

# 查看训练日志
tensorboard --logdir logs
```

### 模型推理

```bash
# 使用最佳模型进行推理
python inference.py --model_path checkpoints/best_model.pth

# 指定测试集
python inference.py \
    --model_path checkpoints/best_model.pth \
    --test_wav_scp dataset/split/test/wav.scp \
    --test_text dataset/split/test/pinyin
```

### 单条音频推理

```python
from inference import inference_single
from model import ASRTransformerCTC
from dataprocess.vocab import Vocab, build_pinyin_list_from_text
import torch

# 加载模型和词表
vocab = Vocab(build_pinyin_list_from_text("dataset/split/train/pinyin"))
model = ASRTransformerCTC(vocab_size=vocab.vocab_size)
checkpoint = torch.load("checkpoints/best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])

# 推理
audio_path = "path/to/your/audio.wav"
pred_pinyin = inference_single(model, audio_path, vocab, device="cpu")
print("预测拼音:", ' '.join(pred_pinyin))
```

## 模型架构

```
输入: Fbank特征 [batch_size, seq_len, 80]
         ↓
卷积下采样 (2层CNN, 4倍压缩)
         ↓
位置编码 (正弦/余弦)
         ↓
Transformer Encoder × 6层
         ↓
CTC Head (Linear + LogSoftmax)
         ↓
输出: log_probs [seq_len//4, batch_size, vocab_size]
```

## 超参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| batch_size | 16 | 批次大小 |
| learning_rate | 1e-4 | 学习率 |
| num_epochs | 50 | 训练轮数 |
| d_model | 512 | 模型维度 |
| nhead | 8 | 注意力头数 |
| num_encoder_layers | 6 | Encoder层数 |
| dim_feedforward | 2048 | 前馈网络维度 |
| dropout | 0.1 | Dropout比率 |
| input_dim | 80 | Fbank特征维度 |

## 评估指标

- **CER (Character Error Rate)**：字符错误率
  - 计算公式：CER = (S + D + I) / N
  - S: 替换错误, D: 删除错误, I: 插入错误, N: 参考序列长度
  - **CER越低，模型性能越好**

## 常见问题

### 1. 显存不足 (Out of Memory)

```bash
# 减小batch_size
python train.py --batch_size 8

# 或减小模型规模
python train.py --d_model 256 --num_encoder_layers 4
```

### 2. 训练速度慢

- 检查是否使用了GPU：`torch.cuda.is_available()`
- 减少数据加载线程数：`--num_workers 2`
- 使用混合精度训练（需要修改代码）

### 3. CER不下降

- 检查学习率是否合适
- 检查数据是否正确加载
- 尝试调整模型规模

## 参考资料

- [CTC论文](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
- [Transformer论文](https://arxiv.org/abs/1706.03762)
- [torchaudio文档](https://pytorch.org/audio/stable/index.html)
