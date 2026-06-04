#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lab6 语音识别推理脚本
功能：加载训练好的模型，对测试集进行推理并计算CER
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import csv

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataprocess.vocab import Vocab, build_pinyin_list_from_text
from dataprocess.dataset import ASRDataset, collate_fn
from dataprocess.extract import extract_fbank
from model import ASRTransformerCTC


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="语音识别模型推理")

    # 数据相关
    parser.add_argument("--test_wav_scp", type=str, default="dataset/split/test/wav.scp",
                        help="测试集wav.scp路径")
    parser.add_argument("--test_text", type=str, default="dataset/split/test/pinyin",
                        help="测试集拼音标签路径")
    parser.add_argument("--train_text", type=str, default="dataset/split/train/pinyin",
                        help="训练集拼音标签路径（用于构建词表）")

    # 模型相关
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth",
                        help="模型检查点路径")
    parser.add_argument("--input_dim", type=int, default=80, help="Fbank特征维度")
    parser.add_argument("--d_model", type=int, default=512, help="Transformer模型维度")
    parser.add_argument("--nhead", type=int, default=8, help="注意力头数")
    parser.add_argument("--num_encoder_layers", type=int, default=6, help="Encoder层数")
    parser.add_argument("--dim_feedforward", type=int, default=2048, help="前馈网络维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout比率")

    # 推理相关
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--output_dir", type=str, default="results", help="结果输出目录")

    return parser.parse_args()


def compute_cer(reference, hypothesis):
    """
    计算字符错误率 (Character Error Rate)
    :param reference: list[str], 参考序列
    :param hypothesis: list[str], 假设序列
    :return: float, CER值
    """
    # 计算编辑距离
    d = np.zeros((len(reference) + 1, len(hypothesis) + 1), dtype=np.int32)

    for i in range(len(reference) + 1):
        d[i][0] = i
    for j in range(len(hypothesis) + 1):
        d[0][j] = j

    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            if reference[i-1] == hypothesis[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    # CER = 编辑距离 / 参考序列长度
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0

    return d[len(reference)][len(hypothesis)] / len(reference)


def greedy_decode(log_probs, vocab):
    """
    贪心解码：将模型输出转换为拼音序列
    :param log_probs: [seq_len, vocab_size], 模型输出的对数概率
    :param vocab: Vocab实例
    :return: list[str], 解码后的拼音序列
    """
    # 1. 取每个时间步概率最大的类别
    _, indices = torch.max(log_probs, dim=-1)  # [seq_len]
    indices = indices.tolist()

    # 2. 合并相邻重复字符
    merged = []
    prev_idx = None
    for idx in indices:
        if idx != prev_idx:
            merged.append(idx)
            prev_idx = idx

    # 3. 移除blank符
    decoded = [vocab.itos[idx] for idx in merged if idx != vocab.blank_id]

    return decoded


@torch.no_grad()
def inference_single(model, audio_path, vocab, device):
    """
    对单条音频进行推理
    :param model: 模型
    :param audio_path: 音频文件路径
    :param vocab: Vocab实例
    :param device: 设备
    :return: list[str], 预测的拼音序列
    """
    model.eval()

    # 提取Fbank特征
    fbank = extract_fbank(audio_path)  # [num_frames, 80]

    # 添加batch维度
    fbank = fbank.unsqueeze(0).to(device)  # [1, num_frames, 80]
    fbank_length = torch.tensor([fbank.size(1)], dtype=torch.long).to(device)

    # 前向传播
    log_probs, output_length = model(fbank, fbank_length)

    # 贪心解码
    pred_pinyin = greedy_decode(log_probs[:, 0, :], vocab)

    return pred_pinyin


@torch.no_grad()
def inference_batch(model, dataloader, vocab, device):
    """
    对测试集进行批量推理
    :return: 所有样本的预测结果和CER
    """
    model.eval()

    all_results = []
    total_cer = 0.0
    num_samples = 0

    print("开始批量推理...")

    for batch_idx, (fbank, fbank_lengths, labels, label_lengths) in enumerate(tqdm(dataloader, desc="推理中")):
        # 将数据移到GPU
        fbank = fbank.to(device)
        fbank_lengths = fbank_lengths.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        # 前向传播
        log_probs, output_lengths = model(fbank, fbank_lengths)

        # 对每个样本进行解码
        for i in range(fbank.size(0)):
            # 获取单条样本的输出
            sample_log_probs = log_probs[:, i, :]  # [seq_len, vocab_size]

            # 贪心解码
            pred_pinyin = greedy_decode(sample_log_probs, vocab)

            # 获取参考标签
            label_length = label_lengths[i].item()
            ref_indices = labels[i][:label_length].tolist()
            ref_pinyin = [vocab.itos[idx] for idx in ref_indices]

            # 计算CER
            cer = compute_cer(ref_pinyin, pred_pinyin)
            total_cer += cer
            num_samples += 1

            # 保存结果
            all_results.append({
                'ref_pinyin': ref_pinyin,
                'pred_pinyin': pred_pinyin,
                'ref_text': ' '.join(ref_pinyin),
                'pred_text': ' '.join(pred_pinyin),
                'cer': cer
            })

    avg_cer = total_cer / num_samples if num_samples > 0 else 0.0

    return all_results, avg_cer


def save_results(results, avg_cer, output_dir):
    """
    保存推理结果到文件
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保存详细结果
    result_file = os.path.join(output_dir, "inference_results.csv")
    with open(result_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['样本ID', '参考拼音', '预测拼音', 'CER'])
        for i, result in enumerate(results):
            writer.writerow([i+1, result['ref_text'], result['pred_text'], f"{result['cer']:.4f}"])

    # 保存统计信息
    stats_file = os.path.join(output_dir, "stats.txt")
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("语音识别推理结果统计\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"总样本数: {len(results)}\n")
        f.write(f"平均CER: {avg_cer:.4f}\n\n")

        # 统计CER分布
        cer_values = [r['cer'] for r in results]
        f.write("CER分布:\n")
        f.write(f"  最小值: {min(cer_values):.4f}\n")
        f.write(f"  最大值: {max(cer_values):.4f}\n")
        f.write(f"  中位数: {np.median(cer_values):.4f}\n")
        f.write(f"  标准差: {np.std(cer_values):.4f}\n\n")

        # 完全正确的样本数
        perfect_count = sum(1 for r in results if r['cer'] == 0.0)
        f.write(f"完全正确的样本数: {perfect_count} ({perfect_count/len(results)*100:.1f}%)\n")

        # CER <= 0.1的样本数
        good_count = sum(1 for r in results if r['cer'] <= 0.1)
        f.write(f"CER <= 0.1的样本数: {good_count} ({good_count/len(results)*100:.1f}%)\n")

    print(f"\n结果已保存到: {output_dir}")
    print(f"  - 详细结果: {result_file}")
    print(f"  - 统计信息: {stats_file}")


def main():
    """主函数"""
    args = parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ==========================================
    # 1. 构建词表
    # ==========================================
    print("=" * 50)
    print("构建拼音词表...")
    pinyin_list = build_pinyin_list_from_text(args.train_text)
    vocab = Vocab(pinyin_list)
    print(f"词表大小: {vocab.vocab_size}")

    # ==========================================
    # 2. 创建模型并加载权重
    # ==========================================
    print("=" * 50)
    print("加载模型...")

    model = ASRTransformerCTC(
        input_dim=args.input_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        vocab_size=vocab.vocab_size
    )

    # 加载检查点
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        sys.exit(1)

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"模型加载成功: {args.model_path}")
    if 'best_cer' in checkpoint:
        print(f"验证集最佳CER: {checkpoint['best_cer']:.4f}")

    # ==========================================
    # 3. 加载测试数据集
    # ==========================================
    print("=" * 50)
    print("加载测试数据集...")

    test_dataset = ASRDataset(args.test_wav_scp, args.test_text, vocab)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"测试集大小: {len(test_dataset)}")

    # ==========================================
    # 4. 批量推理
    # ==========================================
    print("=" * 50)

    results, avg_cer = inference_batch(model, test_loader, vocab, device)

    # ==========================================
    # 5. 打印结果
    # ==========================================
    print("\n" + "=" * 50)
    print("推理完成！")
    print(f"平均CER: {avg_cer:.4f}")
    print("=" * 50)

    # 打印一些示例
    print("\n示例结果:")
    for i, result in enumerate(results[:5]):
        print(f"\n样本 {i+1}:")
        print(f"  参考: {result['ref_text']}")
        print(f"  预测: {result['pred_text']}")
        print(f"  CER:  {result['cer']:.4f}")

    # ==========================================
    # 6. 保存结果
    # ==========================================
    save_results(results, avg_cer, args.output_dir)


if __name__ == "__main__":
    main()
