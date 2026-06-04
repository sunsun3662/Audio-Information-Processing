#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lab6 语音识别训练脚本
模型架构：Transformer Encoder + CTC
功能：训练语音识别模型，将Fbank特征转换为拼音序列
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataprocess.vocab import Vocab, build_pinyin_list_from_text
from dataprocess.dataset import ASRDataset, collate_fn
from model import ASRTransformerCTC


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练语音识别模型")

    # 数据相关
    parser.add_argument("--train_wav_scp", type=str, default="dataset/split/train/wav.scp",
                        help="训练集wav.scp路径")
    parser.add_argument("--train_text", type=str, default="dataset/split/train/pinyin",
                        help="训练集拼音标签路径")
    parser.add_argument("--dev_wav_scp", type=str, default="dataset/split/dev/wav.scp",
                        help="验证集wav.scp路径")
    parser.add_argument("--dev_text", type=str, default="dataset/split/dev/pinyin",
                        help="验证集拼音标签路径")

    # 模型相关
    parser.add_argument("--input_dim", type=int, default=80, help="Fbank特征维度")
    parser.add_argument("--d_model", type=int, default=512, help="Transformer模型维度")
    parser.add_argument("--nhead", type=int, default=8, help="注意力头数")
    parser.add_argument("--num_encoder_layers", type=int, default=6, help="Encoder层数")
    parser.add_argument("--dim_feedforward", type=int, default=2048, help="前馈网络维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout比率")

    # 训练相关
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--grad_clip", type=float, default=5.0, help="梯度裁剪阈值")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")

    # 其他
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="模型保存目录")
    parser.add_argument("--log_dir", type=str, default="logs", help="TensorBoard日志目录")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    return parser.parse_args()


def set_seed(seed):
    """设置随机种子，确保实验可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_cer(reference, hypothesis):
    """
    计算字符错误率 (Character Error Rate)
    使用编辑距离算法
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


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer):
    """
    训练一个epoch
    :return: 平均损失
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (fbank, fbank_lengths, labels, label_lengths) in enumerate(dataloader):
        # 将数据移到GPU
        fbank = fbank.to(device)
        fbank_lengths = fbank_lengths.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        # 前向传播
        log_probs, output_lengths = model(fbank, fbank_lengths)

        # 计算CTC损失
        # CTC要求输入形状: [seq_len, batch_size, vocab_size]
        # 标签形状: [batch_size, max_label_len]
        loss = criterion(log_probs, labels, output_lengths, label_lengths)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # 打印训练进度
        if (batch_idx + 1) % 50 == 0:
            avg_loss = total_loss / num_batches
            print(f"  Epoch [{epoch}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

            # 记录到TensorBoard
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, dataloader, criterion, vocab, device):
    """
    在验证集上评估模型
    :return: 平均损失, CER
    """
    model.eval()
    total_loss = 0.0
    total_cer = 0.0
    num_samples = 0

    for fbank, fbank_lengths, labels, label_lengths in dataloader:
        # 将数据移到GPU
        fbank = fbank.to(device)
        fbank_lengths = fbank_lengths.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        # 前向传播
        log_probs, output_lengths = model(fbank, fbank_lengths)

        # 计算损失
        loss = criterion(log_probs, labels, output_lengths, label_lengths)
        total_loss += loss.item() * fbank.size(0)

        # 计算CER
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

    avg_loss = total_loss / num_samples
    avg_cer = total_cer / num_samples

    return avg_loss, avg_cer


def main():
    """主函数"""
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # ==========================================
    # 1. 构建词表
    # ==========================================
    print("=" * 50)
    print("构建拼音词表...")
    pinyin_list = build_pinyin_list_from_text(args.train_text)
    vocab = Vocab(pinyin_list)
    print(f"词表大小: {vocab.vocab_size}")
    print(f"Blank ID: {vocab.blank_id}")

    # ==========================================
    # 2. 创建数据集和数据加载器
    # ==========================================
    print("=" * 50)
    print("加载数据集...")

    train_dataset = ASRDataset(args.train_wav_scp, args.train_text, vocab)
    dev_dataset = ASRDataset(args.dev_wav_scp, args.dev_text, vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(dev_dataset)}")

    # ==========================================
    # 3. 创建模型
    # ==========================================
    print("=" * 50)
    print("创建模型...")

    model = ASRTransformerCTC(
        input_dim=args.input_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        vocab_size=vocab.vocab_size
    )

    model = model.to(device)

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # ==========================================
    # 4. 定义损失函数和优化器
    # ==========================================
    criterion = nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # ==========================================
    # 5. 恢复训练（如果需要）
    # ==========================================
    start_epoch = 0
    best_cer = float('inf')

    if args.resume:
        print(f"恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_cer = checkpoint.get('best_cer', float('inf'))
        print(f"从Epoch {start_epoch}继续训练")

    # ==========================================
    # 6. 初始化TensorBoard
    # ==========================================
    writer = SummaryWriter(log_dir=args.log_dir)

    # ==========================================
    # 7. 开始训练
    # ==========================================
    print("=" * 50)
    print("开始训练...")
    print(f"训练轮数: {args.num_epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print("=" * 50)

    for epoch in range(start_epoch, args.num_epochs):
        epoch_start_time = time.time()

        # 训练
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)

        # 验证
        val_loss, val_cer = evaluate(model, dev_loader, criterion, vocab, device)

        # 更新学习率
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        epoch_time = time.time() - epoch_start_time

        # 打印Epoch结果
        print(f"\nEpoch [{epoch+1}/{args.num_epochs}] - {epoch_time:.1f}s")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val CER: {val_cer:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")

        # 记录到TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('CER/Validation', val_cer, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        # 保存最佳模型
        if val_cer < best_cer:
            best_cer = val_cer
            best_model_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_cer': best_cer,
                'val_loss': val_loss,
            }, best_model_path)
            print(f"  [OK] 保存最佳模型 (CER: {best_cer:.4f})")

        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_cer': best_cer,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  [OK] 保存检查点: {checkpoint_path}")

    # 训练完成
    print("=" * 50)
    print("训练完成！")
    print(f"最佳验证CER: {best_cer:.4f}")
    print(f"最佳模型保存在: {os.path.join(args.save_dir, 'best_model.pth')}")

    # 关闭TensorBoard
    writer.close()


if __name__ == "__main__":
    main()
