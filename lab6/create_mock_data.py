#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建模拟数据集用于本地测试
生成随机音频文件，用于验证代码是否正确
"""

import os
import torch
import torchaudio
import random

def create_mock_dataset(num_samples=100):
    """
    创建模拟数据集
    :param num_samples: 样本数量
    """
    # 创建目录
    os.makedirs("dataset/Wave", exist_ok=True)
    os.makedirs("dataset/split/train", exist_ok=True)
    os.makedirs("dataset/split/dev", exist_ok=True)
    os.makedirs("dataset/split/test", exist_ok=True)

    # 模拟拼音词表
    pinyins = [
        "a", "o", "e", "i", "u", "v",
        "ba", "bo", "bi", "bu", "be",
        "pa", "po", "pi", "pu", "pe",
        "ma", "mo", "mi", "mu", "me",
        "fa", "fo", "fi", "fu", "fe",
        "da", "de", "di", "du", "duo",
        "ta", "te", "ti", "tu", "tuo",
        "na", "ne", "ni", "nu", "nuo",
        "la", "le", "li", "lu", "luo",
        "ga", "ge", "gu", "guo",
        "ka", "ke", "ku", "kuo",
        "ha", "he", "hu", "huo",
        "ji", "jia", "jie", "jiu", "jian",
        "qi", "qia", "qie", "qiu", "qian",
        "xi", "xia", "xie", "xiu", "xian",
        "zhi", "zha", "zhe", "zhu", "zhua",
        "chi", "cha", "che", "chu", "chua",
        "shi", "sha", "she", "shu", "shua",
        "ri", "ra", "re", "ru", "rua",
        "zi", "za", "ze", "zu", "zuo",
        "ci", "ca", "ce", "cu", "cuo",
        "si", "sa", "se", "su", "suo",
        "yi", "ya", "ye", "yao", "you",
        "wu", "wa", "wo", "wai", "wei",
        "yu", "yue", "yuan", "yun", "yong",
        "an", "en", "in", "un", "ang",
        "eng", "ing", "ong", "er", "ai",
        "ei", "ao", "ou"
    ]

    print("生成模拟数据集...")

    # 生成训练集、验证集、测试集
    splits = {
        "train": int(num_samples * 0.7),
        "dev": int(num_samples * 0.15),
        "test": num_samples - int(num_samples * 0.7) - int(num_samples * 0.15)
    }

    sample_id = 1

    for split, count in splits.items():
        wav_scp_lines = []
        pinyin_lines = []

        for i in range(count):
            # 生成随机音频（1-3秒，16kHz采样率）
            duration = random.uniform(1.0, 3.0)
            sample_rate = 16000
            num_samples_audio = int(duration * sample_rate)

            # 生成随机波形（模拟语音）
            waveform = torch.randn(1, num_samples_audio) * 0.1

            # 保存音频文件
            wav_filename = f"{sample_id:06d}.wav"
            wav_path = f"dataset/Wave/{wav_filename}"
            torchaudio.save(wav_path, waveform, sample_rate)

            # 生成随机拼音序列（3-8个拼音）
            num_pinyins = random.randint(3, 8)
            selected_pinyins = random.choices(pinyins, k=num_pinyins)
            pinyin_text = " ".join(selected_pinyins)

            # 记录到文件
            wav_scp_lines.append(f"{sample_id:06d}\tWave/{wav_filename}")
            pinyin_lines.append(f"{sample_id:06d}\t{pinyin_text}")

            sample_id += 1

        # 保存wav.scp
        wav_scp_path = f"dataset/split/{split}/wav.scp"
        with open(wav_scp_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(wav_scp_lines) + "\n")

        # 保存pinyin标签
        pinyin_path = f"dataset/split/{split}/pinyin"
        with open(pinyin_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(pinyin_lines) + "\n")

        print(f"  {split}集: {count} 条数据")

    print(f"\n模拟数据集生成完成！")
    print(f"数据保存在: dataset/")


if __name__ == "__main__":
    create_mock_dataset(num_samples=100)
