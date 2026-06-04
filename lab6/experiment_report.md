# Lab6 实验报告：基于AI Coding的语音识别

## 一、实验目的

本实验的主要目的包括：

1. **掌握CTC+Transformer在语音识别中的应用**：理解CTC（Connectionist Temporal Classification）如何解决音频帧与文字标签的不等长对齐问题，以及Transformer Encoder如何建模长距离依赖关系。

2. **理解端到端语音识别系统**：从Fbank特征提取、模型训练到解码推理的完整流程。

3. **掌握语音识别评估指标**：理解CER（Character Error Rate）的计算方法及其物理意义。

4. **实践AI Coding工具**：利用大语言模型辅助完成代码编写、调试和优化，提高开发效率。

---

## 二、实验环境

| 项目 | 配置 |
|------|------|
| 操作系统 | Ubuntu (AutoDL服务器) |
| GPU | NVIDIA RTX 4090 (24GB) |
| Python | 3.8+ |
| PyTorch | 1.10+ |
| 关键库 | torchaudio, jiwer, tensorboard |

---

## 三、实验原理

### 3.1 CTC损失函数

CTC是一种用于序列到序列任务的损失函数，其核心思想是：

1. **引入blank符**：在输出词表中添加一个特殊的blank符（本实验中blank_id=0），用于处理以下情况：
   - 连续相同字符的分隔（如"a-b"中的重复字符）
   - 音频中无语音部分的输出

2. **动态规划求和**：对于一个目标序列，CTC会考虑所有可能的对齐路径（即所有能通过合并重复和移除blank得到目标序列的路径），并计算这些路径的总概率。

3. **损失计算**：负对数似然损失，公式为：
   ```
   Loss = -log(P(目标序列|输入音频))
   ```

### 3.2 Transformer Encoder

Transformer Encoder的核心是多头自注意力机制：

1. **自注意力机制**：通过Query、Key、Value三个矩阵计算序列中每个位置与其他位置的关联度，实现长距离依赖建模。

2. **多头注意力**：将注意力计算分成多个头，每个头学习不同的注意力模式，最后拼接。

3. **前馈网络**：两层全连接网络，提供非线性变换能力。

4. **残差连接和层归一化**：缓解梯度消失问题，加速训练。

### 3.3 模型架构

本实验的模型架构如下：

```
输入: Fbank特征 [batch_size, seq_len, 80]
         ↓
卷积下采样 (2层CNN, 4倍压缩) [batch_size, seq_len//4, 512]
         ↓
位置编码 (正弦/余弦)
         ↓
Transformer Encoder × 6层
         ↓
CTC Head (Linear + LogSoftmax)
         ↓
输出: log_probs [seq_len//4, batch_size, vocab_size]
```

---

## 四、实验步骤

### 4.1 数据准备

1. **数据集说明**：
   - 原始数据集：36000条语音数据
   - 实验数据集：抽取10000条（因显存限制）
   - 划分：训练集/验证集/测试集

2. **特征提取**（[extract.py](dataprocess/extract.py)）：
   - 音频加载：torchaudio.load()
   - 预处理：重采样到16kHz，转为单声道
   - Fbank特征：80维梅尔滤波器组，帧长25ms，帧移10ms
   - 归一化：CMVN（倒谱均值归一化）

3. **词表构建**（[vocab.py](dataprocess/vocab.py)）：
   - 词表大小：414个单元
   - 组成：411个声韵母 + blank + space + unk
   - 映射：拼音序列 ↔ 数字序列

### 4.2 模型训练

1. **训练脚本**：[train.py](train.py)

2. **超参数配置**：
   | 参数 | 值 | 说明 |
   |------|-----|------|
   | batch_size | 16 | 批次大小 |
   | learning_rate | 1e-4 | Adam优化器 |
   | num_epochs | 50 | 训练轮数 |
   | d_model | 512 | 模型维度 |
   | nhead | 8 | 注意力头数 |
   | num_encoder_layers | 6 | Encoder层数 |
   | dim_feedforward | 2048 | 前馈网络维度 |
   | dropout | 0.1 | Dropout比率 |

3. **训练命令**：
   ```bash
   python train.py --batch_size 16 --num_epochs 50 --learning_rate 1e-4
   ```

4. **训练监控**：
   ```bash
   tensorboard --logdir logs
   ```

### 4.3 模型推理

1. **推理脚本**：[inference.py](inference.py)

2. **解码策略**：Greedy Decoding（贪心解码）
   - 取每个时间步概率最大的类别
   - 合并相邻重复字符
   - 移除blank符

3. **推理命令**：
   ```bash
   python inference.py --model_path checkpoints/best_model.pth
   ```

4. **评估指标**：CER（Character Error Rate）
   - 使用编辑距离计算
   - CER = (替换 + 删除 + 插入) / 参考序列长度
   - **CER越低，模型性能越好**

---

## 五、实验结果

### 5.1 训练曲线

（在此插入TensorBoard截图或训练曲线图）

- 训练损失变化趋势
- 验证损失变化趋势
- 学习率变化

### 5.2 验证集结果

| 指标 | 值 |
|------|-----|
| 最佳验证CER | ____ |
| 最佳模型Epoch | ____ |

### 5.3 测试集结果

| 指标 | 值 |
|------|-----|
| 测试集平均CER | ____ |
| 完全正确的样本比例 | ____ |
| CER ≤ 0.1的样本比例 | ____ |

### 5.4 示例结果

| 样本ID | 参考拼音 | 预测拼音 | CER |
|--------|----------|----------|-----|
| 1 | ____ | ____ | ____ |
| 2 | ____ | ____ | ____ |
| 3 | ____ | ____ | ____ |

---

## 六、问题与解决

### 6.1 遇到的问题

1. **问题1**：（描述遇到的问题）
   - **原因分析**：（分析问题原因）
   - **解决方法**：（描述解决过程）

2. **问题2**：（描述遇到的问题）
   - **原因分析**：（分析问题原因）
   - **解决方法**：（描述解决过程）

### 6.2 学习心得

（记录在实验过程中的学习体会、对CTC和Transformer的理解等）

---

## 七、AI Coding工具使用记录

### 7.1 使用的AI工具

- Claude Code / ChatGPT / 其他：____

### 7.2 使用场景

1. **代码编写**：（描述使用AI工具辅助编写了哪些代码）
2. **问题调试**：（描述使用AI工具解决了哪些bug）
3. **代码优化**：（描述使用AI工具进行了哪些优化）

### 7.3 AI工具的优势与局限

- **优势**：____
- **局限**：____

---

## 八、思考题

1. **CTC的blank符有什么作用？如果没有blank符会有什么问题？**

2. **为什么使用Transformer Encoder而不是RNN/LSTM？各有什么优缺点？**

3. **卷积下采样的作用是什么？为什么选择4倍下采样？**

4. **Greedy Decoding有什么局限性？如何改进？**

5. **CER和WER有什么区别？在什么场景下使用哪个更合适？**

---

## 九、代码文件说明

| 文件 | 功能 |
|------|------|
| [train.py](train.py) | 模型训练脚本 |
| [inference.py](inference.py) | 模型推理脚本 |
| [dataprocess/extract.py](dataprocess/extract.py) | Fbank特征提取 |
| [dataprocess/vocab.py](dataprocess/vocab.py) | 词表管理 |
| [dataprocess/dataset.py](dataprocess/dataset.py) | 数据集加载 |
| [model/__init__.py](model/__init__.py) | 模型定义 |

---

## 十、参考资料

1. Graves, A., et al. (2006). Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks.
2. Vaswani, A., et al. (2017). Attention Is All You Need.
3. 实验PDF文档：[lab6 - 基于AI Coding的语音识别.pdf](lab6%20-%20基于AI%20Coding的语音识别.pdf)
