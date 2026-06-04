import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. 位置编码 (Sinusoidal Positional Encoding)
# ==========================================
class PositionalEncoding(nn.Module):
    """
    正弦位置编码，为 Transformer 提供时序信息
    输入形状: [seq_len, batch_size, d_model]
    输出形状: [seq_len, batch_size, d_model]
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# ==========================================
# 2. 4倍卷积下采样模块 (算力优化核心)
# ==========================================
class ConvSubsampling(nn.Module):
    """
    卷积下采样模块：将时间维度压缩4倍
    输入形状: [batch_size, seq_len, 80] (Fbank特征)
    输出形状: [batch_size, seq_len//4, d_model]
    """
    def __init__(self, input_dim=80, d_model=512):
        super().__init__()
        # 第一层卷积：通道从1变d_model//2，时间维度压缩2倍
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=d_model//2, 
            kernel_size=3, 
            stride=2, 
            padding=1
        )
        # 第二层卷积：通道从d_model//2变d_model，时间维度再压缩2倍
        self.conv2 = nn.Conv2d(
            in_channels=d_model//2, 
            out_channels=d_model, 
            kernel_size=3, 
            stride=2, 
            padding=1
        )
        # 计算卷积后的特征维度并做线性投影
        # 经过两次 kernel=3, stride=2, padding=1 的卷积后：
        # 时间维度 T -> floor((T+2-3)/2 + 1) = floor(T/2) -> 再一次 -> floor(T/4)
        # 特征维度 F (80) -> 同理 -> floor(F/4) = 20
        self.out = nn.Linear(d_model * (input_dim // 4), d_model)

    def forward(self, x, x_lengths):
        """
        x: [batch_size, seq_len, 80]
        x_lengths: [batch_size] (原始帧数)
        """
        # 1. 增加通道维度：[B, T, F] -> [B, 1, T, F]
        x = x.unsqueeze(1)
        
        # 2. 第一层卷积 + 激活
        x = F.relu(self.conv1(x))
        
        # 3. 第二层卷积 + 激活
        x = F.relu(self.conv2(x))
        
        # 4. 调整形状：[B, C, T', F'] -> [B, T', C*F']
        batch_size, channels, subsampled_time, subsampled_feat = x.size()
        x = x.permute(0, 2, 1, 3).contiguous() # [B, T', C, F']
        x = x.view(batch_size, subsampled_time, channels * subsampled_feat)
        
        # 5. 线性投影到 d_model
        x = self.out(x)
        
        # 6. 计算下采样后的长度 (T -> T//4)
        # 简单的近似计算：每次卷积长度约为 (len + 2*pad - kernel) // stride + 1
        # 这里 kernel=3, stride=2, pad=1，所以公式简化为 (len + 1) // 2
        subsampled_lengths = (x_lengths + 1) // 2
        subsampled_lengths = (subsampled_lengths + 1) // 2
        
        return x, subsampled_lengths

# ==========================================
# 3. 主模型：Transformer Encoder + CTC
# ==========================================
class ASRTransformerCTC(nn.Module):
    def __init__(
        self,
        input_dim=80,        # Fbank特征维度
        d_model=512,         # 模型维度
        nhead=8,             # 注意力头数
        num_encoder_layers=6, # Encoder层数
        dim_feedforward=2048, # 前馈网络维度
        dropout=0.1,
        vocab_size=1000      # 词表大小
    ):
        super().__init__()
        self.d_model = d_model
        
        # A. 卷积下采样 (4倍)
        self.subsampling = ConvSubsampling(input_dim=input_dim, d_model=d_model)
        
        # B. 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # C. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False # 注意：PyTorch默认是 [SeqLen, Batch, Dim]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # D. CTC 输出头
        self.ctc_head = nn.Linear(d_model, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, x_lengths):
        """
        前向传播（全流程）
        :param x: [batch_size, seq_len, 80] (Padding后的Fbank特征)
        :param x_lengths: [batch_size] (原始特征长度)
        :return: 
            log_probs: [subsampled_seq_len, batch_size, vocab_size] (CTC要求的形状)
            output_lengths: [batch_size] (下采样后的长度)
        """
        # 1. 4倍卷积下采样
        # x: [B, T, 80] -> [B, T//4, d_model]
        x, output_lengths = self.subsampling(x, x_lengths)
        
        # 2. 形状转换：[Batch, SeqLen, Dim] -> [SeqLen, Batch, Dim]
        # PyTorch Transformer Encoder 默认要求时序维度在前
        x = x.permute(1, 0, 2)
        
        # 3. 加上位置编码
        x = self.pos_encoder(x * math.sqrt(self.d_model)) # 乘sqrt(d_model)是Transformer原文的技巧
        
        # 4. 生成 Padding Mask (让 Attention 忽略 Padding 部分)
        # mask 形状: [batch_size, subsampled_seq_len]
        # 其中 True/1 表示需要被 mask 掉的位置
        max_len = x.size(0)
        mask = (torch.arange(max_len, device=x.device)[None, :] >= output_lengths[:, None])
        
        # 5. Transformer Encoder 前向传播
        # src_key_padding_mask 就是我们生成的 mask
        encoder_out = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # 6. CTC 输出头
        logits = self.ctc_head(encoder_out)
        log_probs = self.log_softmax(logits)
        
        # 返回形状: [SeqLen, Batch, VocabSize] (这是 PyTorch CTC Loss 要求的输入形状)
        return log_probs, output_lengths
    
if __name__ == "__main__":
    # 模拟参数
    batch_size = 2
    seq_len = 1000  # 模拟10秒音频
    input_dim = 80
    vocab_size = 500 # 假设词表大小500
    
    # 初始化模型
    model = ASRTransformerCTC(
        input_dim=input_dim,
        d_model=256,    # 小模型测试
        nhead=4,
        num_encoder_layers=3,
        vocab_size=vocab_size
    )
    
    # 模拟数据
    dummy_fbank = torch.randn(batch_size, seq_len, input_dim)
    dummy_lengths = torch.tensor([1000, 800], dtype=torch.long) # 第二条音频只有800帧
    
    # 前向传播
    print(f"输入形状: {dummy_fbank.shape}")
    log_probs, output_lengths = model(dummy_fbank, dummy_lengths)
    
    print(f"输出 Log_Prob 形状: {log_probs.shape}")
    print(f"下采样后的长度: {output_lengths}")
    
    # 验证 CTC Loss 计算
    print("\n验证 CTC Loss 兼容性...")
    ctc_loss = nn.CTCLoss(blank=0)
    
    # 模拟一些随机标签
    dummy_targets = torch.randint(1, vocab_size, (batch_size, 10), dtype=torch.long)
    dummy_target_lengths = torch.tensor([10, 8], dtype=torch.long)
    
    # 计算 Loss (应该能正常运行)
    loss = ctc_loss(log_probs, dummy_targets, output_lengths, dummy_target_lengths)
    print(f"CTC Loss 计算成功: {loss.item():.4f}")