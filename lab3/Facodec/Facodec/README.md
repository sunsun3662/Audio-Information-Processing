# Lab3: FACodec 语音离散化、特征解耦、重构与音色转换

## 环境配置

```bash
conda create -n facodec python=3.10 -y
conda activate facodec
pip3 install torch==2.1.2 torchaudio==2.1.2
pip install librosa==0.10.1 pyworld einops soundfile "numpy<2" matplotlib "setuptools<70"
```

## 文件结构

```
Facodec/
├── test.py                    # 主实验脚本
├── ckpt/
│   ├── ns3_facodec_encoder.bin   # 编码器权重
│   └── ns3_facodec_decoder.bin   # 解码器权重
├── samples/
│   ├── speaker1.wav              # 说话人1 原始音频
│   ├── speaker2.wav              # 说话人2 原始音频
│   └── user.wav                  # 用户录音（用于音色转换）
└── ns3_codec/                     # FACodec 模型实现代码
```

## 运行方式

```bash
conda activate facodec
cd lab3/Facodec/Facodec
python test.py
```

## test.py 执行内容

| 步骤 | 说明 | 输出 |
|------|------|------|
| Step 1 | 实例化 FACodecEncoder/FACodecDecoder，加载预训练权重 | - |
| Step 2 | 加载 speaker2.wav（16kHz） | Test Audio Shape: [1, 1, 152418] |
| Step 3 | 编码 → VQ 离散化 → 特征解耦 → 重构 | 各 tensor shape 打印 |
| Step 4 | 保存重构音频 + 波形图/频谱图 | speaker2_rec.wav, *.png |
| Step 5 | 音色转换（speaker1 + user.wav） | speaker1_rec_with_user.wav |

## 手动音色转换命令

如果想用 speaker2 的内容 + user 的音色生成新音频，运行：

```bash
conda activate facodec
cd lab3/Facodec/Facodec
python -c "
import os, torch, librosa, numpy as np, soundfile as sf
from ns3_codec import FACodecEncoder, FACodecDecoder

fa_encoder = FACodecEncoder(ngf=32, up_ratios=[2,4,5,5], out_channels=256)
fa_decoder = FACodecDecoder(in_channels=256, upsample_initial_channel=1024, ngf=32,
    up_ratios=[5,5,4,2], vq_num_q_c=2, vq_num_q_p=1, vq_num_q_r=3,
    vq_dim=256, codebook_dim=8, codebook_size_prosody=10,
    codebook_size_content=10, codebook_size_residual=10,
    use_gr_x_timbre=True, use_gr_residual_f0=True, use_gr_residual_phone=True)

fa_encoder.load_state_dict(torch.load('ckpt/ns3_facodec_encoder.bin'))
fa_decoder.load_state_dict(torch.load('ckpt/ns3_facodec_decoder.bin'))
fa_encoder.eval(); fa_decoder.eval()

sr = 16000
spk2_wav = torch.from_numpy(librosa.load('samples/speaker2.wav', sr=sr)[0]).float().unsqueeze(0).unsqueeze(0)
user_wav = torch.from_numpy(librosa.load('samples/user.wav', sr=sr)[0]).float().unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    enc_spk2 = fa_encoder(spk2_wav)
    enc_user = fa_encoder(user_wav)
    _, _, _, _, spk_embs_user = fa_decoder(enc_user, eval_vq=False, vq=True)
    _, _, _, q = fa_decoder.quantize(enc_spk2)
    all_embs = q[0] + q[1] + q[2]
    rec = fa_decoder.inference(all_embs, spk_embs_user)

sf.write('samples/speaker2_rec_with_user.wav', rec[0][0].cpu().numpy(), sr)
print('Done: speaker2_rec_with_user.wav')
"
```

## 生成文件说明

运行 `test.py` 后 `samples/` 目录下会生成：

| 文件 | 说明 |
|------|------|
| speaker2_rec.wav | speaker2 重构音频（原始音色） |
| speaker2_waveform.png | speaker2 原始波形图 |
| speaker2_rec_waveform.png | speaker2 重构波形图 |
| speaker2_spec.png | speaker2 原始频谱图 |
| speaker2_rec_spec.png | speaker2 重构频谱图 |
| speaker1_rec_with_user.wav | 音色转换：speaker1 内容 + user 音色 |
| speaker1_waveform.png | speaker1 原始波形图 |
| speaker1_to_user_waveform.png | 音色转换后波形图 |
| speaker1_spec.png | speaker1 原始频谱图 |
| speaker1_to_user_spec.png | 音色转换后频谱图 |
| speaker2_rec_with_user.wav | 音色转换：speaker2 内容 + user 音色 |

## Tensor 维度说明

| Tensor | Shape | 含义 |
|--------|-------|------|
| Test Audio | [1, 1, 152418] | [batch, channel, samples] |
| Encoder_out | [1, 256, 762] | [batch, dim, time_steps] |
| Prosody Code | [1, 1, 762] | [1个量化器, batch, time_steps] |
| Content Code | [2, 1, 762] | [2个量化器, batch, time_steps] |
| Residual Code | [3, 1, 762] | [3个量化器, batch, time_steps] |
| Prosody Embedding | [1, 256, 762] | [batch, dim, time_steps] |
| Content Embedding | [1, 256, 762] | [batch, dim, time_steps] |
| Detail Embedding | [1, 256, 762] | [batch, dim, time_steps] |
| Speaker Embedding | [1, 256] | [batch, dim] 全局音色特征 |
| Reconstruct Audio | [1, 1, 152400] | [batch, channel, samples] |
