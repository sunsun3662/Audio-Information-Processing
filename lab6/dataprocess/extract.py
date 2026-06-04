import torch
import torchaudio
import numpy as np

def _load_audio_with_scipy(audio_path):
    """
    使用scipy加载音频文件（Windows兼容方案）
    :param audio_path: str, 音频文件路径
    :return: tuple (waveform_tensor, sample_rate)
    """
    from scipy.io import wavfile
    sample_rate, data = wavfile.read(audio_path)

    # 转为float32并归一化
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float64:
        data = data.astype(np.float32)

    # 转为tensor [num_channels, num_samples]
    if data.ndim == 1:
        waveform = torch.from_numpy(data).unsqueeze(0)
    else:
        waveform = torch.from_numpy(data.T)

    return waveform, sample_rate


def extract_fbank(audio_path, target_sample_rate=16000, num_mel_bins=80):
    """
    提取80维Fbank特征（标准语音识别流程）
    :param audio_path: str, 音频文件路径（.wav格式）
    :param target_sample_rate: int, 目标采样率，默认16000Hz（语音识别标准）
    :param num_mel_bins: int, Fbank特征维度，默认80
    :return: torch.Tensor, Fbank特征，形状为 [num_frames, 80]
    """
    # --------------------------
    # 1. 加载音频文件
    # --------------------------
    # waveform: [num_channels, num_samples], sample_rate: 原始采样率
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
    except RuntimeError:
        # Windows下torchaudio可能缺少后端，使用scipy作为备选
        waveform, sample_rate = _load_audio_with_scipy(audio_path)
    
    # --------------------------
    # 2. 预处理：统一采样率 & 单声道
    # --------------------------
    # 重采样到目标采样率（16000Hz）
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    
    # 转为单声道（如果是多声道，取平均）
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # --------------------------
    # 3. 提取Fbank特征（Kaldi标准参数）
    # --------------------------
    # 使用torchaudio的Kaldi兼容接口，参数完全对齐Kaldi
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        sample_frequency=target_sample_rate,
        frame_length=25.0,       # 帧长25ms（标准）
        frame_shift=10.0,         # 帧移10ms（标准）
        num_mel_bins=num_mel_bins, # 80维
        window_type="hamming",    # 汉明窗
        preemphasis_coefficient=0.97, # 预加重系数
        use_energy=False          # 不单独输出能量（将能量替换为第0维Fbank）
    )
    
    # --------------------------
    # 4. 特征归一化（CMN：倒谱均值归一化，简单有效）
    # --------------------------
    # 对每个特征维度减去均值（消除通道/环境偏移）
    mean = torch.mean(fbank, dim=0, keepdim=True)
    fbank_normalized = fbank - mean
    
    return fbank_normalized


if __name__ == "__main__":
    # 替换为你的一个真实wav文件路径测试
    test_audio_path = "/mnt/nvme0n1/guoyujie_space/work/speechclass/dataset/Wave/008001.wav" 
    
    try:
        fbank_feat = extract_fbank(test_audio_path)
        print(f"Fbank提取成功！")
        print(f"特征形状: {fbank_feat.shape}  -> [帧数, 80维]")
        print(f"特征数据类型: {fbank_feat.dtype}")
        print(f"特征均值（归一化后接近0）: {torch.mean(fbank_feat):.4f}")
    except Exception as e:
        print(f"提取失败，请检查音频路径: {e}")