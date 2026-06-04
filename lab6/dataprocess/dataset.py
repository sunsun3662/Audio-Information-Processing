import torch
from torch.utils.data import Dataset
from dataprocess.extract import extract_fbank
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    对变长序列进行 Padding 对齐
    :param batch: list, 由 Dataset.__getitem__ 返回的 tuple 列表
                  [(fbank_1, label_1), (fbank_2, label_2), ...]
    :return: tuple (padded_features, feature_lengths, padded_labels, label_lengths)
             padded_features: [batch_size, max_frames, 80]
             feature_lengths: [batch_size] (原始帧数，用于CTC损失)
             padded_labels: [batch_size, max_label_len]
             label_lengths: [batch_size] (原始标签长度，用于CTC损失)
    """
    # --------------------------
    # 1. 分离特征和标签
    # --------------------------
    # features: list of [num_frames, 80]
    # labels: list of [num_labels]
    features, labels = zip(*batch)
    
    # --------------------------
    # 2. 计算原始长度（CTC损失必须）
    # --------------------------
    feature_lengths = torch.tensor([len(feat) for feat in features], dtype=torch.long)
    label_lengths = torch.tensor([len(lbl) for lbl in labels], dtype=torch.long)
    
    # --------------------------
    # 3. 对特征进行 Padding
    # --------------------------
    # pad_sequence 默认在第0维填充，我们需要 batch_first，所以先堆叠再处理
    # 输出形状: [batch_size, max_frames, 80]
    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)
    
    # --------------------------
    # 4. 对标签进行 Padding
    # --------------------------
    # 标签用 0 填充（注意：虽然 blank_id 也是 0，但 CTC 损失会通过 label_lengths 忽略 padding 部分）
    # 输出形状: [batch_size, max_label_len]
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)
    
    return padded_features, feature_lengths, padded_labels, label_lengths

class ASRDataset(Dataset):
    """
    自定义语音识别数据集
    输入：wav.scp 和 text 文件路径
    输出：(fbank特征, 标签索引序列)
    """
    def __init__(self, wav_scp_path, text_path, vocab):
        """
        初始化数据集
        :param wav_scp_path: str, wav.scp文件路径
        :param text_path: str, text文件路径
        :param vocab: Vocab类实例，用于文本转索引
        """
        self.vocab = vocab
        
        # --------------------------
        # 1. 读取 wav.scp：构建 {wav_id: wav_path} 映射
        # --------------------------
        self.wav_id2path = {}
        with open(wav_scp_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) == 2:
                    wav_id, wav_path = parts
                    self.wav_id2path[wav_id] = "dataset/" + wav_path
        
        # --------------------------
        # 2. 读取 text：构建 {wav_id: text} 映射
        # --------------------------
        self.wav_id2text = {}
        with open(text_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) == 2:
                    wav_id, text = parts
                    self.wav_id2text[wav_id] = text
        
        # --------------------------
        # 3. 取交集：只保留同时存在于 wav.scp 和 text 中的 wav_id
        # --------------------------
        self.valid_wav_ids = sorted(list(
            set(self.wav_id2path.keys()) & set(self.wav_id2text.keys())
        ))
        
        if len(self.valid_wav_ids) == 0:
            raise ValueError("wav.scp 和 text 中没有匹配的ID！请检查文件格式。")
        
        print(f"数据集加载成功：共 {len(self.valid_wav_ids)} 条有效数据")

    def __len__(self):
        """返回数据集大小"""
        return len(self.valid_wav_ids)

    def __getitem__(self, idx):
        """
        获取单条数据
        :param idx: int, 数据索引
        :return: tuple (fbank_tensor, label_tensor)
                 fbank_tensor: [num_frames, 80]
                 label_tensor: [num_labels]
        """
        # 1. 获取当前 wav_id
        wav_id = self.valid_wav_ids[idx]
        
        # 2. 获取音频路径和文本
        wav_path = self.wav_id2path[wav_id]
        text = self.wav_id2text[wav_id]
        
        # 3. 提取Fbank特征（调用之前写好的函数）
        try:
            fbank_tensor = extract_fbank(wav_path)
        except Exception as e:
            raise RuntimeError(f"提取特征失败: {wav_id}, {wav_path}, 错误: {e}")
        
        # 4. 文本转索引（调用vocab）
        try:
            label_idx = self.vocab.text2idx(text)
        except Exception as e:
            raise RuntimeError(f"文本转索引失败: {wav_id}, {text}, 错误: {e}")
        
        # 5. 转成PyTorch张量
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        
        return fbank_tensor, label_tensor
    
if __name__ == "__main__":
    from dataprocess.vocab import *
    # ==========================================
    # 前置准备：初始化 Vocab
    # ==========================================
    # 实际使用时，请先通过 build_pinyin_list_from_text 从 train.text 生成 pinyin_list
    dummy_pinyin_list = build_pinyin_list_from_text("/mnt/nvme0n1/guoyujie_space/work/speechclass/dataset/split/train/pinyin")
    vocab = Vocab(dummy_pinyin_list)
    
    # ==========================================
    # 测试 Dataset
    # ==========================================
    # 替换为你真实的文件路径
    test_wav_scp = "/mnt/nvme0n1/guoyujie_space/work/speechclass/dataset/split/train/wav.scp"
    test_text = "/mnt/nvme0n1/guoyujie_space/work/speechclass/dataset/split/train/pinyin"
    
    try:
        dataset = ASRDataset(test_wav_scp, test_text, vocab)
        
        # 测试获取第一条数据
        fbank, label = dataset[0]
        
        print(f"\n单条数据测试成功：")
        print(f"Fbank特征形状: {fbank.shape}")
        print(f"标签索引: {label}")
        print(f"标签对应拼音: {vocab.idx2text(label.tolist(), remove_blank=False)}")
        
    except Exception as e:
        print(f"数据集测试失败: {e}")


    # ==========================================
    # 模拟一个 Batch 的数据（2条样本）
    # ==========================================
    # 样本1: 100帧, 标签长度 5
    feat1 = torch.randn(100, 80)
    label1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    
    # 样本2: 150帧, 标签长度 7
    feat2 = torch.randn(150, 80)
    label2 = torch.tensor([6, 7, 8, 9, 10, 11, 12], dtype=torch.long)
    
    # 组成 batch
    dummy_batch = [(feat1, label1), (feat2, label2)]
    
    # ==========================================
    # 测试 collate_fn
    # ==========================================
    padded_feats, feat_lens, padded_labels, label_lens = collate_fn(dummy_batch)
    
    print("=" * 50)
    print(f"Padding后的特征形状: {padded_feats.shape}")
    print(f"原始帧数: {feat_lens}")
    print(f"Padding后的标签形状: {padded_labels.shape}")
    print(f"原始标签长度: {label_lens}")
    print("=" * 50)
    
    # 预期输出：
    # Padding后的特征形状: torch.Size([2, 150, 80])
    # 原始帧数: tensor([100, 150])
    # Padding后的标签形状: torch.Size([2, 7])
    # 原始标签长度: tensor([5, 7])