from dataprocess.vocab import *
from dataprocess.dataset import *

from torch.utils.data import Dataset, DataLoader

def build_asr_dataloaders(
    # 训练集路径（必须）
    train_wav_scp="dataset/split/train/wav.scp", 
    train_text="dataset/split/train/pinyin",
    # 验证集路径（必须）
    dev_wav_scp="dataset/split/dev/wav.scp", 
    dev_text="dataset/split/dev/pinyin",
    # 测试集路径（可选，若为None则不加载测试集）
    test_wav_scp="dataset/split/test/wav.scp", 
    test_text="dataset/split/test/pinyin",
    # 词表（可选，若为None则自动从训练集构建）
    vocab=None,
    # DataLoader 超参数
    batch_size=16,
    num_workers=0,
    shuffle_train=True
):
    """
    一键构建语音识别全套数据加载器
    :return: tuple (datasets, dataloaders, vocab)
             datasets: dict, 包含 'train', 'dev', 'test' (可选) 的 Dataset
             dataloaders: dict, 包含 'train', 'dev', 'test' (可选) 的 DataLoader
             vocab: Vocab 类实例
    """
    # ==========================================
    # 1. 构建/获取词表（核心：仅用训练集）
    # ==========================================
    if vocab is None:
        print("正在从训练集构建词表...")
        train_pinyin_list = build_pinyin_list_from_text(train_text)
        vocab = Vocab(train_pinyin_list)
        print(f"词表构建成功 | 大小: {vocab.vocab_size}, Blank ID: {vocab.blank_id}")
    else:
        print("使用传入的已有词表")

    # ==========================================
    # 2. 加载数据集
    # ==========================================
    datasets = {}
    
    print("\n正在加载训练集...")
    datasets['train'] = ASRDataset(train_wav_scp, train_text, vocab)
    
    print("正在加载验证集...")
    datasets['dev'] = ASRDataset(dev_wav_scp, dev_text, vocab)
    
    if test_wav_scp is not None and test_text is not None:
        print("正在加载测试集...")
        datasets['test'] = ASRDataset(test_wav_scp, test_text, vocab)

    # ==========================================
    # 3. 构建 DataLoader
    # ==========================================
    dataloaders = {}
    
    # 训练集（支持打乱）
    dataloaders['train'] = DataLoader(
        datasets['train'],
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    # 验证集（不打乱）
    dataloaders['dev'] = DataLoader(
        datasets['dev'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    # 测试集（如果有）
    if 'test' in datasets:
        dataloaders['test'] = DataLoader(
            datasets['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )

    print("\n✅ 数据加载全部完成！")
    return datasets, dataloaders, vocab

if __name__ == "__main__":
    datasets, dataloaders, vocab = build_asr_dataloaders(batch_size=8)