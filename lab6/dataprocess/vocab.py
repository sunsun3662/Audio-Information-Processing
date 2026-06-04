def build_pinyin_list_from_text(text_file_path):
    """
    从text文件中提取所有出现过的拼音，生成pinyin_list
    :param text_file_path: text文件路径
    :return: list[str], 所有拼音的列表
    """
    all_pinyins = set()
    with open(text_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 按你的格式：text_id<\t>text
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            text = parts[1]
            # 切分拼音并加入集合
            pinyins = text.split()
            all_pinyins.update(pinyins)
    return list(all_pinyins)

class Vocab:
    """
    拼音词表类：实现 拼音序列 <-> 数字序列 的互相转换
    专为CTC设计，blank_id固定为0
    """
    def __init__(self, pinyin_list):
        """
        初始化词表
        :param pinyin_list: 所有拼音的列表（从训练集text文件中统计得到）
        """
        # --------------------------
        # 1. 定义特殊符号（CTC核心）
        # --------------------------
        self.blank = "<blank>"  # CTC空白符，必须有
        self.special_tokens = [self.blank]
        
        # --------------------------
        # 2. 构建映射关系
        # --------------------------
        # 索引 -> 拼音 (itos: Index TO String)
        # 顺序：[blank, pinyin1, pinyin2, ...]
        self.itos = self.special_tokens + sorted(list(set(pinyin_list)))
        
        # 拼音 -> 索引 (stoi: String TO Index)
        self.stoi = {pinyin: idx for idx, pinyin in enumerate(self.itos)}
        
        # --------------------------
        # 3. 基本属性
        # --------------------------
        self.vocab_size = len(self.itos)
        self.blank_id = self.stoi[self.blank]  # CTC损失需要单独传入blank_id

    def text2idx(self, text):
        """
        文本转索引序列
        :param text: str, 空格分隔的拼音字符串，例如 "yi ge nan ren"
        :return: list[int], 索引序列，例如 [12, 5, 33, 45]
        """
        pinyin_seq = text.strip().split()  # 按空格切分成拼音列表
        idx_seq = []
        for pinyin in pinyin_seq:
            # 安全起见，如果拼音不在词表中（理论上训练集不会出现），暂时跳过或报错
            if pinyin in self.stoi:
                idx_seq.append(self.stoi[pinyin])
            else:
                # 这里可以根据需求改成 <unk>，但拼音识别通常覆盖全量拼音
                raise ValueError(f"拼音 [{pinyin}] 不在词表中！")
        return idx_seq

    def idx2text(self, idx_seq, remove_blank=True):
        """
        索引序列转文本（严格遵循CTC解码逻辑）
        :param idx_seq: list[int], 索引序列，例如 [0, 12, 12, 0, 5, 0, 33, 45]
        :param remove_blank: bool, 是否去除blank符（推理时True，调试时可False）
        :return: str, 空格分隔的拼音字符串
        """
        if not idx_seq:
            return ""
            
        # --------------------------
        # 1. 第一步：合并相邻重复项 (CTC核心)
        # --------------------------
        merged_seq = []
        prev_idx = None
        for idx in idx_seq:
            if idx != prev_idx:
                merged_seq.append(idx)
                prev_idx = idx
        
        # --------------------------
        # 2. 第二步：移除 blank (如果需要)
        # --------------------------
        if remove_blank:
            merged_seq = [idx for idx in merged_seq if idx != self.blank_id]
            
        # --------------------------
        # 3. 转成拼音字符串
        # --------------------------
        pinyin_seq = [self.itos[idx] for idx in merged_seq]
        return " ".join(pinyin_seq)
    

# ==========================================
# 测试逻辑（模拟你的数据）
# ==========================================
if __name__ == "__main__":
    # 1. 模拟从你的train.text文件提取拼音列表
    # 实际使用时改成：build_pinyin_list_from_text("你的train.text路径")
    dummy_pinyin_list = build_pinyin_list_from_text("/mnt/nvme0n1/guoyujie_space/work/speechclass/dataset/split/train/pinyin")
    
    # 2. 初始化词表
    vocab = Vocab(dummy_pinyin_list)
    
    # 3. 测试 text2idx
    sample_text = "yi ge nan ren"
    idx_seq = vocab.text2idx(sample_text)
    print(f"原文: {sample_text}")
    print(f"转索引: {idx_seq}")
    print(f"词表大小: {vocab.vocab_size}, blank_id: {vocab.blank_id}")
    print("-" * 30)
    
    # 4. 测试 idx2text (模拟CTC输出的带blank序列)
    # 模拟CTC输出：[blank, yi, yi, blank, ge, blank, nan, ren, blank]
    ctc_like_idx_seq = [vocab.blank_id, idx_seq[0], idx_seq[0], vocab.blank_id, idx_seq[1], vocab.blank_id, idx_seq[2], idx_seq[3], vocab.blank_id]
    decoded_text = vocab.idx2text(ctc_like_idx_seq, remove_blank=True)
    print(f"CTC模拟索引: {ctc_like_idx_seq}")
    print(f"解码后: {decoded_text}")