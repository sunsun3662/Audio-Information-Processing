class Tokenizer:
    def __init__(self, vocab_file = "./tokenizer/vocab.txt"):
        self.token2id = {}
        self.id2token = {}
        count = 0

        self.special_token = ["<pad>", "<unk>", "<sos>", "<eos>", " ", "<blk>"]
        for token in self.special_token:
            self.token2id[token] = count
            self.id2token[count] = token
            count += 1

        with open(vocab_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                self.token2id[line] = count
                self.id2token[count] = line
                count += 1

    def __call__(self, s:list):
        """
        TODO
        输入token list
        返回 token id list
        """
        has_unk = False
        unk = []
        res = []
        for token in s:
            if token in self.token2id:
                res.append(self.token2id[token])
            else:
                res.append(self.token2id["<unk>"])
                has_unk = True
                unk.append(token)
        if has_unk:
            print("unk: ", unk)
        return res

    def decode(self, ids: list, ignore_special=True):
        """
        TODO
        输入为 token id list
        返回为token list
        ignore_special: 是否忽略特殊字符
        """
        res = []
        for id in ids:
            token =  self.id2token[id]
            if token not in self.special_token:
                res.append(token)
            elif not ignore_special:
                res.append(token)                
        return res
    
    def special_token_ids(self):
        return [self.token2id[token] for token in self.special_token]
    
    def size(self):
        return len(self.token2id)
    
    def padding_id(self):
        return self.token2id["<pad>"]
    
    def blk_id(self):
        return self.token2id["<blk>"]
