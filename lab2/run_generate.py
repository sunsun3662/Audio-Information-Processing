# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn

# ===== PoetryModel（与训练时一致，使用内置nn.LSTM）=====
class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, layer_num):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=layer_num)
        self.linear1 = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input, hidden=None):
        seq_len, batch_size = input.size()
        embeds = self.embeddings(input)
        output, hidden = self.lstm(embeds, hidden)
        output = self.linear1(output.view(seq_len * batch_size, -1))
        return output, hidden


def generate(model, start_words, ix2word, word2ix, max_gen_len, prefix_words=None):
    results = list(start_words)
    start_words_len = len(start_words)
    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long().to(device)
    hidden = None

    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = input.data.new([word2ix[word]]).view(1, 1).to(device)

    for i in range(max_gen_len):
        output, hidden = model(input, hidden)
        if i < start_words_len:
            w = start_words[i]
            input = input.data.new([word2ix[w]]).view(1, 1)
        else:
            top_index = output.data[0].topk(1)[1][0]
            w = ix2word[top_index.item()]
            results.append(w)
            input = input.data.new([top_index]).view(1, 1)
        if w == '<EOP>':
            break
    return results


def gen_acrostic(model, start_words, ix2word, word2ix, max_gen_len, prefix_words=None):
    results = []
    start_words_len = len(start_words)
    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long().to(device)
    hidden = None
    index = 0
    pre_word = '<START>'

    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = input.data.new([word2ix[word]]).view(1, 1).to(device)

    for i in range(max_gen_len):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0]
        w = ix2word[top_index.item()]

        if (pre_word in {'。', '! ', '<START>'}):
            if index == start_words_len:
                break
            else:
                w = start_words[index]
                index += 1
                input = torch.Tensor([word2ix[w]]).view(1, 1).long().to(device)
        else:
            input = torch.Tensor([top_index]).view(1, 1).long().to(device)

        results.append(w)
        pre_word = w
    return results


if __name__ == '__main__':
    # 加载数据
    pickle_path = "./data/tang.npz"
    datas = np.load(pickle_path, allow_pickle=True)
    word2ix = datas['word2ix'].item()
    ix2word = datas['ix2word'].item()
    vocab_size = len(word2ix)

    # 模型参数
    embedding_dim = 256
    hidden_dim = 256
    layer_num = 2
    model_path = './checkpoints/tang_model.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = PoetryModel(vocab_size, embedding_dim, hidden_dim, layer_num)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    prefix_words = '江流天地外，山色有无中。'
    max_gen_len = 200

    # 藏头诗
    print("=" * 50)
    print("藏头诗（深度学习）")
    print("=" * 50)
    start_words = '深度学习'
    result = gen_acrostic(model, start_words, ix2word, word2ix, max_gen_len, prefix_words)
    print(''.join(result))

    print()

    # 续写诗
    print("=" * 50)
    print("续写诗（大漠孤烟照高阁）")
    print("=" * 50)
    start_words = '大漠孤烟照高阁'
    result = generate(model, start_words, ix2word, word2ix, max_gen_len, prefix_words)
    print(''.join(result))
