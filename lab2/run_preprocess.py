# -*- coding: utf-8 -*-
import os, re, json
from tqdm import tqdm
import numpy as np
from opencc import OpenCC

def _sentence_parse(para):
    result = re.sub(r'（.*?）|{.*?}|《.*?》|[\[\]]', '', para)
    result = ''.join(s for s in result if not s.isdigit() and s != '-')
    result = re.sub('。。', '。', result)
    cc = OpenCC('t2s')
    result = cc.convert(result)
    return result

def _parseRawData(data_path, category, author=None, constrain=None):
    def _handle_json(file):
        rst = []
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for poetry in data:
            if author and poetry.get('author') != author:
                continue
            p = poetry.get('paragraphs', [])
            if not p:
                continue
            flag = False
            for s in p:
                sp = re.split(r'[，！。]', s)
                for tr in sp:
                    if constrain and 0 < len(tr) != constrain:
                        flag = True
                        break
                if flag:
                    break
            if flag:
                continue
            pdata = []
            pdata.extend(poetry.get('paragraphs'))
            pdata = _sentence_parse(''.join(pdata))
            if pdata and len(pdata) > 1:
                rst.append(pdata)
        return rst
    data = []
    file_list = [f for f in os.listdir(data_path) if f.startswith(category)]
    for filename in tqdm(file_list, desc='Parsing JSON Files'):
        data.extend(_handle_json(os.path.join(data_path, filename)))
    return data

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = [len(x) for x in sequences]
    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape mismatch')
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

print('Step 1: Loading data...')
data = _parseRawData('./data', 'poet.song')
print(f'Loaded {len(data)} poems')

print('Step 2: Building vocabulary...')
chars = {c for line in data for c in line}
word2ix = {char: ix for ix, char in enumerate(chars)}
word2ix['<EOP>'] = len(word2ix)
word2ix['<START>'] = len(word2ix)
word2ix['<UNK>'] = len(word2ix)
word2ix['</s>'] = len(word2ix)
ix2word = {ix: char for char, ix in word2ix.items()}
print(f'Vocabulary size: {len(word2ix)}')

print('Step 3: Adding tokens and converting to IDs...')
for i in tqdm(range(len(data)), desc='Processing'):
    data[i] = ['<START>'] + list(data[i]) + ['<EOP>']
data_id = [[word2ix[w] for w in line] for line in tqdm(data, desc='Converting')]

print('Step 4: Padding sequences...')
maxlen = 125
pad_data = pad_sequences(data_id, maxlen=maxlen)
print(f'Padded data shape: {pad_data.shape}')

print('Step 5: Saving...')
os.makedirs('./data', exist_ok=True)
np.savez_compressed('./data/tang.npz', data=pad_data, word2ix=word2ix, ix2word=ix2word)
print('Saved to ./data/tang.npz')
print(f'Data shape: {pad_data.shape}')
print(f'word2ix size: {len(word2ix)}, ix2word size: {len(ix2word)}')
