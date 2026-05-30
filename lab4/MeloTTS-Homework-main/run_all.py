"""
严格按 notebook 要求执行所有代码单元，生成输出结果。
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

import sys
sys.path.insert(0, '.')

from melo.api import TTS
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

# ==================== 课程一 ====================
print("=" * 60)
print("课程一：MeloTTS 的安装 - 基本推理")
print("=" * 60)

speed = 1.0
device = 'cpu'

text = "我最近在学习machine learning，希望能够在未来的artificial intelligence领域有所建树。"
model = TTS(language='ZH', device=device)
speaker_ids = model.hps.data.spk2id

output_path = 'zh.wav'
model.tts_to_file(text, speaker_ids['ZH'], output_path, speed=speed)
print(f"课程一音频已保存: {output_path}")

# ==================== 课程二 ====================
print("\n" + "=" * 60)
print("课程二：语音合成中的多音字问题")
print("=" * 60)

# --- 问题二：G2P 和自定义 phones ---

from melo.text.chinese_mix import text_normalize, g2p, get_bert_feature
import torch

text_raw = "我最近在学习machine learning，希望能够在未来的artificial intelligence领域有所建树。"
text_norm = text_normalize(text_raw)
phones, tones, word2ph = g2p(text_norm)
print(f"\nG2P 示例:")
print(f"  text_norm: {text_norm}")
print(f"  phones: {phones}")
print(f"  tones: {tones}")
print(f"  word2ph: {word2ph}")

symbol_to_id_map = model.symbol_to_id
id_to_symbol_map = {v: k for k, v in zip(symbol_to_id_map.keys(), symbol_to_id_map.values())}

phones_id = [symbol_to_id_map[p] for p in phones]
df = pd.DataFrame({
    'phones': phones,
    'phones_id': phones_id,
    'tones': tones,
})
print(f"\nphones/tones DataFrame:")
print(df.T)

# 使用默认 g2p 进行 TTS 合成
speed2 = 0.75
output_path2 = 'zh_2.wav'
model.tts_to_file(text_raw, speaker_ids['ZH'], output_path2, speed=speed2)
print(f"\n默认 g2p 合成音频: {output_path2}")

# 多音字对联
print("\n--- 多音字对联自定义合成 ---")
text_hw2 = "海水朝，朝朝朝，朝朝朝落；浮云长，长长长，长长长消。"
split_sentences = TTS.split_sentences_into_pieces(text_hw2, language='ZH')
print(f"切分子句: {split_sentences}")

# 先跑默认 g2p 看结构
for s in split_sentences:
    norm_s = text_normalize(s)
    p, t, w = g2p(norm_s)
    print(f"\n  子句 '{s}':")
    print(f"    norm: {norm_s}")
    print(f"    phones({len(p)}): {p}")
    print(f"    tones({len(t)}): {t}")
    print(f"    word2ph({len(w)}): {w}, sum={sum(w)}")

# 根据郭沫若读法自定义（word2ph 必须跟默认 g2p 的结构一致：每个字/标点一个值）
# 子句一：海水朝(cháo)，朝(zhāo)朝(zhāo)朝(cháo)，朝(zhāo)朝(cháo)朝(zhāo)落(luò)
# 结构：_ 海 水 朝 ， 朝 朝 朝 ， 朝 朝 朝 落 . _
phones1 = ['_', 'h', 'ai', 'sh', 'ui', 'ch', 'ao', ',', 'zh', 'ao', 'zh', 'ao', 'ch', 'ao', ',', 'zh', 'ao', 'ch', 'ao', 'zh', 'ao', 'l', 'uo', '.', '_']
tones1  = [0,   3,    3,    3,    3,    2,    2,   0,    1,    1,    1,    1,    2,    2,   0,    1,    1,    2,    2,    1,    1,    4,    4,   0,   0]
word2ph1 = [1,  2,    2,    2,    1,    2,    2,    2,    1,    2,    2,    2,    2,    1,    1]

# 子句二：浮云长(zhǎng)，长(cháng)长(cháng)长(zhǎng)，长(cháng)长(zhǎng)长(cháng)消(xiāo)
# 结构：_ 浮 云 长 ， 长 长 长 ， 长 长 长 消 . _
phones2 = ['_', 'f', 'u', 'y', 'vn', 'zh', 'ang', ',', 'ch', 'ang', 'ch', 'ang', 'zh', 'ang', ',', 'ch', 'ang', 'zh', 'ang', 'ch', 'ang', 'x', 'iao', '.', '_']
tones2  = [0,   2,    2,   2,    2,    3,     3,    0,    2,     2,    2,     2,    3,     3,    0,    2,     2,    3,     3,    2,     2,    1,     1,    0,   0]
word2ph2 = [1,  2,    2,    2,    1,    2,    2,    2,    1,    2,    2,    2,    2,    1,    1]

assert sum(word2ph1) == len(phones1), f"子句一: sum={sum(word2ph1)}, len={len(phones1)}"
assert sum(word2ph2) == len(phones2), f"子句二: sum={sum(word2ph2)}, len={len(phones2)}"
assert len(tones1) == len(phones1)
assert len(tones2) == len(phones2)
print(f"\n验证通过: 子句一 {len(phones1)} phones, 子句二 {len(phones2)} phones")

phones_customized = [phones1, phones2]
tones_customized = [tones1, tones2]
word2ph_customized = [word2ph1, word2ph2]

output_path_hw2 = 'homework_2.wav'
model.tts_to_file_custom_frontend(text_hw2, speaker_ids['ZH'], output_path_hw2, speed=0.75,
                                   phones_customized=phones_customized,
                                   tones_customized=tones_customized,
                                   word2ph_customized=word2ph_customized)
print(f"多音字对联合成音频: {output_path_hw2}")

# ==================== 课程三 ====================
print("\n" + "=" * 60)
print("课程三：韵律控制")
print("=" * 60)

text_hw3_example = "落霞与孤鹜齐飞，秋水共长天一色。"
output_path_hw3_example = 'zh_hw3_example.wav'

w_ceil_list, phone_list, tone_list = model.get_original_w_ceil(
    text_hw3_example, speaker_ids['ZH'], output_path_hw3_example,
    speed=1, sdp_ratio=0, noise_scale=0, noise_scale_w=0
)

id_to_symbol = {v: k for k, v in model.symbol_to_id.items()}
df_hw3 = pd.DataFrame({
    'phones': [id_to_symbol.get(item, '') for sublist in phone_list for item in sublist.flatten().tolist()],
    'tones': [item for sublist in tone_list for item in sublist.flatten().tolist()],
    'w_ceil': [item for sublist in w_ceil_list for item in sublist.flatten().int().tolist()]
})
print(f"\n落霞与孤鹜齐飞 示例 w_ceil:")
print(df_hw3.T)

modified_w_ceil_list = w_ceil_list[0].squeeze(0).int().tolist()
modified_w_ceil_list[0][15:23] = [9] * 8
modified_w_ceil_list[0][45:53] = [9] * 8
print(f"\n修改后 w_ceil: {modified_w_ceil_list}")

model.tts_to_file(text_hw3_example, speaker_ids['ZH'], 'zh_hw3_original.wav',
                   speed=1, sdp_ratio=0, noise_scale=0, noise_scale_w=0)
print("原始韵律音频: zh_hw3_original.wav")

model.tts_to_file_custom_duration(text_hw3_example, speaker_ids['ZH'], 'zh_hw3_modified.wav',
                                   speed=1, sdp_ratio=0, noise_scale=0, noise_scale_w=0,
                                   w_ceil_customized=modified_w_ceil_list)
print("修改韵律音频: zh_hw3_modified.wav")

# --- 问题三：渐慢语音 ---
print("\n--- 问题三：渐慢语音合成 ---")
text_hw3 = "你听到了吗？这句话我会说得越来越慢。"
output_path_hw3 = 'homework_3.wav'

w_ceil_hw3, phone_hw3, tone_hw3 = model.get_original_w_ceil(
    text_hw3, speaker_ids['ZH'], output_path_hw3, speed=1,
    sdp_ratio=0, noise_scale=0, noise_scale_w=0
)

modified_hw3 = w_ceil_hw3[0].squeeze(0).int().tolist()
n = len(modified_hw3[0])
for i in range(n):
    scale = 1.0 + 2.0 * i / (n - 1)
    modified_hw3[0][i] = max(1, int(round(modified_hw3[0][i] * scale)))

print(f"原始 w_ceil: {w_ceil_hw3[0].squeeze(0).int().tolist()}")
print(f"修改后 w_ceil: {modified_hw3}")

model.tts_to_file_custom_duration(
    text_hw3, speaker_ids['ZH'], output_path_hw3, speed=1,
    sdp_ratio=0, noise_scale=0, noise_scale_w=0,
    w_ceil_customized=modified_hw3
)
print(f"渐慢语音音频: {output_path_hw3}")

print("\n" + "=" * 60)
print("所有实验代码执行完成！")
for f in ['zh.wav', 'zh_2.wav', 'homework_2.wav', 'zh_hw3_example.wav',
          'zh_hw3_original.wav', 'zh_hw3_modified.wav', 'homework_3.wav']:
    if os.path.exists(f):
        print(f"  {f} ({os.path.getsize(f)} bytes)")
    else:
        print(f"  {f} [未生成]")
print("=" * 60)
