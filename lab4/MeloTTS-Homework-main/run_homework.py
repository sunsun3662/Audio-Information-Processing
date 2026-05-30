import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CONDA_NO_PLUGINS'] = 'true'

import sys
sys.path.insert(0, '.')

from melo.api import TTS

# Test loading model
print("Loading TTS model...")
model = TTS(language='ZH', device='cpu', use_hf=False, use_ms=True)
speaker_ids = model.hps.data.spk2id
print(f"TTS model loaded. Speakers: {speaker_ids}")

# Homework 1: basic TTS
text = "我最近在学习machine learning，希望能够在未来的artificial intelligence领域有所建树。"
output_path = 'zh_hw1.wav'
model.tts_to_file(text, speaker_ids['ZH'], output_path, speed=1.0)
print(f"Homework 1 audio saved to {output_path}")

# Homework 2: polyphonic couplet
text2 = "海水朝，朝朝朝，朝朝朝落；浮云长，长长长，长长长消。"
split_sentences = TTS.split_sentences_into_pieces(text2, language='ZH')
print(f"Split sentences: {split_sentences}")

phones1 = ['h', 'ai', 'ch', 'ao', ',', 'zh', 'ao', 'zh', 'ao', 'ch', 'ao', ',', 'zh', 'ao', 'ch', 'ao', 'zh', 'ao', 'l', 'uo', '.']
tones1  = [3, 3, 2, 2, 0, 1, 1, 1, 1, 2, 2, 0, 1, 1, 2, 2, 1, 1, 4, 4, 0]
word2ph1 = [2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2]

phones2 = ['f', 'u', 'y', 'vn', 'zh', 'ang', ',', 'ch', 'ang', 'ch', 'ang', 'zh', 'ang', ',', 'ch', 'ang', 'zh', 'ang', 'ch', 'ang', 'x', 'iao', '.']
tones2  = [2, 2, 2, 2, 3, 3, 0, 2, 2, 2, 2, 3, 3, 0, 2, 2, 3, 3, 2, 2, 1, 1, 0]
word2ph2 = [2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1]

phones_customized = [phones1, phones2]
tones_customized = [tones1, tones2]
word2ph_customized = [word2ph1, word2ph2]

output_path2 = 'homework_2.wav'
model.tts_to_file_custom_frontend(text2, speaker_ids['ZH'], output_path2, speed=0.75,
                                   phones_customized=phones_customized,
                                   tones_customized=tones_customized,
                                   word2ph_customized=word2ph_customized)
print(f"Homework 2 audio saved to {output_path2}")

# Homework 3: gradually slowing speech
text3 = "你听到了吗？这句话我会说得越来越慢。"
output_path3 = 'homework_3.wav'

w_ceil_list, phone_list, tone_list = model.get_original_w_ceil(
    text3, speaker_ids['ZH'], output_path3, speed=1,
    sdp_ratio=0, noise_scale=0, noise_scale_w=0
)

modified_w_ceil = w_ceil_list[0].squeeze(0).int().tolist()
n = len(modified_w_ceil[0])
for i in range(n):
    scale = 1.0 + 2.0 * i / (n - 1)
    modified_w_ceil[0][i] = max(1, int(round(modified_w_ceil[0][i] * scale)))

print(f"Original w_ceil: {w_ceil_list[0].squeeze(0).int().tolist()}")
print(f"Modified w_ceil: {modified_w_ceil}")

model.tts_to_file_custom_duration(
    text3, speaker_ids['ZH'], output_path3, speed=1,
    sdp_ratio=0, noise_scale=0, noise_scale_w=0,
    w_ceil_customized=modified_w_ceil
)
print(f"Homework 3 audio saved to {output_path3}")
print("All homework done!")
