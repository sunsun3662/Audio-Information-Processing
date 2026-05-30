"""
选题二：MeloTTS 的推理优化 - 完整实验脚本
生成所有实验结果，后续填入 notebook。
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

import sys, time, json
sys.path.insert(0, '.')

import numpy as np
import soundfile as sf

# ===================== 1. 准备测试文本（50+ 条） =====================
test_texts = [
    "今天天气真好，适合出去散步。",
    "人工智能正在改变我们的生活方式。",
    "这道菜的味道非常鲜美，我很喜欢。",
    "请把窗户关一下，外面太吵了。",
    "明天上午九点有一个重要的会议。",
    "中国的经济发展速度令人瞩目。",
    "这部电影的剧情非常引人入胜。",
    "学习一门新的语言需要长期的坚持。",
    "春天来了，公园里的花都开了。",
    "请问去火车站怎么走？",
    "科技的进步让世界变得更加紧密。",
    "他每天早上都会跑步锻炼身体。",
    "这本书的内容非常有深度和见解。",
    "我们需要保护环境，减少污染。",
    "音乐能够治愈人的心灵。",
    "她在比赛中获得了金牌，为国争光。",
    "互联网让信息传播变得更加迅速。",
    "老师耐心地为学生解答问题。",
    "这个项目的进展比预期要快得多。",
    "健康饮食是保持身体健康的关键。",
    "机器学习在医疗领域有广泛的应用。",
    "他是一位非常有经验的工程师。",
    "城市的夜景灯火辉煌，十分壮观。",
    "阅读可以开阔视野，增长知识。",
    "我们应该珍惜身边的每一个人。",
    "云计算技术正在改变企业的运营方式。",
    "这家餐厅的环境非常优雅舒适。",
    "读书使人充实，思考使人深刻。",
    "旅行是了解不同文化的最好方式。",
    "坚持锻炼能够增强体质，预防疾病。",
    "她在音乐方面有着极高的天赋。",
    "创新是推动社会进步的重要动力。",
    "这部电影值得一看，非常感人。",
    "他为了梦想不懈努力，终于成功了。",
    "教育是国家发展的基石。",
    "秋天的树叶变成了金黄色，非常美丽。",
    "我们需要学会管理自己的时间。",
    "这个城市的历史可以追溯到两千年前。",
    "合作是实现共赢的最佳途径。",
    "冬天的早晨，空气格外清新。",
    "他在演讲中表达了自己的观点。",
    "数字化转型是企业发展的必然趋势。",
    "父母的爱是世界上最伟大的爱。",
    "运动可以释放压力，让人更加自信。",
    "她用心制作了一件精美的手工艺品。",
    "人工智能技术在自动驾驶中有重要应用。",
    "志愿服务是一种美好的社会行为。",
    "春天是播种希望的季节。",
    "他在科研领域取得了重大突破。",
    "读书破万卷，下笔如有神。",
]

print(f"测试文本数量: {len(test_texts)}")

# ===================== 2. 原始 MeloTTS (PyTorch) 推理 =====================
print("\n" + "=" * 60)
print("原始 MeloTTS (PyTorch) 推理")
print("=" * 60)

from melo.api import TTS

device = 'cpu'
model = TTS(language='ZH', device=device, use_hf=False, use_ms=True)
speaker_ids = model.hps.data.spk2id

os.makedirs('results_hw4', exist_ok=True)

# 跑 50 条，记录 RTF
pytorch_rtf_list = []
pytorch_audio_files = []

for i, text in enumerate(test_texts):
    output_path = f'results_hw4/pytorch_{i:03d}.wav'
    t_start = time.time()
    model.tts_to_file(text, speaker_ids['ZH'], output_path, speed=1.0, quiet=True)
    t_end = time.time()

    # 计算音频时长
    audio_data, sr = sf.read(output_path)
    audio_duration = len(audio_data) / sr
    inference_time = t_end - t_start
    rtf = inference_time / audio_duration

    pytorch_rtf_list.append(rtf)
    pytorch_audio_files.append(output_path)

    if (i + 1) % 10 == 0:
        print(f"  PyTorch: {i+1}/{len(test_texts)} done, avg RTF={np.mean(pytorch_rtf_list):.4f}")

print(f"PyTorch 平均 RTF: {np.mean(pytorch_rtf_list):.4f}")
print(f"PyTorch 中位数 RTF: {np.median(pytorch_rtf_list):.4f}")

# ===================== 3. sherpa-onnx 推理（不同线程数） =====================
print("\n" + "=" * 60)
print("sherpa-onnx (ONNX) 推理 - 不同线程数")
print("=" * 60)

import sherpa_onnx

model_dir = 'vits-melo-tts-zh_en'
thread_counts = [1, 2, 4]
onnx_results = {}

for num_threads in thread_counts:
    print(f"\n--- num_threads={num_threads} ---")
    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=f'{model_dir}/model.onnx',
                tokens=f'{model_dir}/tokens.txt',
                lexicon=f'{model_dir}/lexicon.txt',
                dict_dir=f'{model_dir}/dict',
            ),
            provider='cpu',
            num_threads=num_threads,
        ),
        max_num_sentences=2,
    )
    tts_onnx = sherpa_onnx.OfflineTts(tts_config)

    rtf_list = []
    audio_files = []

    for i, text in enumerate(test_texts):
        output_path = f'results_hw4/onnx_t{num_threads}_{i:03d}.wav'
        t_start = time.time()
        audio = tts_onnx.generate(text, sid=0, speed=1.0)
        t_end = time.time()

        sf.write(output_path, audio.samples, audio.sample_rate)
        audio_duration = len(audio.samples) / audio.sample_rate
        inference_time = t_end - t_start
        rtf = inference_time / audio_duration

        rtf_list.append(rtf)
        audio_files.append(output_path)

        if (i + 1) % 10 == 0:
            print(f"  ONNX t={num_threads}: {i+1}/{len(test_texts)} done, avg RTF={np.mean(rtf_list):.4f}")

    onnx_results[num_threads] = {
        'rtf_list': rtf_list,
        'audio_files': audio_files,
        'mean_rtf': np.mean(rtf_list),
        'median_rtf': np.median(rtf_list),
    }
    print(f"  ONNX t={num_threads} 平均 RTF: {np.mean(rtf_list):.4f}")

# ===================== 4. 语音质量对比（DNSMOS） =====================
print("\n" + "=" * 60)
print("语音质量评估 - DNSMOS")
print("=" * 60)

import soundfile as sf

def compute_dnsmos(audio_path, sr=16000):
    """计算 DNSMOS 分数"""
    import onnxruntime as ort
    # 使用 speechmos 库
    try:
        from speechmos import dnsmos
        audio, orig_sr = sf.read(audio_path)
        if orig_sr != sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        scores = dnsmos.run(audio, sr)
        return scores
    except Exception as e:
        print(f"  DNSMOS error: {e}")
        return None

# 评估 PyTorch 模型生成的语音质量
print("评估 PyTorch 语音质量...")
pytorch_dnsmos = []
for i, fpath in enumerate(pytorch_audio_files):
    scores = compute_dnsmos(fpath)
    if scores:
        pytorch_dnsmos.append(scores)
    if (i + 1) % 10 == 0:
        print(f"  PyTorch DNSMOS: {i+1}/{len(pytorch_audio_files)}")

if pytorch_dnsmos:
    pytorch_dnsmos_mean = {
        k: np.mean([s[k] for s in pytorch_dnsmos])
        for k in pytorch_dnsmos[0].keys()
    }
    print(f"PyTorch DNSMOS 平均: {pytorch_dnsmos_mean}")
else:
    print("PyTorch DNSMOS 评估失败")

# 评估 ONNX 模型（取 num_threads=2 的结果）
print("\n评估 ONNX 语音质量 (num_threads=2)...")
onnx_audio_files_2 = onnx_results[2]['audio_files']
onnx_dnsmos = []
for i, fpath in enumerate(onnx_audio_files_2):
    scores = compute_dnsmos(fpath)
    if scores:
        onnx_dnsmos.append(scores)
    if (i + 1) % 10 == 0:
        print(f"  ONNX DNSMOS: {i+1}/{len(onnx_audio_files_2)}")

if onnx_dnsmos:
    onnx_dnsmos_mean = {
        k: np.mean([s[k] for s in onnx_dnsmos])
        for k in onnx_dnsmos[0].keys()
    }
    print(f"ONNX DNSMOS 平均: {onnx_dnsmos_mean}")
else:
    print("ONNX DNSMOS 评估失败")

# ===================== 5. 语音质量对比（Whisper CER） =====================
print("\n" + "=" * 60)
print("语音质量评估 - Whisper CER")
print("=" * 60)

try:
    import whisper
    whisper_model = whisper.load_model("base", device="cpu")
    print("Whisper 模型加载成功")

    def compute_cer(reference, hypothesis):
        """计算字符错误率 CER"""
        import editdistance
        return editdistance.eval(reference, hypothesis) / max(len(reference), 1)

    # 评估 PyTorch 语音的 CER
    print("\n评估 PyTorch 语音 CER...")
    pytorch_cers = []
    for i, (text, fpath) in enumerate(zip(test_texts, pytorch_audio_files)):
        result = whisper_model.transcribe(fpath, language="Chinese")
        hyp = result["text"].strip()
        cer = compute_cer(text, hyp)
        pytorch_cers.append(cer)
        if (i + 1) % 10 == 0:
            print(f"  PyTorch CER: {i+1}/{len(test_texts)}, avg={np.mean(pytorch_cers):.4f}")

    print(f"PyTorch 平均 CER: {np.mean(pytorch_cers):.4f}")

    # 评估 ONNX 语音的 CER
    print("\n评估 ONNX 语音 CER (num_threads=2)...")
    onnx_cers = []
    for i, (text, fpath) in enumerate(zip(test_texts, onnx_audio_files_2)):
        result = whisper_model.transcribe(fpath, language="Chinese")
        hyp = result["text"].strip()
        cer = compute_cer(text, hyp)
        onnx_cers.append(cer)
        if (i + 1) % 10 == 0:
            print(f"  ONNX CER: {i+1}/{len(test_texts)}, avg={np.mean(onnx_cers):.4f}")

    print(f"ONNX 平均 CER: {np.mean(onnx_cers):.4f}")

except ImportError as e:
    print(f"Whisper 或 editdistance 未安装: {e}")
    print("尝试安装: pip install openai-whisper editdistance")

# ===================== 6. 汇总结果 =====================
print("\n" + "=" * 60)
print("实验结果汇总")
print("=" * 60)

results = {
    'pytorch': {
        'mean_rtf': float(np.mean(pytorch_rtf_list)),
        'median_rtf': float(np.median(pytorch_rtf_list)),
    },
}

for tc in thread_counts:
    results[f'onnx_t{tc}'] = {
        'mean_rtf': float(onnx_results[tc]['mean_rtf']),
        'median_rtf': float(onnx_results[tc]['median_rtf']),
    }

print(json.dumps(results, indent=2, ensure_ascii=False))

# 保存结果
with open('results_hw4/results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n实验完成！结果已保存到 results_hw4/results.json")
