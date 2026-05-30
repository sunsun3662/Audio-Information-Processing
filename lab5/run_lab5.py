"""
Lab5: 说话人识别实验 - 完整执行脚本
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
import torch.nn.functional as F

PROJECT_ROOT = Path(".").resolve()
DATA_DIR = PROJECT_ROOT / "data" / "aishell_mini"
OUT_DIR = PROJECT_ROOT / "outputs"
CKPT_DIR = PROJECT_ROOT / "ckpt" / "spkrec-ecapa-cnceleb"

for sub in ["embeddings", "scores", "figures", "reports"]:
    (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ===================== 实验一 =====================
print("\n" + "=" * 60)
print("实验一：环境安装与 ECAPA-TDNN 初次推理")
print("=" * 60)

from speechbrain.inference.speaker import EncoderClassifier

classifier = EncoderClassifier.from_hparams(
    source="LanceaKing/spkrec-ecapa-cnceleb",
    savedir="ckpt/spkrec-ecapa-cnceleb",
    run_opts={"device": device})
print(f"Model loaded on: {device}")
print(f"Modules: {list(classifier.mods.keys())}")

# 查看 hyperparams
hparam_path = CKPT_DIR / "hyperparams.yaml"
with open(hparam_path, "r", encoding="utf-8") as f:
    text = f.read()
print(f"\nhyperparams.yaml (前500字符):\n{text[:500]}")

# 读取示例音频
example_wav = "./data/sample.wav"
signal, sr = torchaudio.load(example_wav)
print(f"\nsignal dtype: {signal.dtype}")
print(f"signal shape: {signal.shape}")
print(f"sample rate: {sr}")
print(f"min: {signal.min().item():.4f}, max: {signal.max().item():.4f}")
print(f"duration: {signal.shape[-1] / sr:.2f}s")

# 提取 embedding
@torch.no_grad()
def extract_embedding(signal, sr):
    wavs = signal.squeeze(0).unsqueeze(0).to(device)
    emb = classifier.encode_batch(wavs)
    emb = emb.squeeze().detach().cpu()
    emb_l2 = F.normalize(emb, dim=0)
    return emb, emb_l2

emb, emb_l2 = extract_embedding(signal, sr)
print(f"\nembedding shape: {emb.shape}")
print(f"raw embedding norm: {emb.norm().item():.4f}")
print(f"L2-normalized embedding norm: {emb_l2.norm().item():.4f}")

# 保存 embedding
np.save(OUT_DIR / "embeddings" / "example1_embedding.npy", emb_l2.numpy())
print(f"Saved: {OUT_DIR / 'embeddings' / 'example1_embedding.npy'}")

# SciPy 对比
from scipy.io import wavfile
rate, data = wavfile.read(example_wav)
print(f"\nscipy sample rate: {rate}")
print(f"scipy dtype: {data.dtype}")
print(f"scipy shape: {data.shape}")
print(f"scipy min/max: {data.min()}, {data.max()}")

# 测试 scipy int16 直接输入
data_float = data.astype(np.float32) / 32768.0
data_tensor = torch.from_numpy(data_float).unsqueeze(0).to(device)
emb_scipy = classifier.encode_batch(data_tensor).squeeze().detach().cpu()
emb_scipy_l2 = F.normalize(emb_scipy, dim=0)
print(f"scipy embedding norm: {emb_scipy.norm().item():.4f}")
print(f"cosine(torchaudio, scipy): {F.cosine_similarity(emb_l2.unsqueeze(0), emb_scipy_l2.unsqueeze(0)).item():.4f}")

# 测试不缩放的 int16
data_int16_tensor = torch.from_numpy(data.astype(np.float32)).unsqueeze(0).to(device)
emb_int16 = classifier.encode_batch(data_int16_tensor).squeeze().detach().cpu()
emb_int16_l2 = F.normalize(emb_int16, dim=0)
print(f"int16 (no scale) embedding norm: {emb_int16.norm().item():.4f}")
print(f"cosine(torchaudio, int16_no_scale): {F.cosine_similarity(emb_l2.unsqueeze(0), emb_int16_l2.unsqueeze(0)).item():.4f}")

# ===================== 实验二 =====================
print("\n" + "=" * 60)
print("实验二：说话人嵌入空间与扰动分析")
print("=" * 60)

speakers = pd.read_csv(DATA_DIR / "metadata" / "speakers.csv")
utts = pd.read_csv(DATA_DIR / "metadata" / "utterances.csv")

print(f"num speakers: {speakers['spk_id'].nunique()}")
print(f"num utterances: {len(utts)}")
print(f"gender distribution:\n{speakers['gender'].value_counts()}")
print(f"duration statistics:\n{utts['duration_sec'].describe()}")

# 批量提取 clean embeddings
from tqdm.auto import tqdm

@torch.no_grad()
def extract_embedding_from_path(path):
    wav, sr = torchaudio.load(path)
    wavs = wav.squeeze(0).unsqueeze(0).to(device)
    emb = classifier.encode_batch(wavs).squeeze().detach().cpu()
    emb = F.normalize(emb, dim=0)
    return emb

print("\n提取 clean embeddings...")
emb_rows = []
emb_list = []
for _, row in tqdm(utts.iterrows(), total=len(utts)):
    wav_path = DATA_DIR / row["path"]
    emb = extract_embedding_from_path(wav_path)
    emb_list.append(emb.numpy())
    emb_rows.append({
        "utt_id": row["utt_id"], "spk_id": row["spk_id"],
        "path": row["path"], "condition": "clean",
    })

emb_arr = np.stack(emb_list, axis=0)
emb_meta = pd.DataFrame(emb_rows)
emb_meta_merged = emb_meta.merge(speakers, on="spk_id", how="left")

np.save(OUT_DIR / "embeddings" / "embeddings_clean.npy", emb_arr)
emb_meta.to_csv(OUT_DIR / "embeddings" / "embeddings_clean_meta.csv", index=False)
print(f"embedding array: {emb_arr.shape}")

# t-SNE 可视化
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_tsne(embeddings, meta, title, out_path):
    n = embeddings.shape[0]
    perplexity = min(30, max(5, (n - 1) // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca",
                learning_rate="auto", random_state=42)
    xy = tsne.fit_transform(embeddings)
    meta = meta.copy()
    meta["x"] = xy[:, 0]
    meta["y"] = xy[:, 1]

    spk_ids = sorted(meta["spk_id"].unique())
    gender_markers = {0: "o", 1: "^", "0": "o", "1": "^"}
    has_multi = "condition" in meta.columns and meta["condition"].nunique() > 1

    plt.figure(figsize=(9, 7))
    for spk_id in spk_ids:
        sub_spk = meta[meta["spk_id"] == spk_id]
        group_cols = ["gender", "condition"] if has_multi else ["gender"]
        for group_key, sub in sub_spk.groupby(group_cols):
            gender = group_key[0] if has_multi else group_key
            condition = group_key[1] if has_multi else "clean"
            marker = gender_markers.get(gender, "s")
            alpha = 0.85 if condition == "clean" else 0.35
            gender_name = "Male" if str(gender) == "0" else "Female"
            label = f"{spk_id}-{gender_name}" if not has_multi else f"{spk_id}-{gender_name}-{condition}"
            plt.scatter(sub["x"], sub["y"], marker=marker, label=label, alpha=alpha)

    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")
    return meta

print("\nt-SNE 可视化 clean embeddings...")
tsne_clean_meta = plot_tsne(emb_arr, emb_meta_merged,
    "Clean speaker embeddings t-SNE",
    OUT_DIR / "figures" / "tsne_clean.png")

# 扰动函数
def add_white_noise_snr(wav, snr_db):
    noise = torch.randn_like(wav)
    wav_power = (wav ** 2).mean()
    noise_power = (noise ** 2).mean()
    scale = torch.sqrt(wav_power / (noise_power * (10 ** (snr_db / 10))))
    noisy = wav + scale * noise
    noisy = noisy / (noisy.abs().max() + 1e-8)
    return noisy

def simulate_8k_channel(wav):
    down = torchaudio.transforms.Resample(16000, 8000)(wav)
    up = torchaudio.transforms.Resample(8000, 16000)(down)
    up = up / (up.abs().max() + 1e-8)
    return up

# 自定义扰动：时间拉伸
def time_stretch(wav, rate=1.2):
    import torchaudio.functional as F_ta
    stretched = torchaudio.transforms.TimeStretch(fixed_rate=rate)(wav.unsqueeze(0).to(torch.complex64 if wav.is_complex() else torch.float32))
    return stretched.squeeze(0).real if stretched.is_complex() else stretched.squeeze(0)

# 生成增强音频
print("\n生成增强音频...")
AUG_DIR = DATA_DIR / "augmented"
AUG_DIR.mkdir(parents=True, exist_ok=True)

aug_rows = []
for _, row in tqdm(utts.iterrows(), total=len(utts)):
    wav_path = DATA_DIR / row["path"]
    wav, sr_val = torchaudio.load(wav_path)

    # noise
    noisy = add_white_noise_snr(wav, snr_db=10)
    noisy_rel = f"augmented/noise_snr10/{row['spk_id']}/{row['utt_id']}_noise10.wav"
    noisy_abs = DATA_DIR / noisy_rel
    noisy_abs.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(noisy_abs), noisy.cpu(), 16000)
    aug_rows.append({"utt_id": row["utt_id"] + "_noise10", "orig_utt_id": row["utt_id"],
        "spk_id": row["spk_id"], "path": noisy_rel, "condition": "noise_snr10"})

    # channel
    ch = simulate_8k_channel(wav)
    ch_rel = f"augmented/channel_8k/{row['spk_id']}/{row['utt_id']}_ch8k.wav"
    ch_abs = DATA_DIR / ch_rel
    ch_abs.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(ch_abs), ch.cpu(), 16000)
    aug_rows.append({"utt_id": row["utt_id"] + "_ch8k", "orig_utt_id": row["utt_id"],
        "spk_id": row["spk_id"], "path": ch_rel, "condition": "channel_8k"})

aug_df = pd.DataFrame(aug_rows)
aug_df = aug_df.merge(speakers, on="spk_id", how="left")
aug_df.to_csv(DATA_DIR / "metadata" / "augmentations.csv", index=False)
print(f"augmented utterances: {len(aug_df)}")

# 提取 augmented embeddings
print("\n提取 augmented embeddings...")
aug_embs = []
aug_meta_rows = []
for _, row in tqdm(aug_df.iterrows(), total=len(aug_df)):
    emb = extract_embedding_from_path(DATA_DIR / row["path"])
    aug_embs.append(emb.numpy())
    aug_meta_rows.append({
        "utt_id": row["utt_id"], "orig_utt_id": row["orig_utt_id"],
        "spk_id": row["spk_id"], "path": row["path"],
        "condition": row["condition"], "gender": row["gender"],
    })

aug_emb_arr = np.stack(aug_embs, axis=0)
aug_meta = pd.DataFrame(aug_meta_rows)
np.save(OUT_DIR / "embeddings" / "embeddings_augmented.npy", aug_emb_arr)
aug_meta.to_csv(OUT_DIR / "embeddings" / "embeddings_augmented_meta.csv", index=False)
print(f"augmented embedding array: {aug_emb_arr.shape}")

# clean + augmented t-SNE
print("\nt-SNE 可视化 clean + augmented...")
combined_emb = np.concatenate([emb_arr, aug_emb_arr], axis=0)
clean_meta_for_plot = emb_meta_merged.copy()
clean_meta_for_plot["condition"] = "clean"
clean_meta_for_plot["orig_utt_id"] = clean_meta_for_plot["utt_id"]
combined_meta = pd.concat([clean_meta_for_plot, aug_meta], ignore_index=True)

plot_tsne(combined_emb, combined_meta,
    "Clean and perturbed speaker embeddings t-SNE",
    OUT_DIR / "figures" / "tsne_clean_augmented_joint.png")

# embedding drift 分析
print("\n计算 embedding drift...")
clean_lookup = {row["utt_id"]: emb_arr[i] for i, row in emb_meta.reset_index(drop=True).iterrows()}
drift_rows = []
for i, row in aug_meta.reset_index(drop=True).iterrows():
    clean_emb = clean_lookup[row["orig_utt_id"]]
    aug_emb = aug_emb_arr[i]
    cos = float(np.dot(clean_emb, aug_emb) / (np.linalg.norm(clean_emb) * np.linalg.norm(aug_emb) + 1e-8))
    drift_rows.append({
        "orig_utt_id": row["orig_utt_id"], "aug_utt_id": row["utt_id"],
        "spk_id": row["spk_id"], "gender": row["gender"],
        "condition": row["condition"], "cos_clean_aug": cos, "embedding_drift": 1 - cos,
    })

drift_df = pd.DataFrame(drift_rows)
print(drift_df.groupby("condition")[["cos_clean_aug", "embedding_drift"]].describe())
drift_df.to_csv(OUT_DIR / "scores" / "embedding_drift_clean_augmented.csv", index=False)

# 箱线图
plt.figure(figsize=(7, 4))
conditions = sorted(drift_df["condition"].unique())
data = [drift_df[drift_df["condition"] == c]["embedding_drift"].values for c in conditions]
plt.boxplot(data, labels=conditions)
plt.ylabel("1 - cosine(clean, perturbed)")
plt.title("Embedding drift under perturbations")
plt.tight_layout()
plt.savefig(OUT_DIR / "figures" / "embedding_drift_boxplot.png", dpi=150)
plt.close()
print(f"Saved: {OUT_DIR / 'figures' / 'embedding_drift_boxplot.png'}")

# ===================== 实验三 =====================
print("\n" + "=" * 60)
print("实验三：ASV 与 ASI 评测")
print("=" * 60)

# ASV: 计算 cosine similarity scores
print("\n计算 ASV trials scores...")

# 读取 trials
trials_path = DATA_DIR / "protocols" / "asv_trials.csv"
if trials_path.exists():
    trials = pd.read_csv(trials_path)
    print(f"trials: {len(trials)}")
    print(f"trials columns: {list(trials.columns)}")
    print(trials.head())

    # 构建 embedding lookup
    all_emb_lookup = {}
    for i, row in emb_meta.iterrows():
        all_emb_lookup[row["utt_id"]] = emb_arr[i]
    for i, row in aug_meta.iterrows():
        all_emb_lookup[row["utt_id"]] = aug_emb_arr[i]

    # 计算 scores
    score_rows = []
    for _, trial in tqdm(trials.iterrows(), total=len(trials)):
        utt1 = trial["utt_id_1"] if "utt_id_1" in trials.columns else trial.iloc[0]
        utt2 = trial["utt_id_2"] if "utt_id_2" in trials.columns else trial.iloc[1]
        label = trial["label"] if "label" in trials.columns else trial.iloc[2]

        if utt1 in all_emb_lookup and utt2 in all_emb_lookup:
            emb1 = all_emb_lookup[utt1]
            emb2 = all_emb_lookup[utt2]
            cos = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))
            score_rows.append({"utt_id_1": utt1, "utt_id_2": utt2, "label": label, "score": cos})

    score_df = pd.DataFrame(score_rows)
    score_df.to_csv(OUT_DIR / "scores" / "asv_scores.csv", index=False)
    print(f"scores computed: {len(score_df)}")
    print(f"score distribution by label:\n{score_df.groupby('label')['score'].describe()}")

    # EER 计算
    from sklearn.metrics import roc_curve
    y_true = score_df["label"].values
    y_score = score_df["score"].values

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    print(f"\nEER: {eer:.4f}")

    # 保存 EER 报告
    with open(OUT_DIR / "reports" / "asv_eer.txt", "w") as f:
        f.write(f"EER: {eer:.4f}\n")
        f.write(f"Threshold: {thresholds[eer_idx]:.4f}\n")
        f.write(f"FPR: {fpr[eer_idx]:.4f}\n")
        f.write(f"FNR: {fnr[eer_idx]:.4f}\n")
else:
    print("asv_trials.csv 不存在，跳过 ASV 评测")

# ASI: 说话人识别
asi_path = DATA_DIR / "protocols" / "asi_trials.csv"
if asi_path.exists():
    asi_trials = pd.read_csv(asi_path)
    print(f"\nASI trials: {len(asi_trials)}")
    print(asi_trials.head())
else:
    print("\nasi_trials.csv 不存在，跳过 ASI 评测")

print("\n" + "=" * 60)
print("实验完成！")
print("=" * 60)
