"""
Lab5 实验三：说话人验证与说话人辨认
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

import warnings
warnings.filterwarnings('ignore')

import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(".").resolve()
DATA_DIR = PROJECT_ROOT / "data" / "aishell_mini"
OUT_DIR = PROJECT_ROOT / "outputs"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# 加载模型
from speechbrain.inference.speaker import EncoderClassifier
classifier = EncoderClassifier.from_hparams(
    source="LanceaKing/spkrec-ecapa-cnceleb",
    savedir="ckpt/spkrec-ecapa-cnceleb",
    run_opts={"device": device})

@torch.no_grad()
def extract_embedding_from_path(path):
    wav, sr = torchaudio.load(path)
    wavs = wav.squeeze(0).unsqueeze(0).to(device)
    emb = classifier.encode_batch(wavs).squeeze().detach().cpu()
    return F.normalize(emb, dim=0)

embedding_cache = {}
def get_embedding(path):
    path = str(path)
    if path not in embedding_cache:
        embedding_cache[path] = extract_embedding_from_path(DATA_DIR / path)
    return embedding_cache[path]

# ===================== 读取数据 =====================
enroll_df = pd.read_csv(DATA_DIR / "protocols" / "enroll.csv")
id_test_df = pd.read_csv(DATA_DIR / "protocols" / "identification_test.csv")
utterances_df = pd.read_csv(DATA_DIR / "metadata" / "utterances.csv")
dev_trials = pd.read_csv(DATA_DIR / "protocols" / "verification_trials_dev.csv")
test_trials = pd.read_csv(DATA_DIR / "protocols" / "verification_trials_test.csv")

print(f"enrollment speakers: {enroll_df['spk_id'].nunique()}")
print(f"identification test: {len(id_test_df)}")
print(f"dev trials: {len(dev_trials)}")
print(f"test trials: {len(test_trials)}")
print(f"dev trial labels:\n{dev_trials['label'].value_counts()}")

# ===================== Enrollment Templates =====================
print("\n=== 构建 Enrollment Templates ===")
speaker_templates = {}
for spk_id, group in enroll_df.groupby("spk_id"):
    embs = []
    for _, row in group.iterrows():
        embs.append(get_embedding(row["path"]))
    template = torch.stack(embs, dim=0).mean(dim=0)
    template = F.normalize(template, dim=0)
    speaker_templates[spk_id] = template

print(f"num templates: {len(speaker_templates)}")
print(f"template dim: {next(iter(speaker_templates.values())).shape}")

# ===================== Speaker Verification =====================
print("\n=== Speaker Verification ===")

def cosine_score(emb1, emb2):
    return torch.dot(F.normalize(emb1, dim=0), F.normalize(emb2, dim=0)).item()

lookup = utterances_df.set_index("utt_id")

def score_trials(trials_df):
    rows = []
    for _, trial in trials_df.iterrows():
        enroll_emb = speaker_templates[trial["enroll_spk_id"]]
        test_emb = get_embedding(lookup.loc[trial["test_utt_id"], "path"])
        score = cosine_score(enroll_emb, test_emb)
        rows.append({**trial.to_dict(), "score": score})
    return pd.DataFrame(rows)

dev_scores = score_trials(dev_trials)
test_scores = score_trials(test_trials)

dev_scores.to_csv(OUT_DIR / "scores" / "verification_dev_scores.csv", index=False)
test_scores.to_csv(OUT_DIR / "scores" / "verification_test_scores.csv", index=False)

# dev EER
def compute_far_frr(scores, labels, thresholds):
    rows = []
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores)
    pos, neg = labels == 1, labels == 0
    for tau in thresholds:
        pred_same = scores >= tau
        far = ((pred_same) & neg).sum() / max(neg.sum(), 1)
        frr = ((~pred_same) & pos).sum() / max(pos.sum(), 1)
        rows.append({"threshold": tau, "FAR": far, "FRR": frr, "abs_diff": abs(far - frr)})
    return pd.DataFrame(rows)

thresholds = np.linspace(dev_scores["score"].min(), dev_scores["score"].max(), 300)
dev_curve = compute_far_frr(dev_scores["score"], dev_scores["label"], thresholds)
best = dev_curve.iloc[dev_curve["abs_diff"].argmin()]
tau = best["threshold"]
eer = 0.5 * (best["FAR"] + best["FRR"])

print(f"Dev tau: {tau:.4f}")
print(f"Dev FAR: {best['FAR']:.4f}")
print(f"Dev FRR: {best['FRR']:.4f}")
print(f"Dev EER: {eer:.4f}")

# dev 直方图
same_scores = dev_scores[dev_scores["label"] == 1]["score"]
diff_scores = dev_scores[dev_scores["label"] == 0]["score"]
plt.figure(figsize=(7, 4))
plt.hist(same_scores, bins=20, alpha=0.6, label="same speaker")
plt.hist(diff_scores, bins=20, alpha=0.6, label="different speakers")
plt.axvline(tau, linestyle="--", label="dev threshold")
plt.xlabel("cosine score"); plt.ylabel("count")
plt.title("Verification score distribution on dev trials")
plt.legend(); plt.tight_layout()
plt.savefig(OUT_DIR / "figures" / "verification_score_hist_dev.png", dpi=150); plt.close()

# FAR/FRR 曲线
plt.figure(figsize=(7, 4))
plt.plot(dev_curve["threshold"], dev_curve["FAR"], label="FAR")
plt.plot(dev_curve["threshold"], dev_curve["FRR"], label="FRR")
plt.axvline(tau, linestyle="--", label="approx EER threshold")
plt.xlabel("threshold"); plt.ylabel("rate")
plt.title("FAR / FRR on dev trials")
plt.legend(); plt.tight_layout()
plt.savefig(OUT_DIR / "figures" / "far_frr_curve_dev.png", dpi=150); plt.close()

# test evaluation
def evaluate_at_threshold(score_df, tau):
    labels = score_df["label"].values.astype(int)
    scores = score_df["score"].values
    pred_same = scores >= tau
    pos, neg = labels == 1, labels == 0
    far = ((pred_same) & neg).sum() / max(neg.sum(), 1)
    frr = ((~pred_same) & pos).sum() / max(pos.sum(), 1)
    acc = (pred_same.astype(int) == labels).mean()
    return {"threshold": tau, "FAR": far, "FRR": frr, "accuracy": acc}

test_eval = evaluate_at_threshold(test_scores, tau)
print(f"\nTest evaluation: {test_eval}")

# test 直方图
same_scores_t = test_scores[test_scores["label"] == 1]["score"]
diff_scores_t = test_scores[test_scores["label"] == 0]["score"]
plt.figure(figsize=(7, 4))
plt.hist(same_scores_t, bins=20, alpha=0.6, label="same speaker")
plt.hist(diff_scores_t, bins=20, alpha=0.6, label="different speakers")
plt.axvline(tau, linestyle="--", label="test threshold")
plt.xlabel("cosine score"); plt.ylabel("count")
plt.title("Verification score distribution on test trials")
plt.legend(); plt.tight_layout()
plt.savefig(OUT_DIR / "figures" / "verification_score_hist_test.png", dpi=150); plt.close()

# ===================== Speaker Identification =====================
print("\n=== Speaker Identification (Closed-set) ===")
spk_ids = sorted(speaker_templates.keys())
id_rows = []

for _, row in id_test_df[id_test_df["is_known"] == 1].iterrows():
    test_emb = get_embedding(row["path"])
    scores = [cosine_score(speaker_templates[s], test_emb) for s in spk_ids]
    best_idx = int(np.argmax(scores))
    pred_spk = spk_ids[best_idx]
    id_rows.append({
        "utt_id": row["utt_id"], "true_spk": row["spk_id"],
        "pred_spk": pred_spk, "best_score": scores[best_idx],
        "correct": pred_spk == row["spk_id"],
    })

id_result = pd.DataFrame(id_rows)
id_acc = id_result["correct"].mean()
print(f"Closed-set identification accuracy: {id_acc:.4f}")
id_result.to_csv(OUT_DIR / "scores" / "identification_closed_set.csv", index=False)

# score matrix heatmap
score_mat = np.zeros((len(spk_ids), len(id_result)), dtype=np.float32)
for i, spk_id in enumerate(spk_ids):
    for j, row in id_result.iterrows():
        path = id_test_df.set_index("utt_id").loc[row["utt_id"], "path"]
        score_mat[i, j] = cosine_score(speaker_templates[spk_id], get_embedding(path))

plt.figure(figsize=(10, 5))
plt.imshow(score_mat, aspect="auto")
plt.colorbar(label="cosine score")
plt.xticks(range(len(id_result)), id_result["utt_id"].tolist(), rotation=90)
plt.yticks(range(len(spk_ids)), spk_ids)
plt.xlabel("test utterance"); plt.ylabel("enrollment speaker")
plt.title("Identification score matrix")
plt.tight_layout()
plt.savefig(OUT_DIR / "figures" / "identification_score_matrix.png", dpi=150); plt.close()

# Open-set identification
print("\n=== Speaker Identification (Open-set) ===")
open_rows = []
for _, row in id_test_df.iterrows():
    test_emb = get_embedding(row["path"])
    scores = [cosine_score(speaker_templates[s], test_emb) for s in spk_ids]
    best_idx = int(np.argmax(scores))
    open_rows.append({
        "utt_id": row["utt_id"], "true_spk": row["spk_id"],
        "is_known": row["is_known"], "best_spk": spk_ids[best_idx],
        "best_score": scores[best_idx],
    })

open_df = pd.DataFrame(open_rows)
tau_id = tau
open_df["pred_open"] = open_df.apply(
    lambda r: r["best_spk"] if r["best_score"] >= tau_id else "unknown", axis=1)
print(open_df.to_string())

# 保存结果汇总
results_summary = {
    "verification_dev_eer": float(eer),
    "verification_dev_tau": float(tau),
    "verification_test_far": float(test_eval["FAR"]),
    "verification_test_frr": float(test_eval["FRR"]),
    "verification_test_accuracy": float(test_eval["accuracy"]),
    "identification_closed_set_accuracy": float(id_acc),
}
import json
with open(OUT_DIR / "reports" / "lab5_results.json", "w") as f:
    json.dump(results_summary, f, indent=2)

print(f"\n结果汇总: {json.dumps(results_summary, indent=2)}")
print("实验三完成！")
