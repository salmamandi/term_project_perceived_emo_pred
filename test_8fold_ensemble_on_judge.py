"""
Testing Script: Evaluate 8-Fold Sub-Ensemble on Held-Out Test Movie 'TheJudge'
=============================================================================
STRICT ZERO-LEAKAGE EVALUATION PROTOCOL:
  - All contributing models were trained exclusively on the other 8 movies.
  - Zero models have ever seen a single frame from 'TheJudge'.
  - Evaluates:
      1. 8-Fold RG-MAN Ensemble on The Judge
      2. 8-Fold Decoupled MGN Ensemble on The Judge
      3. Master Combined Ensemble (RG-MAN + Decoupled) on The Judge
  - Metrics: Valence CCC, Arousal CCC, Valence Acc, Arousal Acc, Joint Clip Acc, High-Confidence Acc.

Outputs:
  Thilok_Outputs/results_8fold_no_judge_test_on_judge/
    - the_judge_8fold_ensemble_summary.csv
    - the_judge_per_clip_ensemble_predictions.csv

Usage:
  python test_8fold_ensemble_on_judge.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BTP_DIR = SCRIPT_DIR
ROOT_DIR = os.path.dirname(BTP_DIR)
sys.path.insert(0, BTP_DIR)

def resolve_path(p):
    if not p:
        return p
    if os.path.exists(p):
        return p
    p_btp = os.path.join(BTP_DIR, p)
    if os.path.exists(p_btp):
        return p_btp
    p_root = os.path.join(ROOT_DIR, p)
    if os.path.exists(p_root):
        return p_root
    return p

from Thilok_perceive_emo_pred_model import config
from Thilok_perceive_emo_pred_model.dataset import SceneFeatureDataset, collate_fn
from Thilok_perceive_emo_pred_model.model import PerceivedVA_Model
from Thilok_perceive_emo_pred_model.rule_guided_attention_model import RuleGuided_PerceivedVA_Model
from Thilok_perceive_emo_pred_model.utils import (
    load_labels, load_genre_labels, load_binary_labels,
    bin_value_binary, compute_metrics, compute_classification_metrics
)

# ── 1. Directories ─────────────────────────────────────────────────────────────
RGMAN_CKPT_DIR = os.path.join(BTP_DIR, "Thilok_Outputs", "saved_models_8fold_no_judge", "rg_man")
VAL_CKPT_DIR   = os.path.join(BTP_DIR, "Thilok_Outputs", "saved_models_8fold_no_judge", "valence")
ARO_CKPT_DIR   = os.path.join(BTP_DIR, "Thilok_Outputs", "saved_models_8fold_no_judge", "arousal")
OUTPUT_DIR     = os.path.join(BTP_DIR, "Thilok_Outputs", "results_8fold_no_judge_test_on_judge")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Judge Test] Using device: {device}")

# ── 2. Load Rule Priors ────────────────────────────────────────────────────────
audio_err_file = resolve_path("audio_valence_misalignment_check.csv")
bad_audio_clips = set()
if os.path.exists(audio_err_file):
    df_a_err = pd.read_csv(audio_err_file)
    bad_audio_clips = set(df_a_err["movie_clip"].astype(str).str.replace(".wav", "").values)

video_err_file = resolve_path("video_valence_misalignment_check.csv")
bad_video_clips = set()
if os.path.exists(video_err_file):
    df_v_err = pd.read_csv(video_err_file)
    bad_video_clips = set(df_v_err["movie_clip"].astype(str).str.replace(".mp4", "").values)

rule_json_file = resolve_path("useful_nonuseful_names_manual_correct.json")
useful_clips = set()
non_useful_clips = set()
if os.path.exists(rule_json_file):
    with open(rule_json_file, "r") as f:
        data = json.load(f)
        for entry in data:
            for c in entry.get("useful", []):
                useful_clips.add(str(c).replace(".mp4", "").replace(".wav", ""))
            for c in entry.get("non_useful", []):
                non_useful_clips.add(str(c).replace(".mp4", "").replace(".wav", ""))

def get_rule_prior(clip_name):
    c_base = str(clip_name).replace(".mp4", "").replace(".wav", "")
    if c_base in bad_video_clips or c_base in non_useful_clips:
        v_prior = 0.0
    elif c_base in useful_clips:
        v_prior = 1.0
    else:
        v_prior = 0.5

    if c_base in bad_audio_clips or c_base in non_useful_clips:
        a_prior = 0.0
    elif c_base in useful_clips:
        a_prior = 1.0
    else:
        a_prior = 0.5

    return torch.tensor([v_prior, a_prior], dtype=torch.float32)

# ── 3. Load Test Data: The Judge ONLY ──────────────────────────────────────────
video_root = config.VIDEO_FEATURE_ROOT
audio_root = config.AUDIO_FEATURE_ROOT
dataset = SceneFeatureDataset(video_root, audio_root)
label_dict = load_labels(config.LABEL_FILE)
genre_dict = load_genre_labels(config.LABEL_FILE)
binary_label_dict = load_binary_labels(
    config.LABEL_FILE, config.VALENCE_THRESH_FILE, config.AROUSAL_THRESH_FILE
)

val_thresh_df = pd.read_csv(config.VALENCE_THRESH_FILE)
aro_thresh_df = pd.read_csv(config.AROUSAL_THRESH_FILE)
val_row = val_thresh_df[val_thresh_df["movie_name"] == "TheJudge"]
aro_row = aro_thresh_df[aro_thresh_df["movie_name"] == "TheJudge"]
VAL_THRESH = float(val_row["mid_val"].values[0]) if len(val_row) > 0 else 0.41
ARO_THRESH = float(aro_row["mid_val"].values[0]) if len(aro_row) > 0 else 0.57

judge_indices = [i for i, s in enumerate(dataset.samples) if s[0] == "TheJudge"]
print(f"[Judge Test] Found {len(judge_indices)} clips from 'TheJudge' (100% unseen by models).")
judge_dataset = torch.utils.data.Subset(dataset, judge_indices)
judge_loader = DataLoader(judge_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

# ── 4. Load Models ─────────────────────────────────────────────────────────────
TRAIN_POOL_MOVIES = [
    'AboutTime', 'Gifted', 'LadyBird', 'TheBlindSide',
    'TheFaultInOurStars', 'TheSpectacularNow', 'TheTheoryOfEverything', 'Wonder'
]

# Load 8 RG-MAN models (Supports both Decoupled and Joint)
rgman_val_models = []
rgman_aro_models = []
rgman_joint_models = []

for m in TRAIN_POOL_MOVIES:
    p_val = os.path.join(RGMAN_CKPT_DIR, "valence", f"rgman_val_val_{m}.pth")
    p_aro = os.path.join(RGMAN_CKPT_DIR, "arousal", f"rgman_aro_val_{m}.pth")
    p_joint = os.path.join(RGMAN_CKPT_DIR, f"rgman_no_judge_val_{m}.pth")

    if os.path.exists(p_val) and os.path.exists(p_aro):
        net_v = RuleGuided_PerceivedVA_Model(
            video_dim=config.VIDEO_DIM, audio_dim=config.AUDIO_DIM, hidden_dim=config.HIDDEN_DIM,
            genre_dim=config.GENRE_DIM, genre_emb_dim=config.GENRE_EMB_DIM, rule_dim=2, rule_emb_dim=16,
            text_dim=0, task="valence"
        )
        net_v.load_state_dict(torch.load(p_val, map_location=device))
        net_v.to(device)
        net_v.eval()
        rgman_val_models.append(net_v)

        net_a = RuleGuided_PerceivedVA_Model(
            video_dim=config.VIDEO_DIM, audio_dim=config.AUDIO_DIM, hidden_dim=config.HIDDEN_DIM,
            genre_dim=config.GENRE_DIM, genre_emb_dim=config.GENRE_EMB_DIM, rule_dim=2, rule_emb_dim=16,
            text_dim=0, task="arousal"
        )
        net_a.load_state_dict(torch.load(p_aro, map_location=device))
        net_a.to(device)
        net_a.eval()
        rgman_aro_models.append(net_a)
    elif os.path.exists(p_joint):
        net_j = RuleGuided_PerceivedVA_Model(
            video_dim=config.VIDEO_DIM, audio_dim=config.AUDIO_DIM, hidden_dim=config.HIDDEN_DIM,
            genre_dim=config.GENRE_DIM, genre_emb_dim=config.GENRE_EMB_DIM, rule_dim=2, rule_emb_dim=16,
            text_dim=0, task="both"
        )
        net_j.load_state_dict(torch.load(p_joint, map_location=device))
        net_j.to(device)
        net_j.eval()
        rgman_joint_models.append(net_j)

is_rgman_decoupled = (len(rgman_val_models) == len(TRAIN_POOL_MOVIES) and len(rgman_aro_models) == len(TRAIN_POOL_MOVIES)) or (len(rgman_val_models) > 0 and len(rgman_joint_models) == 0)

# Load 8 Decoupled Valence models
val_models = []
for m in TRAIN_POOL_MOVIES:
    p = os.path.join(VAL_CKPT_DIR, f"decoupled_val_val_{m}.pth")
    if os.path.exists(p):
        net = PerceivedVA_Model(
            video_dim=config.VIDEO_DIM, audio_dim=config.AUDIO_DIM, hidden_dim=config.HIDDEN_DIM,
            genre_dim=config.GENRE_DIM, genre_emb_dim=config.GENRE_EMB_DIM, text_dim=0
        )
        net.load_state_dict(torch.load(p, map_location=device))
        net.to(device)
        net.eval()
        val_models.append(net)

# Load 8 Decoupled Arousal models
aro_models = []
for m in TRAIN_POOL_MOVIES:
    p = os.path.join(ARO_CKPT_DIR, f"decoupled_aro_val_{m}.pth")
    if os.path.exists(p):
        net = PerceivedVA_Model(
            video_dim=config.VIDEO_DIM, audio_dim=config.AUDIO_DIM, hidden_dim=config.HIDDEN_DIM,
            genre_dim=config.GENRE_DIM, genre_emb_dim=config.GENRE_EMB_DIM, text_dim=0
        )
        net.load_state_dict(torch.load(p, map_location=device))
        net.to(device)
        net.eval()
        aro_models.append(net)

if is_rgman_decoupled:
    print(f"[Judge Test] Loaded {len(rgman_val_models)}/8 Decoupled RG-MAN Valence checkpoints")
    print(f"[Judge Test] Loaded {len(rgman_aro_models)}/8 Decoupled RG-MAN Arousal checkpoints")
else:
    print(f"[Judge Test] Loaded {len(rgman_joint_models)}/8 Joint RG-MAN checkpoints")
print(f"[Judge Test] Loaded {len(val_models)}/8 Decoupled Valence checkpoints")
print(f"[Judge Test] Loaded {len(aro_models)}/8 Decoupled Arousal checkpoints")

has_rgman = (len(rgman_val_models) > 0 and len(rgman_aro_models) > 0) or (len(rgman_joint_models) > 0)
if not has_rgman and len(val_models) == 0:
    print("\n[ERROR] No trained checkpoints found! Please run the training scripts first:")
    print("  1. python train_8fold_no_judge_rgman.py")
    print("  2. python train_8fold_no_judge_decoupled.py")
    sys.exit(1)

# ── 5. Run Inference on The Judge ──────────────────────────────────────────────
print("\n[Judge Test] Running zero-leakage ensemble inference on all 149 clips of The Judge ...")

clip_rgman_v, clip_rgman_a = {}, {}
clip_dec_v, clip_dec_a = {}, {}

with torch.no_grad():
    for video, audio, mask, clip_names in judge_loader:
        video = video.float().to(device)
        audio = audio.float().to(device)
        mask  = mask.to(device)
        genres = torch.stack([genre_dict.get(c) for c in clip_names]).to(device)
        rules  = torch.stack([get_rule_prior(c) for c in clip_names]).to(device)

        # 1. RG-MAN predictions
        if is_rgman_decoupled:
            for m_v in rgman_val_models:
                pv, _, _ = m_v(video, audio, mask=mask, genre=genres, rule_prior=rules)
                for i, c in enumerate(clip_names):
                    if c not in clip_rgman_v:
                        clip_rgman_v[c] = []
                    clip_rgman_v[c].append(pv[i].item())
            for m_a in rgman_aro_models:
                _, pa, _ = m_a(video, audio, mask=mask, genre=genres, rule_prior=rules)
                for i, c in enumerate(clip_names):
                    if c not in clip_rgman_a:
                        clip_rgman_a[c] = []
                    clip_rgman_a[c].append(pa[i].item())
        else:
            for m in rgman_joint_models:
                pv, pa, _ = m(video, audio, mask=mask, genre=genres, rule_prior=rules)
                for i, c in enumerate(clip_names):
                    if c not in clip_rgman_v:
                        clip_rgman_v[c] = []
                        clip_rgman_a[c] = []
                    clip_rgman_v[c].append(pv[i].item())
                    clip_rgman_a[c].append(pa[i].item())

        # 2. Decoupled predictions
        for m_v in val_models:
            pv, _, _ = m_v(video, audio, mask=mask, genre=genres)
            for i, c in enumerate(clip_names):
                if c not in clip_dec_v:
                    clip_dec_v[c] = []
                clip_dec_v[c].append(pv[i].item())

        for m_a in aro_models:
            _, pa, _ = m_a(video, audio, mask=mask, genre=genres)
            for i, c in enumerate(clip_names):
                if c not in clip_dec_a:
                    clip_dec_a[c] = []
                clip_dec_a[c].append(pa[i].item())

# ── 6. Aggregate & Score ───────────────────────────────────────────────────────
rows = []
for idx in judge_indices:
    c = dataset.samples[idx]
    clip_name = os.path.basename(c[1])

    # RG-MAN ensemble
    rg_v = np.mean(clip_rgman_v[clip_name]) if clip_name in clip_rgman_v else None
    rg_a = np.mean(clip_rgman_a[clip_name]) if clip_name in clip_rgman_a else None

    # Decoupled ensemble
    dc_v = np.mean(clip_dec_v[clip_name]) if clip_name in clip_dec_v else None
    dc_a = np.mean(clip_dec_a[clip_name]) if clip_name in clip_dec_a else None

    # Master Blend (Combined)
    blend_v = float(np.mean([x for x in [rg_v, dc_v] if x is not None]))
    blend_a = float(np.mean([x for x in [rg_a, dc_a] if x is not None]))

    # Ground-truth
    gt_v, gt_a = label_dict.get(clip_name, (None, None))
    gt_bin = binary_label_dict.get(clip_name, (None, None))

    # Binary predictions & confidence for all configurations
    def get_bin_and_conf(val, aro):
        if val is None or aro is None:
            return None, None, None, None, False
        bv = bin_value_binary(val, VAL_THRESH)
        ba = bin_value_binary(aro, ARO_THRESH)
        cv = abs(2.0 * val - 1.0)
        ca = abs(2.0 * aro - 1.0)
        hc = (cv >= 0.65) and (ca >= 0.65)
        return bv, ba, cv, ca, hc

    rg_bin_v, rg_bin_a, rg_cv, rg_ca, rg_hc = get_bin_and_conf(rg_v, rg_a)
    dc_bin_v, dc_bin_a, dc_cv, dc_ca, dc_hc = get_bin_and_conf(dc_v, dc_a)
    bl_bin_v, bl_bin_a, bl_cv, bl_ca, bl_hc = get_bin_and_conf(blend_v, blend_a)

    rows.append({
        "clip": clip_name,
        "rgman_val": round(rg_v, 4) if rg_v is not None else None,
        "rgman_aro": round(rg_a, 4) if rg_a is not None else None,
        "rgman_bin_val": rg_bin_v,
        "rgman_bin_aro": rg_bin_a,
        "decoupled_val": round(dc_v, 4) if dc_v is not None else None,
        "decoupled_aro": round(dc_a, 4) if dc_a is not None else None,
        "decoupled_bin_val": dc_bin_v,
        "decoupled_bin_aro": dc_bin_a,
        "blend_val": round(blend_v, 4) if blend_v is not None else None,
        "blend_aro": round(blend_a, 4) if blend_a is not None else None,
        "blend_bin_val": bl_bin_v,
        "blend_bin_aro": bl_bin_a,
        "blend_conf_val": round(bl_cv, 4) if bl_cv is not None else None,
        "blend_conf_aro": round(bl_ca, 4) if bl_ca is not None else None,
        "blend_high_conf": bl_hc,
        "gt_val": gt_v,
        "gt_aro": gt_a,
        "gt_bin_val": gt_bin[0],
        "gt_bin_aro": gt_bin[1],
        "rgman_correct_joint": (rg_bin_v == gt_bin[0] and rg_bin_a == gt_bin[1]) if rg_bin_v is not None else False,
        "decoupled_correct_joint": (dc_bin_v == gt_bin[0] and dc_bin_a == gt_bin[1]) if dc_bin_v is not None else False,
        "blend_correct_joint": (bl_bin_v == gt_bin[0] and bl_bin_a == gt_bin[1]) if bl_bin_v is not None else False,
    })

df_res = pd.DataFrame(rows)
df_res.to_csv(os.path.join(OUTPUT_DIR, "the_judge_per_clip_ensemble_predictions.csv"), index=False)

# Helper function to compute and print metrics for any setup
def evaluate_config(name, pred_v_col, pred_a_col, bin_v_col, bin_a_col, joint_corr_col):
    valid_mask = df_res[pred_v_col].notnull() & df_res[pred_a_col].notnull()
    sub_df = df_res[valid_mask]
    if len(sub_df) == 0:
        return None

    pv = sub_df[pred_v_col].values
    pa = sub_df[pred_a_col].values
    gv = sub_df["gt_val"].values
    ga = sub_df["gt_aro"].values

    _, _, ccc_v = compute_metrics(pv, gv)
    _, _, ccc_a = compute_metrics(pa, ga)
    acc_v, _, _, f1_v = compute_classification_metrics(sub_df["gt_bin_val"].values, sub_df[bin_v_col].values)
    acc_a, _, _, f1_a = compute_classification_metrics(sub_df["gt_bin_aro"].values, sub_df[bin_a_col].values)
    joint_acc = np.mean(sub_df[joint_corr_col].values) * 100.0

    print(f"\n[{name.upper()}] Evaluated on {len(sub_df)} clips:")
    print(f"  Valence -> CCC: {ccc_v:.4f}  |  Acc: {acc_v*100:.2f}%  |  Macro-F1: {f1_v*100:.2f}%")
    print(f"  Arousal -> CCC: {ccc_a:.4f}  |  Acc: {acc_a*100:.2f}%  |  Macro-F1: {f1_a*100:.2f}%")
    print(f"  Joint Clip Accuracy : {joint_acc:.2f}%")

    return {
        "Configuration": name,
        "Clips_Evaluated": len(sub_df),
        "Valence_CCC": round(ccc_v, 4),
        "Valence_Acc": round(acc_v * 100, 2),
        "Valence_F1": round(f1_v * 100, 2),
        "Arousal_CCC": round(ccc_a, 4),
        "Arousal_Acc": round(acc_a * 100, 2),
        "Arousal_F1": round(f1_a * 100, 2),
        "Joint_Clip_Acc": round(joint_acc, 2)
    }

print("\n" + "=" * 80)
print("ZERO-LEAKAGE 8-FOLD EVALUATION BREAKDOWN ON 'THE JUDGE' (100% UNSEEN)")
print("=" * 80)

summary_rows = []
if has_rgman:
    rgman_label = "8-Fold Decoupled RG-MAN Ensemble Alone" if is_rgman_decoupled else "8-Fold RG-MAN Ensemble Alone"
    res_rg = evaluate_config(rgman_label, "rgman_val", "rgman_aro", "rgman_bin_val", "rgman_bin_aro", "rgman_correct_joint")
    if res_rg: summary_rows.append(res_rg)

if len(val_models) > 0 and len(aro_models) > 0:
    res_dc = evaluate_config("8-Fold Decoupled MGN Ensemble Alone", "decoupled_val", "decoupled_aro", "decoupled_bin_val", "decoupled_bin_aro", "decoupled_correct_joint")
    if res_dc: summary_rows.append(res_dc)

if has_rgman and (len(val_models) > 0 or len(aro_models) > 0):
    res_bl = evaluate_config("Combined Master Ensemble (RG-MAN + Decoupled)", "blend_val", "blend_aro", "blend_bin_val", "blend_bin_aro", "blend_correct_joint")
    if res_bl: summary_rows.append(res_bl)

print("=" * 80)

if summary_rows:
    df_sum = pd.DataFrame(summary_rows)
    sum_path = os.path.join(OUTPUT_DIR, "the_judge_8fold_ensemble_summary.csv")
    df_sum.to_csv(sum_path, index=False)
    print(f"\n[SAVED] Comparison summary saved to: {sum_path}")
    print(f"[SAVED] Detailed per-clip predictions saved to: {OUTPUT_DIR}/the_judge_per_clip_ensemble_predictions.csv")

