"""
Training Script: 8-Fold Sub-Ensemble for Decoupled Task-Specific Attention Net
=============================================================================
STRICT ZERO-LEAKAGE EXPERIMENTAL SETUP:
  - Test Movie: TheJudge (149 clips) — 100% EXCLUDED, NEVER SEEN, NEVER TRAINED ON.
  - Training Pool: Exactly 8 movies (747 clips):
      AboutTime, Gifted, LadyBird, TheBlindSide, TheFaultInOurStars,
      TheSpectacularNow, TheTheoryOfEverything, Wonder.
  - Protocol: 8-Fold Cross-Validation among the 8 movies:
      In each fold k in 1..8:
        • 7 movies are used for training
        • 1 movie (from within the 8) is used for validation (early stopping)
        • Dedicated Valence checkpoints -> Thilok_Outputs/saved_models_8fold_no_judge/valence/
        • Dedicated Arousal checkpoints -> Thilok_Outputs/saved_models_8fold_no_judge/arousal/
  - Generates 8 distinct Valence and 8 distinct Arousal checkpoints with 0% knowledge of The Judge.

Usage:
  CUDA_VISIBLE_DEVICES=1 nohup python -u train_8fold_no_judge_decoupled.py >> Thilok_Outputs/train_8fold_decoupled_log.txt 2>&1 &
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BTP_DIR = SCRIPT_DIR
ROOT_DIR = os.path.dirname(BTP_DIR)
sys.path.insert(0, BTP_DIR)

from Thilok_perceive_emo_pred_model import config
from Thilok_perceive_emo_pred_model.dataset import SceneFeatureDataset, collate_fn
from Thilok_perceive_emo_pred_model.model import PerceivedVA_Model
from Thilok_perceive_emo_pred_model.loss import compute_loss_task
from Thilok_perceive_emo_pred_model.utils import load_labels, load_genre_labels

# ── 1. Directories & Checkpoints ───────────────────────────────────────────────
SAVE_DIR_VAL = os.path.join(BTP_DIR, "Thilok_Outputs", "saved_models_8fold_no_judge", "valence")
SAVE_DIR_ARO = os.path.join(BTP_DIR, "Thilok_Outputs", "saved_models_8fold_no_judge", "arousal")
os.makedirs(SAVE_DIR_VAL, exist_ok=True)
os.makedirs(SAVE_DIR_ARO, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[8-Fold Decoupled] Using device: {device}")
print(f"[8-Fold Decoupled] Valence Checkpoints -> {SAVE_DIR_VAL}")
print(f"[8-Fold Decoupled] Arousal Checkpoints -> {SAVE_DIR_ARO}")

# ── 2. Dataset & Explicit Exclusion of The Judge ───────────────────────────────
video_root = config.VIDEO_FEATURE_ROOT
audio_root = config.AUDIO_FEATURE_ROOT
dataset = SceneFeatureDataset(video_root, audio_root)
label_dict = load_labels(config.LABEL_FILE)
genre_dict = load_genre_labels(config.LABEL_FILE)

HELD_OUT_TARGET = "TheJudge"
all_available = sorted(set(s[0] for s in dataset.samples))
TRAIN_POOL_MOVIES = [m for m in all_available if m != HELD_OUT_TARGET]

print("=" * 70)
print(f"STRICT CHECK: Held-out Test Movie (0% seen): {HELD_OUT_TARGET}")
print(f"8 Training Pool Movies: {TRAIN_POOL_MOVIES}")
print("=" * 70)

pool_indices = [i for i, s in enumerate(dataset.samples) if s[0] in TRAIN_POOL_MOVIES]
print(f"Total clips in 8-movie training pool: {len(pool_indices)} (Excluded all clips from {HELD_OUT_TARGET})")

# ── 3. Helper: Train Specialist Model ──────────────────────────────────────────
EPOCHS = 20
BATCH_SIZE = config.BATCH_SIZE
LEARNING_RATE = config.LEARNING_RATE

def train_specialist(task_name, train_loader, val_loader, save_path):
    """
    Trains a dedicated specialist model (either 'valence' or 'arousal').
    """
    model = PerceivedVA_Model(
        video_dim=config.VIDEO_DIM,
        audio_dim=config.AUDIO_DIM,
        hidden_dim=config.HIDDEN_DIM,
        genre_dim=config.GENRE_DIM,
        genre_emb_dim=config.GENRE_EMB_DIM,
        text_dim=0
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    best_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        count = 0

        for video, audio, mask, clip_names in train_loader:
            video  = video.float().to(device)
            audio  = audio.float().to(device)
            mask   = mask.to(device)
            genres = torch.stack([genre_dict.get(c) for c in clip_names]).to(device)

            pred_v, pred_a, delta = model(video, audio, mask, genres)

            if task_name == "valence":
                gt = torch.tensor([label_dict[c][0] for c in clip_names], dtype=torch.float32).to(device)
                pred = pred_v
            else:
                gt = torch.tensor([label_dict[c][1] for c in clip_names], dtype=torch.float32).to(device)
                pred = pred_a

            loss = compute_loss_task(pred, gt, delta, alpha=0.5)
            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            count += 1

        train_loss /= max(1, count)

        # Validation
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for video, audio, mask, clip_names in val_loader:
                video  = video.float().to(device)
                audio  = audio.float().to(device)
                mask   = mask.to(device)
                genres = torch.stack([genre_dict.get(c) for c in clip_names]).to(device)

                pred_v, pred_a, delta = model(video, audio, mask, genres)

                if task_name == "valence":
                    gt = torch.tensor([label_dict[c][0] for c in clip_names], dtype=torch.float32).to(device)
                    pred = pred_v
                else:
                    gt = torch.tensor([label_dict[c][1] for c in clip_names], dtype=torch.float32).to(device)
                    pred = pred_a

                loss = compute_loss_task(pred, gt, delta, alpha=0.5)
                if not torch.isnan(loss):
                    val_loss += loss.item()
                    val_count += 1

        val_loss /= max(1, val_count)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            print(f"    [{task_name.upper()}] Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} {'*' if val_loss == best_loss else ''}")

    return best_loss

# ── 4. Main 8-Fold Training Loop ───────────────────────────────────────────────
for fold_idx, val_movie in enumerate(TRAIN_POOL_MOVIES, 1):
    print(f"\n{'='*70}")
    print(f"[8-Fold Decoupled] FOLD {fold_idx}/8: Validating on '{val_movie}' | Training on other 7 movies")
    print(f"{'='*70}")

    train_indices = [i for i in pool_indices if dataset.samples[i][0] != val_movie]
    val_indices   = [i for i in pool_indices if dataset.samples[i][0] == val_movie]

    assert all(dataset.samples[i][0] != HELD_OUT_TARGET for i in train_indices), "LEAKAGE: TheJudge in train_indices!"
    assert all(dataset.samples[i][0] != HELD_OUT_TARGET for i in val_indices), "LEAKAGE: TheJudge in val_indices!"

    train_sub = torch.utils.data.Subset(dataset, train_indices)
    val_sub   = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    val_ckpt_path = os.path.join(SAVE_DIR_VAL, f"decoupled_val_val_{val_movie}.pth")
    aro_ckpt_path = os.path.join(SAVE_DIR_ARO, f"decoupled_aro_val_{val_movie}.pth")

    print(f"  Training Dedicated Valence Specialist for Fold {fold_idx} ...")
    best_val_loss = train_specialist("valence", train_loader, val_loader, val_ckpt_path)
    print(f"  [SAVED] Valence Checkpoint -> {val_ckpt_path} (Best Loss: {best_val_loss:.4f})")

    print(f"  Training Dedicated Arousal Specialist for Fold {fold_idx} ...")
    best_aro_loss = train_specialist("arousal", train_loader, val_loader, aro_ckpt_path)
    print(f"  [SAVED] Arousal Checkpoint -> {aro_ckpt_path} (Best Loss: {best_aro_loss:.4f})")

print("\n" + "=" * 70)
print(f"SUCCESS: All 8 Valence & 8 Arousal checkpoints generated without seeing '{HELD_OUT_TARGET}'!")
print(f"Valence Checkpoints: {SAVE_DIR_VAL}/")
print(f"Arousal Checkpoints: {SAVE_DIR_ARO}/")
print("=" * 70)

