"""
Rule-Guided Multimodal Attention Network (RG-MAN)
==================================================
Combines:
  1. The Rule-Based Model: Heuristic modality informativeness assessments from
     useful_nonuseful_names_manual_correct.json (speech presence, face visibility).
  2. The Attention-Based Model (MGN): Bidirectional Cross-Modal Attention,
     Modality Gating, Scene Attention Pooling, RoBERTa Subtitles, and Genre Priors.

Key Neuro-Symbolic Innovation:
  Instead of hard binary filtering (which discards partial signals in pure rule models),
  the rule informativeness score is embedded as an inductive prior directly into the
  cross-modal gating layers:
      g_v = sigmoid(W_v * h_va + W_rv * e_rule)
      g_a = sigmoid(W_a * h_av + W_ra * e_rule)

  - If rules classify a modality as uninformative, the network receives a strong
    negative prior to suppress noise.
  - If attention detects subtle salient cues, the deep network can softly modulate
    the prior, avoiding the rigidity of pure heuristic rules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class RuleGuided_PerceivedVA_Model(nn.Module):
    def __init__(
        self,
        video_dim=768,
        audio_dim=1024,
        hidden_dim=256,
        genre_dim=9,
        genre_emb_dim=32,
        rule_dim=2,          # [rule_video_useful, rule_audio_useful]
        rule_emb_dim=16,     # Rule prior embedding dimension
        text_dim=768,
        task="both"          # 'valence', 'arousal', or 'both'
    ):
        super().__init__()
        self.task = task
        self.genre_dim = genre_dim
        self.rule_dim = rule_dim
        self.rule_emb_dim = rule_emb_dim
        self.text_dim = text_dim
        fusion_dim = hidden_dim * 2

        # ── 1. Feature Projections ────────────────────────────────────────────
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)

        # ── 2. Temporal Context Encoders (BiLSTM) ─────────────────────────────
        self.video_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.audio_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)

        # ── 3. Rule Prior Projection (Symbolic -> Continuous Prior) ───────────
        self.rule_proj = nn.Sequential(
            nn.Linear(rule_dim, rule_emb_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # Gate bias projections conditioned on rule embedding
        self.rule_gate_v = nn.Linear(rule_emb_dim, fusion_dim)
        self.rule_gate_a = nn.Linear(rule_emb_dim, fusion_dim)

        # ── 4. Cross-Modal Attention ──────────────────────────────────────────
        self.cross_attn = nn.MultiheadAttention(fusion_dim, num_heads=4, batch_first=True)

        # ── 5. Rule-Guided Modality Gating Layers ─────────────────────────────
        self.gate_v = nn.Linear(fusion_dim, fusion_dim)
        self.gate_a = nn.Linear(fusion_dim, fusion_dim)

        # ── 6. Subtitle / Dialogue Cross-Attention (Optional) ─────────────────
        if text_dim > 0:
            self.text_proj = nn.Linear(text_dim, hidden_dim)
            self.text_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.cross_attn_vt = nn.MultiheadAttention(fusion_dim, num_heads=4, batch_first=True)
            self.gate_t = nn.Linear(fusion_dim, fusion_dim)
        else:
            self.text_proj = None

        # ── 7. Scene Attention Pooling (Property 1: Dominant Emotion) ─────────
        self.scene_attention = nn.Linear(fusion_dim, 1)

        # ── 8. Genre Prior Projection ─────────────────────────────────────────
        if genre_dim > 0:
            self.genre_proj = nn.Sequential(
                nn.Linear(genre_dim, genre_emb_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            reg_in_dim = fusion_dim + genre_emb_dim + rule_emb_dim
        else:
            self.genre_proj = None
            reg_in_dim = fusion_dim + rule_emb_dim

        # ── 9. Prediction Regression Heads ────────────────────────────────────
        self.fc1 = nn.Linear(reg_in_dim, 128)
        self.dropout = nn.Dropout(0.2)

        if task == "valence" or task == "arousal":
            self.fc_out = nn.Linear(128, 1)
        else:
            self.fc_out = nn.Linear(128, 2)

    def forward(self, video, audio, mask=None, genre=None, text=None, rule_prior=None):
        B, T, _ = video.shape
        device = video.device

        # 1. Linear Projection & Temporal Encoding
        v = self.video_proj(video)
        a = self.audio_proj(audio)

        v, _ = self.video_lstm(v)
        a, _ = self.audio_lstm(a)

        key_padding_mask = (mask == 0) if mask is not None else None

        # 2. Cross-Modal Attention (Video queries Audio, Audio queries Video)
        v_attn, _ = self.cross_attn(v, a, a, key_padding_mask=key_padding_mask)
        a_attn, _ = self.cross_attn(a, v, v, key_padding_mask=key_padding_mask)

        # 3. Rule Prior Embedding Injection into Modality Gates
        if rule_prior is None:
            rule_prior = torch.ones((B, self.rule_dim), device=device) * 0.5  # Neutral default

        r_emb = self.rule_proj(rule_prior)                       # (B, rule_emb_dim)
        r_bias_v = self.rule_gate_v(r_emb).unsqueeze(1)          # (B, 1, fusion_dim)
        r_bias_a = self.rule_gate_a(r_emb).unsqueeze(1)          # (B, 1, fusion_dim)

        # Rule-conditioned continuous gating:
        g_v = torch.sigmoid(self.gate_v(v_attn) + r_bias_v)       # (B, T, fusion_dim)
        g_a = torch.sigmoid(self.gate_a(a_attn) + r_bias_a)       # (B, T, fusion_dim)

        fused = g_v * v_attn + g_a * a_attn                       # (B, T, fusion_dim)

        # 4. Optional Subtitle Dialogue Fusion
        if text is not None and self.text_proj is not None:
            t = self.text_proj(text)
            t, _ = self.text_lstm(t)
            vt_attn, _ = self.cross_attn_vt(v, t, t, key_padding_mask=key_padding_mask)
            g_t = torch.sigmoid(self.gate_t(vt_attn))
            fused = fused + g_t * vt_attn

        # 5. Temporal Transition Vector (Property 2: Transitions)
        delta = fused[:, 1:, :] - fused[:, :-1, :]

        # 6. Scene Attention Pooling (Property 1: Dominant Emotion)
        scores = self.scene_attention(fused).squeeze(-1)         # (B, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)     # (B, T, 1)
        clip_repr = torch.sum(weights * fused, dim=1)            # (B, fusion_dim)

        # 7. Semantic Prior Concatenation (Genre + Rule Residual)
        aux_features = [clip_repr, r_emb]
        if self.genre_proj is not None:
            if genre is None:
                genre = torch.zeros((B, self.genre_dim), device=device)
            g_emb = self.genre_proj(genre)
            aux_features.append(g_emb)

        combined_repr = torch.cat(aux_features, dim=-1)

        # 8. Prediction Head
        feat = self.dropout(F.relu(self.fc1(combined_repr)))
        raw_out = self.fc_out(feat)

        # Strictly align output activations to [0, 1] range using Sigmoid
        if self.task == "valence":
            pred_v = torch.sigmoid(raw_out[:, 0])
            pred_a = pred_v  # fallback placeholder
        elif self.task == "arousal":
            pred_a = torch.sigmoid(raw_out[:, 0])
            pred_v = pred_a  # fallback placeholder
        else:
            pred_v = torch.sigmoid(raw_out[:, 0])
            pred_a = torch.sigmoid(raw_out[:, 1])

        return pred_v, pred_a, delta

