import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceivedVA_Model(nn.Module):

    # adding task parameter to learn task specific model with genre injection and text/subtitle integration
    def __init__(self, video_dim, audio_dim, hidden_dim=256, genre_dim=9, genre_emb_dim=32, text_dim=768):

        super().__init__()
        self.genre_dim = genre_dim
        self.genre_emb_dim = genre_emb_dim
        self.text_dim = text_dim
        
        # Projection
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)

        # Text/Subtitle projection and cross-modal attention
        if text_dim > 0:
            self.text_proj = nn.Linear(text_dim, hidden_dim)
            self.text_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.cross_attn_vt = nn.MultiheadAttention(hidden_dim * 2, num_heads=4, batch_first=True)
            self.gate_t = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        else:
            self.text_proj = None

        # Temporal Context Encoder
        self.video_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.audio_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)

        fusion_dim = hidden_dim * 2

        # Cross modal attention
        self.cross_attn = nn.MultiheadAttention(fusion_dim, num_heads=4, batch_first=True)

        # Modality gating
        self.gate_v = nn.Linear(fusion_dim, fusion_dim)
        self.gate_a = nn.Linear(fusion_dim, fusion_dim)

        # Attention pooling
        self.scene_attention= nn.Linear(fusion_dim, 1)

        # Genre projection
        if genre_dim > 0:
            self.genre_proj = nn.Sequential(
                nn.Linear(genre_dim, genre_emb_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            reg_in_dim = fusion_dim + genre_emb_dim
        else:
            self.genre_proj = None
            reg_in_dim = fusion_dim

        # VA regression head
        self.fc1 = nn.Linear(reg_in_dim, 128)
        
        self.fc2 = nn.Linear(128, 2) #make the dimension of output 1 when modeling for single output

    def forward(self, video, audio, mask=None, genre=None, text=None):

        # Projection
        v = self.video_proj(video)
        a = self.audio_proj(audio)

        # Temporal encoder
        v, _ = self.video_lstm(v)
        a, _ = self.audio_lstm(a)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask == 0)

        # Cross-modal attention
        v_attn, _ = self.cross_attn(v, a, a, key_padding_mask=key_padding_mask)
        a_attn, _ = self.cross_attn(a, v, v, key_padding_mask=key_padding_mask)

        # Modality gating
        g_v = torch.sigmoid(self.gate_v(v_attn))
        g_a = torch.sigmoid(self.gate_a(a_attn))

        fused = g_v * v_attn + g_a * a_attn

        # Subtitle / text integration
        if text is not None and self.text_proj is not None:
            t = self.text_proj(text)
            t, _ = self.text_lstm(t)
            vt_attn, _ = self.cross_attn_vt(v, t, t, key_padding_mask=key_padding_mask)
            g_t = torch.sigmoid(self.gate_t(vt_attn))
            fused = fused + g_t * vt_attn

        # emotion transition
        delta = fused[:, 1:, :] - fused[:, :-1, :]

        # Attention pooling
        scores = self.scene_attention(fused).squeeze(-1)   # (B,T)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = torch.softmax(scores, dim=1).unsqueeze(-1)

        clip_repr = torch.sum(weights * fused, dim=1)

        # Genre injection
        if self.genre_proj is not None:
            if genre is None:
                genre = torch.zeros((video.size(0), self.genre_dim), device=video.device)
            g_emb = self.genre_proj(genre)
            clip_repr = torch.cat([clip_repr, g_emb], dim=-1)

        # Regression head
        x = F.relu(self.fc1(clip_repr))
        
        # use the uncomment part if you wish to learn both valence and arousal together
        
        va = self.fc2(x)

        valence = torch.sigmoid(va[:,0])
        arousal = torch.sigmoid(va[:,1])

        return valence, arousal, delta
        
        # code to learn either valence or arousal
        # out = self.fc2(x).squeeze(1)
        # pred = torch.sigmoid(out)

        # return pred, delta