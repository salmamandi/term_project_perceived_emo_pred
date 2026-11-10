import os

_CURR_DIR = os.path.dirname(os.path.abspath(__file__))       # btp_Thilok/Thilok_perceive_emo_pred_model
_BTP_DIR  = os.path.dirname(_CURR_DIR)                       # btp_Thilok
_ROOT_DIR = os.path.dirname(_BTP_DIR)                        # EmoTx-CVPR2023

def _resolve(rel_path):
    if not rel_path:
        return rel_path
    if os.path.exists(rel_path):
        return rel_path
    p_btp = os.path.join(_BTP_DIR, rel_path)
    if os.path.exists(p_btp):
        return p_btp
    p_root = os.path.join(_ROOT_DIR, rel_path)
    if os.path.exists(p_root):
        return p_root
    return rel_path

# Feature directories
VIDEO_FEATURE_ROOT = _resolve("checkpoints_scene_srt/mvit_v1_scene_features")
AUDIO_FEATURE_ROOT = _resolve("checkpoints_scene_srt/wav2vec2_scene_features")
SRT_FEATURE_ROOT = _resolve("checkpoints_scene_srt/srt_feats/concat/finetuned_t10")
MODEL_SAVE_PATH = _resolve("Thilok_Outputs/saved_models_lightweight")
#MODEL_SAVE_PATH="Thilok_Outputs/saved_models"

# Label file
LABEL_FILE = _resolve("Thilok_perceive_emo_pred_model/genre_emotion.csv")
VALENCE_THRESH_FILE = _resolve("movie_cluster2_valence_info.csv")
AROUSAL_THRESH_FILE = _resolve("movie_cluster2_arousal_info.csv")

# Feature dimensions
VIDEO_DIM = 768
AUDIO_DIM = 1024
TEXT_DIM = 768
GENRE_DIM = 9
GENRE_EMB_DIM = 32

# Model parameters
HIDDEN_DIM = 64 # changing from 256 to 64

# Training parameters
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 20