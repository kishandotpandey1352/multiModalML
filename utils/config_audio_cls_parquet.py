# config_audio_cls_parquet.py
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class AudioClsParquetConfig:
    """
    ESC-50 (Parquet) streaming config for audio classification fine-tuning.
    Works with data_modules/audio_hf.py that expects parquet_repo + parquet_glob.
    """

    # --- Parquet source (no dataset scripts needed) ---
    parquet_repo = None   # HF dataset repo with Parquet shards
    parquet_glob = None # filled in __post_init__
    preferred_repos = ("ashraq/esc50", "mskov/ESC50")    # Split names used by your trainer
    split_train = "train"
    split_val = "test"

    # Column names / classes
    audio_field: str = "audio"          # not used in parquet path mode; kept for compatibility
    label_field: str = "target"         # ESC-50 mirrors often expose 'target' (int)
    num_classes: int = 50

    # Audio → bytes → ids
    sample_rate: int = 16000                # ESC-50 native SR
    max_secs: float  = 0.25
    src_max_len: int = 192
    pad_token: int   = 255
    remap_255_to: int = 254

    # Runtime (streaming-friendly)
    batch_size: int     = 4
    val_batch_size: int = 4
    num_workers: int    = 0             # streaming → keep ≤ 1
    pin_memory: bool    = False
    stream_take = 200   # e.g., 500 for smoke tests
    val_stream_take = None   
    # Repro / AMP / clipping
    seed: int            = 42
    amp: bool            = True
    clip_grad_norm: float = 1.0

    # Encoder / head / logging
    init_checkpoint: str     = "checkpoints/V1/model.pth"
    encoder_config_path: str = "utils/config.py"
    encoder_config_class: Optional[str] = None

    lr_head: float     = 2e-3
    weight_decay: float = 0.01
    epochs: int         = 20
    log_every_n: int    = 100

    save_dir: str       = "checkpoints/esc50"
    save_head_as: str   = "esc50_head.pth"
    save_encoder_as: Optional[str] = None
    csv_log_path: str   = "logs/esc50_audio.csv"
    best_by: str        = "val_acc"
     # turn OFF audio augments for stability
    use_augs: bool       = False
    random_offset_crop: bool = True

    def __post_init__(self):
        # Wildcards so you don’t need to know hash suffixes on shard names
        self.parquet_glob = {
            "train": "train-*.parquet",
            "test":  "test-*.parquet",
            # If your repo has 'validation-*.parquet' and you prefer that:
            # "validation": "validation-*.parquet",
        }
