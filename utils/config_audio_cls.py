# config_audio_cls.py
from dataclasses import dataclass

@dataclass
class AudioClsConfig:
    # HF dataset (easy options)
    # "speech_commands"  -> keyword/command classification (good starter)
    # "esc50"            -> environmental sound classification
    dataset_name =  "mskov/ESC50"  
    dataset_config = None
    split_val   = "test"     # superb/ks provides validation
    num_classes: int = 50
    split_train = "train"
    use_hf_streaming: bool = True
    verification_no_checks: bool = True
    stream_take: int | None = None     # e.g., 20000 to cap streamed examples

    # Columns
    audio_field = "audio"
    label_field = "target" 

    # Audio prep
    sample_rate: int = 44100
    max_secs    = 5.0             # 1s clips fit well
    random_offset_crop: bool = True
    use_augs: bool = True

    # Byte tokenizer budget
    src_max_len: int = 1024            # 1s @16kHz int16 -> 32kB; we cap at 1024 tokens
    pad_token: int = 255
    remap_255_to: int = 254            # avoid pad collision

    # Runtime
    batch_size: int = 128
    val_batch_size: int = 256
    num_workers: int = 0
    pin_memory: bool = False
    seed: int = 42
    amp: bool = True
    clip_grad_norm: float = 1.0

    # Encoder checkpoint & config discovery
    init_checkpoint: str = "checkpoints/V1/model.pth"
    encoder_config_path: str = "utils/config.py"  # your existing encoder config file
    encoder_config_class: str | None = None

    # Optim
    lr_head: float = 2e-3
    weight_decay: float = 0.01
    epochs: int = 20
    log_every_n: int = 200

    # Logging / saves
    save_dir: str = "checkpoints/audio"
    save_head_as: str = "audio_cls_head.pth"
    save_encoder_as: str | None = None
    csv_log_path: str = "logs/audio_cls.csv"
    best_by: str = "val_acc"  # or "-val_loss"
