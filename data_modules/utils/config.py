
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Data
    task: str = "agnews"
    use_hf: bool = True
    hf_val_split: float = 0.05
    max_len: int = 1024
    pad_token: int = 255
    num_classes: int = 4

    # Checkpoints
    checkpoints_dir: str = "checkpoints"
    init_checkpoint: str = "checkpoints/V1/model.pth"
    save_encoder_as: str = "shared_encoder_finetuned.pth"
    save_head_as: str = "agnews_head.pth"

    # Model
    embed_dim: int = 512
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 512
    dropout: float = 0.1

    # Train
    epochs: int = 3
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    num_workers: int = 2
    amp: bool = True
    seed: int = 42

    # NEW: freeze the encoder (shared) and only train task head
    freeze_encoder: bool = True
