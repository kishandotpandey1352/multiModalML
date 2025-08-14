
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class NERTrainingConfig:
    # Data
    task: str = "ner_wikiann_en"
    dataset_name: str = "unimelb-nlp/wikiann"
    dataset_config: str | None = "en"   # language code for WikiANN; set None for datasets without configs
    split_train: str = "train"
    split_val: str = "validation"
    split_test: str = "test"
    use_hf_streaming: bool = False
    max_len: int = 1024
    pad_token: int = 255
    bos_token: int = 257
    eos_token: int = 258
    add_bos: bool = False
    add_eos: bool = False
    ignore_index: int = -100

    # Default label list (fallback)
    label_list: List[str] = field(default_factory=lambda: [
        "O",
        "B-corporation","I-corporation",
        "B-creative-work","I-creative-work",
        "B-group","I-group",
        "B-location","I-location",
        "B-person","I-person",
        "B-product","I-product",
    ])

    # Checkpoints
    checkpoints_dir: str = "checkpoints"
    init_checkpoint: str = "checkpoints/V1/model.pth"
    save_encoder_as: str = "shared_encoder_finetuned_ner.pth"
    save_head_as: str = "ner_head.pth"

    # Model
    embed_dim: int = 512
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 512
    dropout: float = 0.1

    # Train
    epochs: int = 5
    train_batch_size: int = 16
    eval_batch_size: int = 32
    lr_head: float = 3e-4
    lr_encoder: float = 1e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    num_workers: int = 0
    pin_memory = False
    amp: bool = True
    seed: int = 42
    freeze_encoder: bool = True

    # Logging
    csv_log_path: str = "logs/ner_log.csv"
    save_best_by: str = "f1"
