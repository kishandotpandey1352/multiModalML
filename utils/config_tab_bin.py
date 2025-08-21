# utils/config_tab_bin.py
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class TabBinConfig:
    # HF dataset
    dataset_name: str = "adult"          # e.g., "adult", "bank_marketing"
    dataset_config: Optional[str] = None
    split_train: str = "train"
    split_val:   str = "test"
    label_column: str = "income"         # adult: "income"; bank_marketing: "y"
    positive_label: Optional[str] = ">50K"  # adult: ">50K"; bank_marketing: "yes"; if None → numeric {0,1}

    # Streaming / loading
    use_hf_streaming: bool = True
    verification_no_checks: bool = True
    num_workers: int = 0
    pin_memory: bool = False

    # Byte adapter (keep consistent with your encoder)
    src_max_len: int = 1024
    pad_token: int = 255
    remap_255_to: int = 254

    # Optional: include lightweight schema/type hints in the serialization
    add_schema_header: bool = True
    include_date_decompose: bool = True   # if a value looks like YYYY-MM-DD

    # Train
    train_batch_size: int = 64
    val_batch_size: int   = 64
    epochs: int = 15
    seed: int = 123
    amp: bool = True
    log_every_n: int = 200
    max_train_batches: int | None = None

    # Optim
    lr_head: float = 2e-3
    weight_decay: float = 0.01
    clip_grad_norm: float = 1.0

    # Encoder init
    init_checkpoint: str = "checkpoints/V1/model.pth"
    encoder_config_path: str | None = r"D:\dissertationCode\fine-tuning\multiModalML\utils\config.py"
    encoder_config_class: str | None = "TrainingConfig"

    # Saving / logs
    save_dir: str = "checkpoints/tab_bin"
    save_head_as: str = "tab_bin_head.pth"
    csv_log_path: str = "logs/tab_bin.csv"
