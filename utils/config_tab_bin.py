# utils/config_tab_bin.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class TabBinConfig:
    # HF dataset (your loader maps "adult" -> scikit-learn/adult-census-income)
    dataset_name: str = "adult"
    dataset_config: Optional[str] = None
    split_train: str = "train"
    split_val:   str = "test"
    label_column: str = "income"
    positive_label: Optional[str] = ">50K"

    # Streaming / loading
    use_hf_streaming: bool = True
    verification_no_checks: bool = True
    num_workers: int = 0
    pin_memory: bool = False

    # Byte adapter
    src_max_len: int = 1024
    pad_token: int = 255
    remap_255_to: int = 254

    # Serialization hints
    add_schema_header: bool = True
    include_date_decompose: bool = True

    # Train (not used in predict, but fine to keep)
    train_batch_size: int = 64
    val_batch_size: int   = 8
    epochs: int = 15
    seed: int = 123
    amp: bool = True
    log_every_n: int = 200
    max_train_batches: Optional[int] = None

    # Optim
    lr_head: float = 2e-3
    weight_decay: float = 0.01
    clip_grad_norm: float = 1.0

    # Encoder init — load from checkpoint only (portable)
    init_checkpoint: str = "checkpoints/V1/model.pth"
    encoder_config_path: Optional[str] = None
    encoder_config_class: Optional[str] = None

    # Head checkpoint (absolute)
    head_checkpoint: Optional[str] = "/users/acp24kp/finetuning/multiModalML/checkpoints/tab_bin/tab_bin_head.pth"

    # Saving / logs
    save_dir: str = "checkpoints/tab_bin"
    save_head_as: str = "tab_bin_head.pth"
    csv_log_path: str = "logs/tab_bin.csv"
