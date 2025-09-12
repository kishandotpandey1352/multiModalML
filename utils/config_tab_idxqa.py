# config_tab_idxqa.py
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TabIdxQAConfig:
    # Data
    csv_path: str = "data/adult.csv"
    use_header: bool = True
    delimiter: str = ","
    max_rows_per_table: int = 50_000
    # Indexing/query style
    index_base: int = 1            # 1 for human-friendly in questions; internal is 0-based
    use_col_index: bool = False    # ask with numeric column index instead of name

    # Serialization / bytes
    src_max_len: int = 1024
    pad_token: int = 255
    remap_255_to: int = 254        # remap content byte 255 → 254; 255 is PAD
    add_schema_header: bool = False
    include_date_decompose: bool = True

    # Train
    epochs: int = 5
    train_batch_size: int = 32
    val_batch_size: int = 16
    amp: bool = False
    lr_head: float = 1e-3
    weight_decay: float = 1e-2
    clip_grad_norm: float = 1.0
    seed: int = 42
    log_every_n: int = 10
    train_steps_per_epoch: int = 200
    val_steps: int = 10
    windowed_loss: bool = True     # True = mask logits outside target value span

    # Encoder init / saving
    init_checkpoint: str = "checkpoints/V1/model.pth"
    encoder_config_path: Optional[str] = None
    encoder_config_class: Optional[str] = None
    save_dir: str = "checkpoints/tab_idxqa_adult"
    save_head_as: str = "tab_idxqa_span_head.pth"
    csv_log_path: str = "logs/tab_idxqa_adult.csv"

    # Loader perf
    num_workers_train: int = 1
    num_workers_val: int = 1
    prefetch_factor: int = 2

    # Optional: unfreeze last N encoder blocks (0 = fully frozen)
    unfreeze_n_layers: int = 1
