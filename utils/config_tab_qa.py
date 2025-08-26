# utils/config_tab_qa.py
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class TabQAConfig:
    # ---------- Data source (no downloads) ----------
    # Provide either csv_path OR a dataframe at runtime (see fine_tune_tab_qa.py)
    csv_path: Optional[str] = None            # e.g., "data/my_table.csv"
    use_header: bool = True                   # CSV has header
    delimiter: str = ","
    # columns explicitly allowed as selectors / targets (None = infer)
    selector_cols: Optional[List[str]] = None
    target_cols: Optional[List[str]] = None
    # split by rows (90/10 default)
    val_ratio: float = 0.1
    # synthesis knobs
    min_selectors: int = 1                    # choose 1..max_selectors selectors
    max_selectors: int = 4
    avoid_id_like: bool = True                # don't target columns that look like IDs
    max_rows_per_table: int = 50000           # cap rows loaded from CSV

    # ---------- Byte adapter ----------
    src_max_len: int = 1024
    pad_token: int = 255
    remap_255_to: int = 254
    add_schema_header: bool = True
    include_date_decompose: bool = True

    # ---------- Training ----------
    epochs: int = 8
    train_batch_size: int = 24
    val_batch_size: int = 24
    amp: bool = False
    lr_head: float = 2e-3
    weight_decay: float = 0.01
    clip_grad_norm: float = 1.0
    seed: int = 42
    log_every_n: int = 20
    train_steps_per_epoch: int = 100   # tweak for speed; e.g., 200 for a quick run
    val_steps: int = 20                # number of validation batches per epoch

    # ---------- Encoder init (same pattern you already use) ----------
    init_checkpoint: str = "checkpoints/V1/model.pth"
    encoder_config_path: Optional[str] = None   # optional: path to your encoder config.py
    encoder_config_class: Optional[str] = None  # class name in that config

    # ---------- Saving ----------
    save_dir: str = "checkpoints/tab_qa"
    save_head_as: str = "tab_qa_span_head.pth"
    csv_log_path: str = "logs/tab_qa.csv"
