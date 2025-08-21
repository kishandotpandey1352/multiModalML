
from dataclasses import dataclass
from typing import Optional

@dataclass
class SumExtConfig:
    # Data: XSum via Parquet streaming
    dataset_name: str = "EdinburghNLP/xsum"
    dataset_revision: str = "refs/convert/parquet"   # Parquet branch for script-free loading
    split_train = "train"       # subset to 5k docs
    split_val   = "validation"
    split_test  = "test"
    use_hf_streaming = False            # download parquet once, faster than streaming

    # Fields inside the dataset
    src_field: str = "document"
    tgt_field: str = "summary"

    # Byte encoder / sequence settings
    src_max_len: int = 2048           # bytes (fits most XSum docs after truncation)
    pad_token: int = 255
    add_space_between_sentences: bool = True

    # Sentence selection
    max_sentences: int = 30          # cap sentences per doc for efficiency
    top_k: int = 2                    # number of sentences to extract
    max_summary_tokens: Optional[int] = None  # alternatively, a byte/token budget (None => use top_k)

    # Model (uses your shared encoder)
    embed_dim: int = 512
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 512
    dropout: float = 0.1

    # Train
    epochs: int = 3
    train_batch_size: int = 8
    eval_batch_size: int = 8
    lr_head: float = 3e-4
    lr_encoder: float = 1e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    num_workers: int = 0            # use 0 for IterableDataset / HPC stability
    pin_memory: bool = False
    amp: bool = True
    seed: int = 42
    freeze_encoder: bool = True      # start safe; you can unfreeze later

    # Checkpoints & logging
    checkpoints_dir: str = "checkpoints"
    init_checkpoint: str = "checkpoints/V1/model.pth"  # your shared encoder
    save_encoder_as: str = "shared_encoder_finetuned_sumext.pth"
    save_head_as: str = "sumext_head.pth"
    csv_log_path: str = "logs/sumext_log.csv"
    save_best_by: str = "rougeL"     # or "loss"
    log_every_n: int = 200
    max_train_batches: int | None = None   # None for full; 200 to smoke-test
    verification_no_checks: bool = True   # skip HF size checks for Parquet/streaming

    oracle_metric: str = 'rouge1'
