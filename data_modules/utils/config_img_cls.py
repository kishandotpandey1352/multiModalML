# config_img_cls.py
from dataclasses import dataclass

@dataclass
class ImgClsConfig:
    # Data
    dataset_name: str = "cifar10"     # e.g., "cifar10", "frgfm/imagenette"
    dataset_config: str | None = None # e.g., "320px" for imagenette, else None
    split_train: str = "train"
    split_val: str   = "test"         # CIFAR-10 uses 'test' as eval split
    img_field_candidates: tuple[str,...] = ("img","image")
    label_field: str = "label"
    use_hf_streaming: bool = True
    verification_no_checks: bool = True

    # Adapter / tokenization (bytes-first baseline)
    resize: int = 20        # shorter side; 0 to disable
    to_png_bytes: bool = False   # encode PIL -> PNG bytes in-memory
    src_max_len: int = 1024    # byte sequence cap
    pad_token: int = 255        # must match your encoder
    remap_255_to: int = 254     # avoid masking true 255 bytes
    num_classes: int = 10

    # Train
    train_batch_size: int = 64
    val_batch_size: int   = 64
    num_workers: int = 0
    pin_memory: bool = False
    epochs: int = 30
    seed: int = 123
    amp: bool = True

    # Optim
    lr_head: float = 2e-3
    weight_decay: float = 0.01
    clip_grad_norm: float = 1.0

    # Checkpoints / logs
    init_checkpoint: str = "checkpoints/V1/model.pth"  # your shared encoder
    save_encoder_as: str | None = None   # keep None → don’t overwrite shared encoder
    save_head_as: str = "img_cls_head.pth"
    csv_log_path: str = "logs/img_cls.csv"
    save_dir: str = "checkpoints/img"
    best_by: str = "val_acc"

    # Misc
    log_every_n: int = 200
    max_train_batches: int | None = None  # e.g., 500 for quick debug
    encoder_config_module: str | None = None
    encoder_config_class: str | None = "TrainingConfig"
    # config_img_cls.py  (append fields)
    encoder_config_path: str | None = r"D:\dissertationCode\fine-tuning\multiModalML\utils\config.py"
