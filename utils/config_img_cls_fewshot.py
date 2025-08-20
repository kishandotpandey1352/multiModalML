# config_img_cls_fewshot.py
from dataclasses import dataclass
from utils.config_img_cls import ImgClsConfig

@dataclass
class ImgClsFewShotConfig(ImgClsConfig):
    fewshot_per_class: int = 20     # K per class (e.g., CIFAR-10 → 200 total)
    max_total: int | None = None    # optional extra cap
    stream_take: int | None = None  # optional cap on raw stream before few-shot filter

    # Separate logs/checkpoints so you don't overwrite full-data runs
    csv_log_path: str = "logs/img_cls_fewshot.csv"
    save_head_as: str = "img_cls_head_fewshot.pth"
    save_dir: str = "checkpoints/img_fewshot"
