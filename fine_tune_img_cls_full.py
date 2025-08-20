# fine_tune_img_cls_full.py
from utils.config_img_cls import ImgClsConfig
from train_img_cls import main

if __name__ == "__main__":
    cfg = ImgClsConfig()
    # If you want a separate log/checkpoint path for full runs:
    cfg.csv_log_path = "logs/img_cls_full.csv"
    cfg.save_dir = "checkpoints/img_full"
    main(cfg)
