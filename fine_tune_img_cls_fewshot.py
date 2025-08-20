# fine_tune_img_cls_fewshot.py
from utils.config_img_cls_fewshot import ImgClsFewShotConfig
from train_img_cls_fewshot import main

if __name__ == "__main__":
    cfg = ImgClsFewShotConfig()
    # Example tweaks (optional):
    # cfg.fewshot_per_class = 10
    # cfg.stream_take = 20000
    # cfg.epochs = 50
    main(cfg)
