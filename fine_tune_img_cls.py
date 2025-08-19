# fine_tune_img_cls.py
from utils.config_img_cls import ImgClsConfig
from train_img_cls import main

if __name__ == "__main__":
    cfg = ImgClsConfig()
    # Example switches:
    # cfg.dataset_name = "frgfm/imagenette"; cfg.dataset_config = "320px"; cfg.num_classes = 10
    # cfg.src_max_len = 4096
    main(cfg)
