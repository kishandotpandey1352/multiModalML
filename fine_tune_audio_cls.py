# fine_tune_audio_cls.py
from utils.config_audio_cls import AudioClsConfig
from train_audio_cls import main

if __name__ == "__main__":
    cfg = AudioClsConfig()
    # Examples:
    # cfg.dataset_name = "esc50"; cfg.num_classes = 50; cfg.split_val = "test"
    # cfg.stream_take = 20000
    # cfg.epochs = 30
    main(cfg)
