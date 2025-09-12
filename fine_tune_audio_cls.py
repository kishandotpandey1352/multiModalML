# fine_tune_audio_cls.py
import os, importlib.util, inspect
from utils.config_audio_cls import AudioClsConfig  # your config

BASE = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE, "train_audio_cls.py")  # <- pinned to THIS folder

spec = importlib.util.spec_from_file_location("train_audio_cls_pinned", TRAIN_PATH)
mod = importlib.util.module_from_spec(spec); assert spec.loader is not None
spec.loader.exec_module(mod)

if __name__ == "__main__":
    cfg = AudioClsConfig()
    # safety knobs for the first run; tweak later
    # cfg.disable_eval = True
    # cfg.encoder_microbatch = 1
    # cfg.freeze_first_batch = True
    # cfg.max_train_steps = 10
    # cfg.batch_size = 1
    # cfg.val_batch_size = 1
    # cfg.stream_take = 64
    # cfg.src_max_len = 512
    cfg.freeze_first_batch = False     # use the real DataLoader again
    cfg.disable_eval = False           # turn eval back on
    cfg.max_val_batches = 10           # cap eval size for now

    cfg.encoder_microbatch = 1         # keep for safety
    cfg.batch_size = 1                 # start tiny
    cfg.val_batch_size = 1
    cfg.stream_take = None              # small streamed subset; remove later
    cfg.src_max_len = 4096                      # keep context small for stability
    cfg.max_train_steps = 0       # optional cap for the first epoch
    cfg.epochs = 20                     # short shakedown
    cfg.encoder_config_path = os.path.join(BASE, "utils", "config.py")
    print("using encoder_config_path:", cfg.encoder_config_path)
    print(">> USING train_audio_cls from:", getattr(mod, "__file__", TRAIN_PATH))
    print(">>> SAFE PATCH ACTIVE <<<")
    mod.main(cfg)
