# fine_tune_tab_qa.py
from utils.config_tab_qa import TabQAConfig
from train_tab_qa import main

if __name__ == "__main__":
    cfg = TabQAConfig()

    # === Your local table ===
    cfg.csv_path = "data/adult.csv"          # <--- set this
    cfg.val_ratio = 0.10

    # (Optional but recommended) help synthesis pick rows & answers
    cfg.selector_cols = ["age", "education", "marital-status"]
    cfg.target_cols   = ["hours-per-week", "capital-gain", "occupation"]

    # Byte adapter / training knobs (tweak if needed)
    cfg.src_max_len = min(cfg.src_max_len, 1024)
    cfg.train_batch_size = max(16, cfg.train_batch_size)
    cfg.val_batch_size   = 16
    cfg.train_steps_per_epoch = 200   # fast sanity run
    cfg.val_steps = 10
    cfg.epochs = 5
    cfg.log_every_n = 10
    cfg.max_selectors = 4

    cfg.unfreeze_n_layers = 1
    # Encoder checkpoint you already use
    cfg.init_checkpoint = "checkpoints/V1/model.pth"

    # Where to save QA head + logs
    cfg.save_dir = "checkpoints/tab_qa_adult"
    cfg.csv_log_path = "logs/tab_qa_adult.csv"

    main(cfg)
