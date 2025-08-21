from utils.config_tab_bin import TabBinConfig
from train_tab_bin import main

if __name__ == "__main__":
    cfg = TabBinConfig()

    # Adult (scikit-learn mirror; no script; safe with datasets>=3)
    cfg.dataset_name   = "scikit-learn/adult-census-income"
    cfg.dataset_config = None
    cfg.label_column   = "income"      # <-- was 'target'
    cfg.positive_label = ">50K"
    cfg.split_train    = "train[:90%]" # single split -> slice it
    cfg.split_val      = "train[90%:]"
    cfg.use_hf_streaming = False       # keep off on Windows

    cfg.csv_log_path = "logs/tab_bin_adult_sklearn.csv"
    cfg.save_dir     = "checkpoints/tab_bin_adult_sklearn"

    main(cfg)
