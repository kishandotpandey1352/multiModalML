# fine_tune_tab_idxqa.py
from utils.config_tab_idxqa import TabIdxQAConfig
from train_tab_idxqa import main

if __name__ == "__main__":
    cfg = TabIdxQAConfig()
    # tweak here if needed, e.g.:
    # cfg.csv_path = "data/adult.csv"
    # cfg.windowed_loss = True
    main(cfg)
