
from utils.config_sum_ext import SumExtConfig
from train_sum_extractive import main as run

if __name__ == "__main__":
    cfg = SumExtConfig()
    run(cfg)
