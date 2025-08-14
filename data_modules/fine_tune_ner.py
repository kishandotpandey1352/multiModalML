
from utils.config_ner import NERTrainingConfig
from train_ner import main as run

if __name__ == "__main__":
    cfg = NERTrainingConfig()
    run(cfg)
