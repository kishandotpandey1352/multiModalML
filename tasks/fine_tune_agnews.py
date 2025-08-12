
from utils.config import TrainingConfig
from train_task import main as run

if __name__ == "__main__":
    cfg = TrainingConfig(task="agnews", use_hf=True)
    run(cfg)
