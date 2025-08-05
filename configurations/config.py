SEQ_LEN = 1024
VOCAB_SIZE = 256
MASK_PROB = 0.15
MASK_TOKEN = 255
IGNORE_INDEX = -100
EMBED_DIM = 512
HIDDEN_DIM = 512
NUM_LAYERS = 6
NUM_HEADS = 8
DROPOUT = 0.1
MODALITY = 'text'
MASK_SPAN_LENGTH = 4
USE_SPAN_MASKING = True
MODEL_NAME = 'byte_encoder_decoder'
CHECKPOINT_PATH = 'checkpoints/V1/model.pth'
VERSION = 'V1'
CHECKPOINT_DIR = 'checkpoints/V1'
TEST_MODEL_PATH = 'checkpoints/V1/model.pth'
NUM_MODALITIES = 4
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_DELTA = 0.001
SAMPLE_SIZE = -1
BATCH_SIZE = 1
EPOCHS = 10
LR = 0.0001
DEVICE = 'cuda'
DATA_PATH = 'dataset'

config = {
    "seq_len": SEQ_LEN,
    "vocab_size": VOCAB_SIZE,
    "mask_prob": MASK_PROB,
    "mask_token": MASK_TOKEN,
    "ignore_index": IGNORE_INDEX,
    "embed_dim": EMBED_DIM,
    "hidden_dim": HIDDEN_DIM,
    "num_layers": NUM_LAYERS,
    "num_heads": NUM_HEADS,
    "dropout": DROPOUT,
    "modality": MODALITY,
    "mask_span_length": MASK_SPAN_LENGTH,
    "use_span_masking": USE_SPAN_MASKING,
    "model_name": MODEL_NAME,
    "checkpoint_path": CHECKPOINT_PATH,
    "version": VERSION,
    "checkpoint_dir": CHECKPOINT_DIR,
    "test_model_path": TEST_MODEL_PATH,
    "num_modalities": NUM_MODALITIES,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "early_stopping_delta": EARLY_STOPPING_DELTA,
    "sample_size": SAMPLE_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "lr": LR,
    "device": DEVICE,
    "data_path": DATA_PATH,
}