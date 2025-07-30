# ----- Data -----
SEQ_LEN = 512
VOCAB_SIZE = 256          # Byte range (0-255)

# ----- Masking -----
MASK_PROB = 0.30
MASK_TOKEN = 255          # Used to replace masked bytes
IGNORE_INDEX = -100       # Used in loss (ignore unmasked)

# ----- Model -----
EMBED_DIM = 512
HIDDEN_DIM = 2048
NUM_LAYERS = 8
NUM_HEADS = 16
DROPOUT = 0.1
MODALITY = "text"
MASK_SPAN_LENGTH = 10    #For spanMasking
USE_SPAN_MASKING = True #For spanMasking
# Optional dynamic filename
MODEL_NAME = f"best_spanboundary_{MODALITY}_{EMBED_DIM}d_{NUM_LAYERS}L.pt"
CHECKPOINT_PATH = f"checkpoints/{MODEL_NAME}"
TEST_MODEL_PATH = CHECKPOINT_PATH
NUM_MODALITIES = 4
# ----- Training -----
BATCH_SIZE = 2
EPOCHS = 100
LR = 1e-4
DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
