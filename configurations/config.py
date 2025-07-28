# ----- Data -----
SEQ_LEN = 512
VOCAB_SIZE = 256          # Byte range (0-255)

# ----- Masking -----
MASK_PROB = 0.15
MASK_TOKEN = 255          # Used to replace masked bytes
IGNORE_INDEX = -100       # Used in loss (ignore unmasked)

# ----- Model -----
EMBED_DIM = 384
HIDDEN_DIM = 512
NUM_LAYERS = 8
NUM_HEADS = 8
DROPOUT = 0.1

# ----- Training -----
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4
DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
