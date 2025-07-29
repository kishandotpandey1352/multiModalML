import sys
import torch
from byte_transformer import ByteTransformerClassifier
import random

INPUT_LEN = 2048
LABELS = ['audio', 'image', 'text', 'table']
MODEL_PATH = 'classifier_model.pth'

model = ByteTransformerClassifier(input_len=INPUT_LEN)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

if len(sys.argv) < 2:
    print("Usage: python predict_byte.py path/to/file")
    sys.exit(1)

file_path = sys.argv[1]

with open(file_path, 'rb') as f:
    f.seek(0, 2)
    file_size = f.tell()

    if file_size <= 2048:
        f.seek(0)
        raw_bytes = f.read()
    else:
        offset = random.randint(0, file_size - 2048)
        f.seek(offset)
        raw_bytes = f.read(2048)

x = torch.tensor(list(raw_bytes), dtype=torch.uint8).float() / 255.0
if x.size(0) < INPUT_LEN:
    pad = torch.zeros(INPUT_LEN - x.size(0))
    x = torch.cat([x, pad])

with torch.no_grad():
    logits = model(x.unsqueeze(0))
    pred = logits.argmax(dim=1).item()
    print(f"Predicted modality: {LABELS[pred]}")