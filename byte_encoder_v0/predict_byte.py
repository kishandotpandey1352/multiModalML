import sys
import torch
from byte_transformer import ByteTransformerClassifier

INPUT_LEN = 2048
LABELS = ['audio', 'image', 'text', 'table']
MODEL_PATH = 'byte_model.pth'

model = ByteTransformerClassifier(input_len=INPUT_LEN)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

if len(sys.argv) < 2:
    print("Usage: python predict_byte.py path/to/file")
    sys.exit(1)

file_path = sys.argv[1]

with open(file_path, 'rb') as f:
    byte_data = f.read(INPUT_LEN)

x = torch.tensor(list(byte_data), dtype=torch.uint8).float() / 255.0
if x.size(0) < INPUT_LEN:
    pad = torch.zeros(INPUT_LEN - x.size(0))
    x = torch.cat([x, pad])

with torch.no_grad():
    logits = model(x.unsqueeze(0))
    pred = logits.argmax(dim=1).item()
    print(f"Predicted modality: {LABELS[pred]}")