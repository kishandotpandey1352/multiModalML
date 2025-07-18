# predict.py

import torch
import torch.nn.functional as F
from mod_classifier import GeneralModalityEncoder
import torchaudio
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
import os

INPUT_DIM = 1024
LABELS = ['audio', 'image', 'text', 'table']
MODEL_PATH = "modality_encoder.pth"

# Load model
model = GeneralModalityEncoder(input_dim=INPUT_DIM, num_classes=4)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# --- Preprocessing Functions ---
def preprocess_audio(path):
    waveform, sr = torchaudio.load(path)
    mel = torchaudio.transforms.MelSpectrogram()(waveform)  # shape: [channel, freq, time]
    mel_mean = mel.mean(dim=0)  # shape: [freq, time]
    return mel_mean.flatten()   # ensure 1D vector

def preprocess_image(path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    img = Image.open(path).convert('RGB')
    vec = transform(img).flatten()
    return vec[:INPUT_DIM]

def preprocess_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    ascii_vals = [ord(c)/128 for c in text if ord(c) < 128]
    return torch.tensor(ascii_vals[:INPUT_DIM], dtype=torch.float)

def preprocess_table(path):
    try:
        df = pd.read_csv(path).select_dtypes(include=[np.number]).dropna()
        return torch.tensor(df.values.flatten()[:INPUT_DIM], dtype=torch.float)
    except:
        return torch.zeros(INPUT_DIM)

# --- Auto detect and preprocess ---
def preprocess_file(path):
    for name, func in [
        ("audio", preprocess_audio),
        ("image", preprocess_image),
        ("text", preprocess_text),
        ("table", preprocess_table)
    ]:
        try:
            return func(path)
        except Exception as e:
            print(f"⚠️ {name} preprocessing failed: {e}")
    raise ValueError("❌ Unsupported file or unreadable content.")

# --- Main ---
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/file")
        exit(1)

    file_path = sys.argv[1]
    x = preprocess_file(file_path)

    # Pad if needed
    if x.shape[0] < INPUT_DIM:
        x = F.pad(x, (0, INPUT_DIM - x.shape[0]))

    with torch.no_grad():
        logits = model(x.unsqueeze(0))  # add batch dimension
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).item()

    print(f"Predicted modality: {LABELS[pred]}")
