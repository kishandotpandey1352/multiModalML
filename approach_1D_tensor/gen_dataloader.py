import os
import torch
from torch.utils.data import Dataset
import torchaudio
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import mimetypes
import warnings
import string

class UniversalModalityDataset(Dataset):
    def __init__(self, root_dir, input_dim=1024):
        self.root_dir = root_dir
        self.input_dim = input_dim
        self.samples = []
        self.label_map = {'audio': 0, 'image': 1, 'text': 2, 'table': 3}

        for modality, label in self.label_map.items():
            dir_path = os.path.join(root_dir, modality)
            if not os.path.isdir(dir_path):
                continue
            for fname in os.listdir(dir_path):
                fpath = os.path.join(dir_path, fname)
                if os.path.isfile(fpath):
                    self.samples.append((fpath, label))

        self.img_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

        self.mel_transform = torchaudio.transforms.MelSpectrogram()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        try:
            x = self.load_and_process(fpath)
            x = x.flatten()
        except Exception as e:
            warnings.warn(f"Failed to load {fpath}: {e}")
            x = torch.zeros(self.input_dim)

        if x.shape[0] < self.input_dim:
            x = torch.nn.functional.pad(x, (0, self.input_dim - x.shape[0]))
        elif x.shape[0] > self.input_dim:
            x = x[:self.input_dim]

        return x, label

    def load_and_process(self, fpath):
        mime_type, _ = mimetypes.guess_type(fpath)

        if mime_type is None:
            ext = os.path.splitext(fpath)[1].lower()
            if ext in ['.csv', '.xls', '.xlsx']:
                return self.process_table(fpath)
            elif ext in ['.wav', '.mp3', '.flac']:
                return self.process_audio(fpath)
            elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                return self.process_image(fpath)
            elif ext in ['.txt', '.log', '.json', '.xml']:
                return self.process_text(fpath)
            else:
                return torch.zeros(self.input_dim)

        if 'audio' in mime_type:
            return self.process_audio(fpath)
        elif 'image' in mime_type:
            return self.process_image(fpath)
        elif 'text' in mime_type:
            return self.process_text(fpath)
        elif 'csv' in mime_type or 'spreadsheet' in mime_type:
            return self.process_table(fpath)
        else:
            return torch.zeros(self.input_dim)

    def process_audio(self, fpath):
        waveform, sr = torchaudio.load(fpath)
        mel = self.mel_transform(waveform)  # [channel, freq, time]
        mel_mean = mel.mean(dim=0)  # average across channels → [freq, time]
        return mel_mean.flatten()   # flatten to 1D tensor

    def process_image(self, fpath):
        with Image.open(fpath).convert('RGB') as img:
            img_tensor = self.img_transform(img).flatten()
        return img_tensor

    def process_text(self, fpath):
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        ascii_vals = [ord(c)/128 for c in text if c in string.printable]
        return torch.tensor(ascii_vals, dtype=torch.float)

    def process_table(self, fpath):
        try:
            df = pd.read_csv(fpath)
        except:
            try:
                df = pd.read_excel(fpath)
            except:
                return torch.zeros(self.input_dim)
        df = df.select_dtypes(include=[np.number]).dropna()
        if df.empty:
            return torch.zeros(self.input_dim)
        return torch.tensor(df.values.flatten(), dtype=torch.float)