import os
import torch
from torch.utils.data import Dataset, DataLoader

MODALITY_LIST = ['text', 'image', 'audio', 'table']  # Add others like 'tabular' if needed
# MODALITY_TO_INDEX = {mod: i for i, mod in enumerate(MODALITY_LIST)}
MODALITY_TO_INDEX = {
    'text': 0,
    'audio': 1,
    'image': 2,
    'table': 3
}
NUM_MODALITIES = len(MODALITY_TO_INDEX)
class MultiModalDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for modality in MODALITY_LIST:
            folder = os.path.join(root_dir, modality)
            if not os.path.exists(folder): continue
            for fname in os.listdir(folder):
                fpath = os.path.join(folder, fname)
                if os.path.isfile(fpath):
                    self.samples.append({
                        'file_path': fpath,
                        'modality': modality
                    })
                    self.samples = self.samples[:400] #restrict to first 200 samples of each modality

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        file_path = entry['file_path']
        modality = entry['modality']
        mod_idx = MODALITY_TO_INDEX.get(modality, -1)

        try:
            with open(file_path, 'rb') as f:
                byte_data = f.read()
            byte_tensor = torch.tensor(list(byte_data), dtype=torch.long)[:512]
        except Exception as e:
            print(f"[Error reading {file_path}]: {e}")
            byte_tensor = torch.full((512,), 255, dtype=torch.long)  # masked padding

        # pad to fixed length
        padded = torch.full((512,), 255, dtype=torch.long)
        padded[:len(byte_tensor)] = byte_tensor

        return {
            'byte_input': padded,
            'modality': modality,
            'modality_index': torch.tensor(mod_idx, dtype=torch.long),
            'file_path': file_path
        }

def get_multimodal_loader(root, batch_size=8, shuffle=True):
    dataset = MultiModalDataset(root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: x)
