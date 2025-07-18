import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd

class TextDataset(Dataset):
    def __init__(self, csv_file, text_column='text', tokenizer_name='bert-base-uncased', max_len=128):
        self.data = pd.read_csv(csv_file)
        self.text_column = text_column
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx][self.text_column]

        # Tokenize
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        # Remove batch dimension
        return {
            'input_ids': encoded['input_ids'].squeeze(0),  # shape: (max_len,)
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'text': text
        }
