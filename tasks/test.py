from ml_datasets.agnews_dataset import AGNewsByteDataset

ds = AGNewsByteDataset(split="train", max_len=256)
sample = ds[0]

print("Byte Input:", sample["input"].shape)  # torch.Size([256])
print("Label:", sample["label"])             # e.g., 0 (World)
    