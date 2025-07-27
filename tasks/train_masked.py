# tasks/train_masked.py

import torch
import os
from torch.utils.data import DataLoader, random_split
from loader.text_dataloader import ByteTextDataset
from utility.masking import mask_input
from models.autoencoder_factory import autoencoder_factory
from decoders.masked_prediction import masked_byte_loss
from configurations import config
from utility.experiment_logger import log_experiment

# --------------------------------
# Parameters
# --------------------------------
DEVICE = config.DEVICE
TOTAL_EPOCHS = config.EPOCHS
VAL_SPLIT = 0.1
CHECKPOINT_PATH = "checkpoints/masked_autoencoder.pt"

# --------------------------------
# Dataset
# --------------------------------
dataset = ByteTextDataset(folder_path="dataset/text", seq_len=config.SEQ_LEN)
val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

# --------------------------------
# Model + Optimizer
# --------------------------------
autoencoder = autoencoder_factory(task="masked").to(DEVICE)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=config.LR)

# --------------------------------
# Training + Validation
# --------------------------------
def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0

    for batch in dataloader:
        batch = batch.to(DEVICE)
        masked_input, labels = mask_input(batch)
        logits = model(masked_input.to(DEVICE))
        loss = masked_byte_loss(logits, labels.to(DEVICE))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def validate(model, dataloader):
    model.eval()
    total_correct, total_masked, total_loss = 0, 0, 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(DEVICE)
            masked_input, labels = mask_input(batch)
            logits = model(masked_input)
            loss = masked_byte_loss(logits, labels.to(DEVICE))
            preds = torch.argmax(logits, dim=-1)

            mask = (labels != -100).to(DEVICE)
            correct = ((preds == batch) & mask).sum().item()
            total = mask.sum().item()

            total_correct += correct
            total_masked += total
            total_loss += loss.item()

    accuracy = total_correct / total_masked if total_masked > 0 else 0
    avg_loss = total_loss / len(dataloader)
    return accuracy, avg_loss

# --------------------------------
# Train Loop
# --------------------------------
for epoch in range(1, TOTAL_EPOCHS + 1):
    train_loss = train_epoch(autoencoder, train_loader, optimizer)
    val_accuracy, val_loss = validate(autoencoder, val_loader)

    print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.2%}")

    # Save best checkpoint or every few epochs
    if epoch % 5 == 0 or epoch == TOTAL_EPOCHS:
        torch.save(autoencoder.state_dict(), f"checkpoints/masked_autoencoder_epoch{epoch}.pt")

        log_experiment(
            dataset_name=f"text-{len(dataset)}",
            num_files=len(dataset),
            mask_prob=config.MASK_PROB,
            mask_strategy="random",
            embed_dim=config.EMBED_DIM,
            num_layers=config.NUM_LAYERS,
            hidden_dim=config.HIDDEN_DIM,
            num_heads=config.NUM_HEADS,
            epochs=epoch,
            batch_size=config.BATCH_SIZE,
            learning_rate=config.LR,
            accuracy=val_accuracy,
            loss=val_loss,
            notes="epoch-level validation"
        )
