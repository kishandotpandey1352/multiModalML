from transformers import get_linear_schedule_with_warmup

# Rebuild model, optimizer, scheduler, and training loop
from configurations import config
import torch
from torch.utils.data import DataLoader, random_split
from loader.text_dataloader import ByteTextDataset
from utility.masking import mask_input
from models.autoencoder_factory import autoencoder_factory
from decoders.masked_prediction import masked_byte_loss
from utility.experiment_logger import log_experiment

DEVICE = config.DEVICE
VAL_SPLIT = 0.1
TOTAL_EPOCHS = 100
PATIENCE = 10
CHECKPOINT_PATH = "checkpoints/best_masked_autoencoder.pt"

# Dataset loading and split
dataset = ByteTextDataset(folder_path="dataset/text", seq_len=config.SEQ_LEN)
val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

# Model, optimizer, scheduler
autoencoder = autoencoder_factory(task="masked").to(DEVICE)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=config.LR)

# Add LR scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=200,
    num_training_steps=TOTAL_EPOCHS * len(train_loader)
)

# Training and validation
def train_epoch(model, dataloader, optimizer, scheduler):
    model.train()
    total_loss = 0

    for batch in dataloader:
        batch = batch.to(DEVICE)
        masked_input, labels = mask_input(batch)
        logits = model(masked_input)
        loss = masked_byte_loss(logits, labels.to(DEVICE))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

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

# Training loop with updated patience and scheduler
best_accuracy = 0
patience_counter = 0

for epoch in range(1, TOTAL_EPOCHS + 1):
    train_loss = train_epoch(autoencoder, train_loader, optimizer, scheduler)
    val_accuracy, val_loss = validate(autoencoder, val_loader)

    print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.2%}")

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        patience_counter = 0
        torch.save(autoencoder.state_dict(), CHECKPOINT_PATH)
        print("Best model updated.")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("⏹Early stopping triggered.")
            break

    log_experiment(
        dataset_name=f"text-{len(dataset)}",
        num_files=len(dataset),
        mask_prob=config.MASK_PROB,
        mask_strategy="span" if "span" in mask_input.__name__ else "random",
        embed_dim=config.EMBED_DIM,
        num_layers=config.NUM_LAYERS,
        hidden_dim=config.HIDDEN_DIM,
        num_heads=config.NUM_HEADS,
        epochs=epoch,
        batch_size=config.BATCH_SIZE,
        learning_rate=config.LR,
        accuracy=val_accuracy,
        loss=val_loss,
        notes="warmup+decay, patience=10, dynamic masking"
    )
