import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader
from ml_datasets.agnews_dataset import AGNewsByteDataset
from encoder.byte_encoder import ByteEncoder
from models.byte_classifier import ByteClassifier
from configurations.config import config as config_module  # Correct way to import

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Create dictionary config expected by ByteEncoder ===
config = {
    'embed_dim': config_module['embed_dim'],
    'num_layers': config_module['num_layers'],
    'num_heads': config_module['num_heads'],
    'dropout': config_module['dropout'],
    'seq_len': config_module['seq_len'],
    'vocab_size': config_module['vocab_size'],
    'num_modalities': config_module.get('num_modalities', 1),
    'hidden_dim': config_module['hidden_dim']
}

# === Model Initialization ===
encoder = ByteEncoder(config).to(device)
classifier = ByteClassifier(
    encoder=encoder,
    embed_dim=config['embed_dim'],
    num_classes=4
).to(device)
# === Freeze encoder ===
for param in encoder.parameters():
    param.requires_grad = False

# === Load pre-trained encoder weights if available ===
ckpt_path = 'checkpoints/V1/model.pth'
if os.path.exists(ckpt_path):
    print(f"Loading encoder from {ckpt_path}")
    encoder.load_state_dict(torch.load(ckpt_path)["encoder"])
else:
    print("No pretrained encoder found. Exiting.")
    exit(1)

# === Dataset & Dataloaders ===
train_dataset = AGNewsByteDataset(split="train")
val_dataset = AGNewsByteDataset(split="test")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# === Training Setup ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=1e-4)
num_epochs = 10
log_path = "logs/agnews_finetune_log.csv"
os.makedirs("logs", exist_ok=True)

# === Logging Header ===
with open(log_path, "w") as f:
    f.write("epoch,train_loss,val_loss,val_accuracy,timestamp\n")

# === Training Loop ===
for epoch in range(1, num_epochs + 1):
    encoder.eval()
    classifier.train()
    total_train_loss = 0

    for batch in train_loader:
        print(batch.keys())
        byte_input = batch["byte_input"].to(device).long()
        labels = batch["label"].to(device)
        print(f"[DEBUG] dtype={byte_input.dtype}, device={byte_input.device}")

        logits = classifier(byte_input, modality_index=torch.zeros_like(labels).to(device))
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # === Validation ===
    encoder.eval()
    classifier.eval()
    total_val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            byte_input = batch["byte_input"].to(device).long()
            labels = batch["label"].to(device)
            print(f"[DEBUG] dtype={byte_input.dtype}, device={byte_input.device}")

            logits = classifier(byte_input, modality_index=torch.zeros_like(labels).to(device))

            loss = criterion(logits, labels)
            total_val_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = total_val_loss / len(val_loader)
    accuracy = correct / total * 100

    # === Logging ===
    with open(log_path, "a") as f:
        f.write(f"{epoch},{avg_train_loss:.4f},{avg_val_loss:.4f},{accuracy:.2f},{datetime.now()}\n")

    print(f"\nEpoch {epoch}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.2f}%")
    
# === Save Fine-Tuned Classifier ===
os.makedirs("checkpoints", exist_ok=True)
torch.save({
    'encoder': classifier.encoder.state_dict(),
    'classifier': classifier.state_dict()
}, "checkpoints/agnews_classifier_ft.pth")
print("Fine-tuned model saved to checkpoints/agnews_classifier_ft.pth")
