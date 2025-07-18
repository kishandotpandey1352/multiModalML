# train_byte_model.py
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from byte_dataloader import ByteDataset
from byte_transformer import ByteTransformerClassifier

# Config
ROOT_DIR = 'dataset'
INPUT_LEN = 2048
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4
NUM_CLASSES = 4
PATIENCE = 5

# Dataset and Split
dataset = ByteDataset(root_dir=ROOT_DIR, input_len=INPUT_LEN)
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ByteTransformerClassifier(input_len=INPUT_LEN, num_classes=NUM_CLASSES).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

# Early Stopping Setup
best_val_loss = float('inf')
patience_counter = 0

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    train_acc = correct / total
    train_loss = total_loss / total

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            val_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            val_correct += (preds == y).sum().item()
            val_total += x.size(0)

    val_acc = val_correct / val_total
    val_loss /= val_total
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n Early stopping triggered at epoch {epoch+1}")
            break

# Load best model
model.load_state_dict(torch.load("best_model.pth"))

# Final Evaluation on Test Set
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(y.tolist())

# Accuracy
acc = accuracy_score(y_true, y_pred)
print(f"\nFinal Test Accuracy: {acc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Audio', 'Image', 'Text', 'Table'])
disp.plot(cmap='Blues')
plt.title("Byte-Level Modality Classification (Test Set)")
plt.savefig("confusion_matrix_test.png")
print("Saved confusion matrix as 'confusion_matrix_test.png'")