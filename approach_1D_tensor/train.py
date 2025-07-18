# train.py

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from gen_dataloader import UniversalModalityDataset as ModalityDataset
from mod_classifier import GeneralModalityEncoder

# Parameters
root_dir = 'dataset'         
input_dim = 1024
batch_size = 32
num_epochs = 10
learning_rate = 1e-3
num_classes = 4

# Dataset and DataLoader
dataset = ModalityDataset(root_dir=root_dir, input_dim=input_dim)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model and Optimizer
model = GeneralModalityEncoder(input_dim=input_dim, num_classes=num_classes)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    total_loss, correct, count = 0.0, 0, 0
    model.train()

    for x, y in dataloader:
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        count += x.size(0)

    acc = correct / count
    avg_loss = total_loss / count
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")

# Save the model
torch.save(model.state_dict(), "modality_encoder.pth")

# Reload the dataset for full evaluation (or reuse dataloader)
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for x, y in dataloader:
        logits = model(x)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Audio', 'Image', 'Text', 'Table'])
disp.plot(cmap='Blues')
plt.title("Modality Classification Confusion Matrix")
plt.show()