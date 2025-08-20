import matplotlib.pyplot as plt
import pandas as pd

# Load CSV
df = pd.read_csv("logs/img_cls.csv", header=None, skiprows=1)
df.columns = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc1", "val_acc5"]

# Plot losses
plt.figure(figsize=(10,5))
plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker='o')
plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss (CIFAR Image Classification)")
plt.legend()
plt.grid(True)
plt.savefig("training_vs_validation_loss.png")

# Plot accuracies
plt.figure(figsize=(10,5))
plt.plot(df["epoch"], df["train_acc"], label="Train Accuracy (Top-1)", marker='o')
plt.plot(df["epoch"], df["val_acc1"], label="Validation Accuracy (Top-1)", marker='o')
plt.plot(df["epoch"], df["val_acc5"], label="Validation Accuracy (Top-5)", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy (CIFAR Image Classification)")
plt.legend()
plt.grid(True)
plt.savefig("training_vs_validation_accuracy.png")
