# tasks/test_masked.py

import torch
from loader.text_dataloader import ByteTextDataset
from utility.masking import mask_input
from models.autoencoder_factory import autoencoder_factory
from decoders.masked_prediction import masked_byte_loss
from configurations import config
from utility.experiment_logger import log_experiment
import argparse

DEVICE = config.DEVICE

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="text-1000", help="Name of dataset or experiment")
parser.add_argument("--notes", type=str, default="", help="Any notes to attach to the log")
args = parser.parse_args()
# -----------------------------
# Byte decoding for visualization
# -----------------------------
def tensor_to_display_text(tensor, mask_token=config.MASK_TOKEN):
    out = []
    for val in tensor:
        byte = val.item()
        if byte == mask_token:
            out.append('■')
        elif 32 <= byte <= 126:
            out.append(chr(byte))
        elif byte in [9, 10, 13]:  # tab, newline, carriage return
            out.append(chr(byte))
        else:
            out.append('.')
    return ''.join(out)

# -----------------------------
# Load model and dataset
# -----------------------------
model = autoencoder_factory(task="masked").to(DEVICE)
model.load_state_dict(torch.load("checkpoints/best_masked_autoencoder.pt", map_location=DEVICE))
# model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

dataset = ByteTextDataset(folder_path="dataset/text", seq_len=config.SEQ_LEN)
sample = dataset[0].unsqueeze(0).to(DEVICE)

# Mask input
masked_input, labels = mask_input(sample)
masked_input, labels = masked_input.to(DEVICE), labels.to(DEVICE)

# Run model
with torch.no_grad():
    logits = model(masked_input)
    predictions = torch.argmax(logits, dim=-1)

# Prepare for display and analysis
original = sample.squeeze()
masked = masked_input.squeeze()
predicted = predictions.squeeze()

print(" Original (text):")
print(tensor_to_display_text(original))

print("\n Masked input (with [■]):")
print(tensor_to_display_text(masked))

print("\n Predicted (text):")
print(tensor_to_display_text(predicted))

# Compute byte-level accuracy
correct = ((predicted == original) & (labels.squeeze() != -100)).sum().item()
total = (labels.squeeze() != -100).sum().item()
accuracy = correct / total if total > 0 else 0

print("\n Evaluation Summary:")
print(f"Byte-level accuracy on masked tokens: {accuracy:.2%}")

# Optional: Calculate average loss
loss = masked_byte_loss(logits, labels).item()

# -----------------------------
# Log experiment
# -----------------------------
log_experiment(
    dataset_name=args.dataset,
    num_files=len(dataset),
    mask_prob=config.MASK_PROB,
    mask_strategy="random",
    embed_dim=config.EMBED_DIM,
    num_layers=config.NUM_LAYERS,
    hidden_dim=config.HIDDEN_DIM,
    num_heads=config.NUM_HEADS,
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    learning_rate=config.LR,
    accuracy=accuracy,
    loss=loss,
    notes=args.notes
)
