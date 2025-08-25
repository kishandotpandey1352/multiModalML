# test_spanmasked.py

import argparse
import torch
from torch.utils.data import DataLoader
from loader.multiModal_dataloader import MultiModalDataset, MODALITY_TO_INDEX
from encoder.byte_encoder import ByteEncoder
from decoders.span_boundary_decoder import SpanBoundaryDecoder
from utility.span_masking import span_mask_input
from configurations.config import config as CFG  # <-- dict config

def evaluate(model_path: str, modality: str, data_path: str):
    # Validate modality
    if modality not in MODALITY_TO_INDEX:
        raise ValueError(f"Unsupported modality '{modality}'. Supported: {list(MODALITY_TO_INDEX.keys())}")

    device = torch.device(CFG["device"] if torch.cuda.is_available() else "cpu")

    # Load model
    print(f" Loading model from: {model_path}")
    encoder = ByteEncoder(CFG).to(device)
    decoder = SpanBoundaryDecoder(CFG).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    encoder.eval(); decoder.eval()

    # Dataset
    dataset = MultiModalDataset(data_path, modality=modality, split='test', from_classifier=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # default collate -> dict of tensors

    correct, total = 0, 0
    print(f" Evaluating {len(dataset)} file(s) from '{data_path}' with modality '{modality}'...")

    with torch.no_grad():
        for batch in dataloader:
            byte_input = batch['byte_input'].unsqueeze(0).to(device)      # (1, L)
            mod_index  = batch['modality_index'].unsqueeze(0).to(device)  # (1,)

            masked, labels = span_mask_input(
                byte_input,
                mask_prob=CFG["mask_prob"],
                max_span_length=CFG["mask_span_length"],
                ignore_index=CFG["ignore_index"]
            )
            masked = masked.to(device); labels = labels.to(device)

            encoded = encoder(masked, mod_index)  # (1, L, D)

            # positions of masked tokens in sample 0
            positions = (labels[0] != CFG["ignore_index"]).nonzero(as_tuple=True)[0]
            if len(positions) == 0:
                continue

            positions = positions[(positions > 0) & (positions < CFG["seq_len"] - 1)]
            MAX_SPANS = 20
            if len(positions) > MAX_SPANS:
                positions = positions[:MAX_SPANS]

            left_batch  = encoded[0, positions - 1]
            right_batch = encoded[0, positions + 1]
            rel_pos = torch.zeros(left_batch.size(0), dtype=torch.long, device=device)

            logits = decoder(left_batch, right_batch, rel_pos)
            preds = logits.argmax(dim=-1)
            targets = byte_input[0, positions]

            print(f"Targets: {targets.tolist()}")
            print(f"Preds: {preds.tolist()}")

            correct += (preds == targets).sum().item()
            total += targets.size(0)

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    print(f" Evaluation Complete — Accuracy: {accuracy:.2f}% on {total} tokens.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint (*.pt)")
    parser.add_argument('--modality', type=str, required=True, help="Modality (e.g., text, audio, image, table)")
    parser.add_argument('--data_path', type=str, required=True, help="Root dataset folder")
    args = parser.parse_args()

    evaluate(model_path=args.checkpoint, modality=args.modality, data_path=args.data_path)
