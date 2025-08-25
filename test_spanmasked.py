import os
import torch
import argparse
from tqdm import tqdm
from loader.multiModal_dataloader import MultiModalDataset, MODALITY_TO_INDEX
from torch.utils.data import DataLoader
from encoder.byte_encoder import ByteEncoder
from decoders.span_boundary_decoder import SpanBoundaryDecoder
from utility.span_masking import span_mask_input
import configurations.config as config

def evaluate(model_path, modality, data_path):
    # Validate modality
    config.MODALITY = modality
    if modality not in MODALITY_TO_INDEX:
        raise ValueError(f"Unsupported modality '{modality}'. Supported: {list(MODALITY_TO_INDEX.keys())}")

    # Load model
    print(f" Loading model from: {model_path}")
    encoder = ByteEncoder(config).to(config.DEVICE)
    decoder = SpanBoundaryDecoder(config).to(config.DEVICE)
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    encoder.eval()
    decoder.eval()

    # Prepare dataset
    dataset = MultiModalDataset(data_path, modality=modality, split='test', from_classifier=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)

    correct, total = 0, 0

    print(f" Evaluating {len(dataset)} file(s) from '{data_path}' with modality '{modality}'...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = batch[0]  # batch size is 1
            byte_input = batch['byte_input'].unsqueeze(0).to(config.DEVICE)
            mod_index = batch['modality_index'].unsqueeze(0).to(config.DEVICE)

            masked, labels = span_mask_input(byte_input, mask_prob=config.MASK_PROB, max_span_length=config.MASK_SPAN_LENGTH)
            encoded = encoder(masked, mod_index)

            positions = (labels[0] != config.IGNORE_INDEX).nonzero(as_tuple=True)[0]
            if len(positions) == 0:
                continue

            positions = positions[(positions > 0) & (positions < config.SEQ_LEN - 1)]
            MAX_SPANS = 20
            if len(positions) > MAX_SPANS:
                positions = positions[:MAX_SPANS]

            left_batch = encoded[0, positions - 1]
            right_batch = encoded[0, positions + 1]

            rel_pos = torch.zeros(left_batch.size(0), dtype=torch.long, device=left_batch.device)
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
    parser.add_argument('--modality', type=str, default=config['modality'])
    parser.add_argument('--data_path', type=str, default=config['data_path'])
    parser.add_argument('--checkpoint', type=str, default=config['checkpoint_path'])
    args = parser.parse_args()

    modality = args.modality
    ckpt_path = args.checkpoint
    full_dataset = MultiModalDataset(data_path=args.data_path, modality=modality, split='train')
    evaluate(model_path=args.checkpoint, modality=args.modality, data_path=args.data_path)
