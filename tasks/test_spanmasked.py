import torch
import os
from encoder.byte_encoder import ByteEncoder
from decoders.span_boundary_decoder import SpanBoundaryDecoder
from utility.span_masking import span_mask_input
from configurations import config

DEVICE = config.DEVICE
MODALITY_TO_INDEX = {'text': 0, 'audio': 1, 'image': 2, 'table': 3}
test_folder = "dataset/table"

# Initialize models
encoder = ByteEncoder(config).to(DEVICE)
decoder = SpanBoundaryDecoder(config).to(DEVICE)

checkpoint = torch.load(config.TEST_MODEL_PATH, map_location=DEVICE)
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])
encoder.eval()
decoder.eval()

total = 0
correct = 0

print(f"\nEvaluating all text files in: {test_folder}\n")

for filename in os.listdir(test_folder):
    if not filename.endswith(".txt"):
        continue

    filepath = os.path.join(test_folder, filename)
    modality_name = os.path.basename(os.path.dirname(filepath))
    mod_index = torch.tensor([MODALITY_TO_INDEX[modality_name]], dtype=torch.long).to(DEVICE)

    with open(filepath, 'rb') as f:
        byte_data = f.read()

    byte_tensor = torch.tensor(list(byte_data), dtype=torch.long)[:config.SEQ_LEN]
    input_ids = torch.full((config.SEQ_LEN,), 255, dtype=torch.long)
    input_ids[:len(byte_tensor)] = byte_tensor
    input_ids = input_ids.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        masked, labels = span_mask_input(
            input_ids,
            mask_prob=config.MASK_PROB,
            max_span_length=config.MASK_SPAN_LENGTH
        )
        masked = masked.to(DEVICE)
        labels = labels.to(DEVICE)

        encoded = encoder(masked, mod_index)  # (1, seq_len, D)
        span_mask = labels != -100
        positions = span_mask.squeeze().nonzero(as_tuple=True)[0]

        if len(positions) == 0:
            continue
        
        positions = positions[(positions > 0) & (positions < config.SEQ_LEN - 1)]
        left_boundary = encoded[0, positions - 1]
        right_boundary = encoded[0, positions + 1]
        rel_pos = torch.zeros_like(positions).to(DEVICE)

        logits = decoder(left_boundary, right_boundary, rel_pos)
        preds = torch.argmax(logits, dim=-1)

        actual_bytes = labels[0, positions].tolist()
        predicted_bytes = preds.tolist()

        print(f"\n File: {filename}")
        print("Position | Actual | Predicted | Correct?")
        print("----------------------------------------")
        for i, pos in enumerate(positions.tolist()):
            a, p = actual_bytes[i], predicted_bytes[i]
            total += 1
            is_correct = (a == p)
            correct += is_correct
            print(f"{pos:8d} | {chr(a) if a != 255 else '?':6} | {chr(p) if p != 255 else '?':9} | {'Correct' if is_correct else 'Incorrect'}")

# Final Accuracy
print("\n Evaluation Summary:")
print(f"Total Masked Tokens: {total}")
print(f"Correct Predictions: {correct}")
acc = (correct / total * 100) if total > 0 else 0.0
print(f"Accuracy @ Masked Positions: {acc:.2f}%")
