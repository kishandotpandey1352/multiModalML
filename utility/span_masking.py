import torch
import random

def span_mask_input(input_tensor, mask_prob=0.15, max_span_length=5, mask_token=255):
    """
    Mask contiguous spans of input tokens for SpanBERT-style pretraining.
    """
    input_tensor = input_tensor.clone()
    labels = torch.full_like(input_tensor, -100)
    batch_size, seq_len = input_tensor.size()

    for b in range(batch_size):
        num_mask = int(mask_prob * seq_len)
        masked = 0
        while masked < num_mask:
            span_length = min(random.randint(1, max_span_length), seq_len - 1)
            start = random.randint(0, seq_len - span_length)
            if (labels[b, start:start+span_length] != -100).any():
                continue  # overlap, skip

            labels[b, start:start+span_length] = input_tensor[b, start:start+span_length]
            input_tensor[b, start:start+span_length] = mask_token
            masked += span_length

    return input_tensor, labels
