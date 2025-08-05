import torch
import random

def span_mask_input(input_tensor, mask_prob=0.6, max_span_length=10):
    import torch
    import random

    masked = input_tensor.clone()
    labels = torch.full_like(input_tensor, fill_value=-100)
    batch_size, seq_len = input_tensor.size()

    for b in range(batch_size):
        num_masked = 0
        attempts = 0
        while num_masked == 0 and attempts < 10:
            temp_masked = masked[b].clone()
            temp_labels = labels[b].clone()
            i = 0
            while i < seq_len:
                if random.random() < mask_prob:
                    span_len = random.randint(1, max_span_length)
                    span_end = min(i + span_len, seq_len)
                    temp_labels[i:span_end] = input_tensor[b, i:span_end]
                    temp_masked[i:span_end] = 0
                    i = span_end
                else:
                    i += 1
            num_masked = (temp_labels != -100).sum().item()
            attempts += 1

        masked[b] = temp_masked
        labels[b] = temp_labels

    return masked, labels

