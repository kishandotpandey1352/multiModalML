import os
import numpy as np
import random

def vectorize_file_bytes(filepath, dim=128, chunk_size=512):
    with open(filepath, "rb") as f:
        data = f.read()
    
    if len(data) < chunk_size:
        # Pad with zeros if file too small
        chunk = data + bytes(chunk_size - len(data))
    else:
        start = random.randint(0, len(data) - chunk_size)
        chunk = data[start:start + chunk_size]
    
    # Convert bytes to int values [0, 255] and normalize
    vector = np.frombuffer(chunk, dtype=np.uint8).astype(np.float32)
    if len(vector) < dim:
        vector = np.pad(vector, (0, dim - len(vector)))
    elif len(vector) > dim:
        vector = vector[:dim]
    
    return vector
