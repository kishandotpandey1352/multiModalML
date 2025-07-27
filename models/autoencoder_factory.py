# autoencoder_factory.py

from encoder.byte_encoder import ByteEncoder
from models.autoencoder import ByteAutoencoder

# Import available decoders
from decoders.masked_prediction import MaskedByteDecoder
# from contrastive_decoder import ContrastiveDecoder
# from classifier import ClassificationDecoder
# from span_prediction import SpanDecoder
# Add more as needed

def autoencoder_factory(task: str, embed_dim=128, vocab_size=256):
    """
    Constructs a ByteAutoencoder based on the task.

    Args:
        task (str): One of ["masked", "contrastive", "classifier", ...]
        embed_dim (int): Embedding dimension (should match encoder + decoder)
        vocab_size (int): Size of vocabulary (e.g., 256 for bytes)

    Returns:
        ByteAutoencoder instance
    """
    encoder = ByteEncoder(embed_dim=embed_dim)

    if task == "masked":
        decoder = MaskedByteDecoder(embed_dim=embed_dim, vocab_size=vocab_size)
    # elif task == "contrastive":
    #     from contrastive_decoder import ContrastiveDecoder
    #     decoder = ContrastiveDecoder(embed_dim=embed_dim)
    # elif task == "classifier":
    #     from classifier import ClassificationDecoder
    #     decoder = ClassificationDecoder(embed_dim=embed_dim, num_classes=vocab_size)
    # elif task == "span":
    #     from span_prediction import SpanDecoder
    #     decoder = SpanDecoder(embed_dim=embed_dim)
    else:
        raise ValueError(f"Unknown task: {task}")

    return ByteAutoencoder(encoder, decoder)
