from models.autoencoder import ByteAutoencoder
from decoders.masked_prediction import MaskedPredictionDecoder
from configurations import config
from encoder.byte_encoder import ByteEncoder

def autoencoder_factory(task="masked"):
    """
    Factory function to build autoencoder model for different tasks.
    Uses configuration from config.py.
    """
    
    encoder = ByteEncoder(
            embed_dim=config.EMBED_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            num_heads=config.NUM_HEADS,
            dropout=config.DROPOUT
        )

    if task == "masked":
        decoder = MaskedPredictionDecoder(
            embed_dim=config.EMBED_DIM,
            vocab_size=256
        )
    else:
        raise ValueError(f"Unknown decoder task: {task}")

    return ByteAutoencoder(encoder, decoder)

