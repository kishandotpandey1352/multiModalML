from models.autoencoder import ByteAutoencoder
from decoders.masked_prediction import MaskedPredictionDecoder
from configurations import config
from encoder.byte_encoder import ByteEncoder

def autoencoder_factory(task):
    from encoder.byte_encoder import ByteEncoder
    from decoders.masked_prediction import MaskedPredictionDecoder
    from models.autoencoder import ByteAutoencoder
    from configurations import config

    encoder = ByteEncoder(
        vocab_size=256,
        embed_dim=config.EMBED_DIM,
        num_layers=config.NUM_LAYERS,
        hidden_dim=config.HIDDEN_DIM,
        num_heads=config.NUM_HEADS,
        dropout=config.DROPOUT,
    )

    if task == "masked_prediction":
        decoder = MaskedPredictionDecoder(config.EMBED_DIM)
        return ByteAutoencoder(encoder, decoder)
    
    elif task == "encoder":
        return encoder  # Return encoder only
    
    else:
        raise ValueError(f"Unknown decoder task: {task}")
