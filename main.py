from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoFeatureExtractor,   
)
from typing import Optional
from config import SpeechLMConfig
from model import SpeechLMModel

def create_speech_lm_model(
    encoder_name: str,
    decoder_name: str,
    projection_hidden_size: Optional[int] = None,
    encoder_dropout: float = 0.0,
) -> SpeechLMModel:
    """
    Create a Speech LM model from pretrained encoder and decoder.
    
    Args:
        encoder_name: HuggingFace model name for audio encoder
        decoder_name: HuggingFace model name for decoder-only LLM
        projection_hidden_size: Hidden size for projection (default: decoder hidden size)
        encoder_dropout: Dropout rate for encoder features
    
    Returns:
        SpeechLMModel instance
    
    Examples:
        >>> model = create_speech_lm_model(
        ...     encoder_name="facebook/wav2vec2-base",
        ...     decoder_name="Qwen/Qwen2.5-0.5B-Instruct"
        ... )
    """
    print(f"Loading encoder: {encoder_name}")
    print(f"Loading decoder: {decoder_name}")
    
    # Create config
    config = SpeechLMConfig(
        encoder_name_or_path=encoder_name,
        decoder_name_or_path=decoder_name,
        projection_hidden_size=projection_hidden_size,
        encoder_dropout=encoder_dropout,
    )
    
    # Create model
    model = SpeechLMModel(config)

    feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_name)
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)

    # Load pretrained weights
    print("Loading pretrained encoder weights...")
    pretrained_encoder = AutoModel.from_pretrained(encoder_name)
    model.encoder = pretrained_encoder
    
    print("Loading pretrained decoder weights...")
    pretrained_decoder = AutoModelForCausalLM.from_pretrained(decoder_name)
    model.decoder = pretrained_decoder
    
    print("✓ Model created successfully!")
    return feature_extractor, tokenizer, model


if __name__ == "__main__":
    
    feature_extractor, tokenizer, model = create_speech_lm_model(
        encoder_name="facebook/mms-300m",
        decoder_name="Qwen/Qwen3-0.6B",
    )
    
    # Print model info
    print("\nModel Information:")
    print(f"  Encoder: {model.config.encoder_name_or_path}")
    print(f"  Decoder: {model.config.decoder_name_or_path}")
    print(f"  Encoder hidden size: {model.config.encoder.hidden_size}")
    print(f"  Decoder hidden size: {model.config.decoder.hidden_size}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    projection_params = total_params - encoder_params - decoder_params
    
    print(f"\nParameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Encoder: {encoder_params:,}")
    print(f"  Decoder: {decoder_params:,}")
    print(f"  Projection: {projection_params:,}")

    model_card = f"harikc456/speech_lm_mms_qwen3-{round(total_params/10e8, 1)}B"
    # print(model_card)
    
    model.push_to_hub(model_card)
    feature_extractor.push_to_hub(model_card)
    tokenizer.push_to_hub(model_card)