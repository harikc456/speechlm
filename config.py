from transformers import (
    PretrainedConfig,
    AutoConfig,
)
from typing import Optional, Union, Dict


class SpeechLMConfig(PretrainedConfig):
    """
    Configuration for Speech Language Model (encoder + decoder-only LLM).
    
    This config combines any audio encoder with any decoder-only LLM.
    """
    model_type = "speech_lm"
    is_composition = True
    
    def __init__(
        self,
        encoder_config: Optional[Union[Dict, PretrainedConfig]] = None,
        decoder_config: Optional[Union[Dict, PretrainedConfig]] = None,
        encoder_name_or_path: Optional[str] = None,
        decoder_name_or_path: Optional[str] = None,
        projection_hidden_size: Optional[int] = None,
        use_encoder_pooling: bool = False,
        pooling_mode: str = "mean",  # mean, max, first, last
        encoder_dropout: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Store model paths
        self.encoder_name_or_path = encoder_name_or_path
        self.decoder_name_or_path = decoder_name_or_path
        
        # Handle case where config is instantiated with no arguments (for serialization)
        if encoder_config is None and encoder_name_or_path is None:
            # Create dummy configs - will be replaced when loading
            from transformers import Wav2Vec2Config, LlamaConfig
            self.encoder = Wav2Vec2Config()
            self.decoder = LlamaConfig()
        else:
            # Load encoder config
            if encoder_config is not None:
                if isinstance(encoder_config, dict):
                    self.encoder = AutoConfig.for_model(**encoder_config)
                else:
                    self.encoder = encoder_config
            else:
                self.encoder = AutoConfig.from_pretrained(encoder_name_or_path)
            
            # Load decoder config
            if decoder_config is None and decoder_name_or_path is None:
                raise ValueError("Must provide either decoder_config or decoder_name_or_path")
            
            if decoder_config is not None:
                if isinstance(decoder_config, dict):
                    self.decoder = AutoConfig.for_model(**decoder_config)
                else:
                    self.decoder = decoder_config
            else:
                self.decoder = AutoConfig.from_pretrained(decoder_name_or_path)
        
        # Projection settings
        self.projection_hidden_size = projection_hidden_size
        self.use_encoder_pooling = use_encoder_pooling
        self.pooling_mode = pooling_mode
        self.encoder_dropout = encoder_dropout
        
    def to_dict(self):
        """Serialize config to dictionary."""
        output = super().to_dict()
        output["encoder_config"] = self.encoder.to_dict() if hasattr(self, 'encoder') else {}
        output["decoder_config"] = self.decoder.to_dict() if hasattr(self, 'decoder') else {}
        output["encoder_name_or_path"] = self.encoder_name_or_path
        output["decoder_name_or_path"] = self.decoder_name_or_path
        output["projection_hidden_size"] = self.projection_hidden_size
        output["use_encoder_pooling"] = self.use_encoder_pooling
        output["pooling_mode"] = self.pooling_mode
        output["encoder_dropout"] = self.encoder_dropout
        return output
    
    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """Create config from dictionary."""
        # Extract encoder and decoder configs
        encoder_config = config_dict.pop("encoder_config", None)
        decoder_config = config_dict.pop("decoder_config", None)
        
        # Convert to Config objects if they're dicts
        if encoder_config is not None and isinstance(encoder_config, dict):
            encoder_config = AutoConfig.for_model(**encoder_config)
        if decoder_config is not None and isinstance(decoder_config, dict):
            decoder_config = AutoConfig.for_model(**decoder_config)
        
        # Create config
        return cls(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            **config_dict,
            **kwargs
        )
