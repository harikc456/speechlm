import torch
import torch.nn as nn
from transformers import (
    PreTrainedModel,
    AutoModel,
    AutoModelForCausalLM,
    GenerationMixin
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, Union
from config import SpeechLMConfig

class SpeechLMModel(PreTrainedModel, GenerationMixin):
    """
    Speech Language Model combining audio encoder with decoder-only LLM.
    
    This model concatenates encoded audio features with text token embeddings,
    allowing the decoder-only LLM to process both modalities.
    
    Architecture:
        1. Audio → Encoder → Hidden States
        2. Hidden States → Projection → Normalized Features
        3. Text → Decoder Embeddings
        4. [Audio Features | Text Embeddings] → Decoder → Output
    """
    config_class = SpeechLMConfig
    base_model_prefix = "speech_lm"
    supports_gradient_checkpointing = True
    
    def __init__(self, config: SpeechLMConfig):
        super().__init__(config)
        self.config = config
        
        # Initialize encoder (audio)
        self.encoder = AutoModel.from_config(config.encoder)
        
        # Initialize decoder (LLM)
        self.decoder = AutoModelForCausalLM.from_config(config.decoder)
        
        # Projection layers
        encoder_hidden_size = config.encoder.hidden_size
        decoder_hidden_size = config.decoder.hidden_size
        projection_size = config.projection_hidden_size or decoder_hidden_size
        
        # Multi-layer projection for better representation learning
        self.encoder_projection = nn.Sequential(
            nn.Linear(encoder_hidden_size, projection_size, bias=False),
            nn.GELU(),
            nn.Linear(projection_size, decoder_hidden_size, bias=False),
        )
        
        # Layer normalization
        self.encoder_layer_norm = nn.LayerNorm(decoder_hidden_size, eps=1e-5)
        
        # Optional dropout
        self.encoder_dropout = nn.Dropout(config.encoder_dropout) if config.encoder_dropout > 0 else None
        
        # Special tokens for audio boundaries
        # We'll add these to help the model distinguish audio from text
        self.audio_start_token = nn.Parameter(torch.randn(1, 1, decoder_hidden_size))
        self.audio_end_token = nn.Parameter(torch.randn(1, 1, decoder_hidden_size))
        
        # Initialize weights
        self.post_init()
    
    def _init_weights(self, module):
        """Initialize weights for projection layers."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder
    
    def freeze_encoder(self):
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("✓ Encoder frozen")
    
    def freeze_decoder(self):
        """Freeze all decoder parameters."""
        for param in self.decoder.parameters():
            param.requires_grad = False
        print("✓ Decoder frozen")
    
    def freeze_decoder_except_lm_head(self):
        """Freeze decoder except the language model head."""
        for name, param in self.decoder.named_parameters():
            if 'lm_head' not in name:
                param.requires_grad = False
        print("✓ Decoder frozen (except lm_head)")
    
    def freeze_projection(self):
        """Freeze projection layers."""
        for param in self.encoder_projection.parameters():
            param.requires_grad = False
        for param in self.encoder_layer_norm.parameters():
            param.requires_grad = False
        self.audio_start_token.requires_grad = False
        self.audio_end_token.requires_grad = False
        print("✓ Projection layers frozen")
    
    def get_audio_features(
        self,
        input_values: Optional[torch.FloatTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        """
        Extract and project audio features from encoder.
        
        Returns:
            Projected audio features of shape (batch_size, audio_seq_len + 2, hidden_size)
            The +2 is for start and end tokens.
        """
        # Prepare encoder inputs
        encoder_inputs = {
            'output_attentions': output_attentions,
            'output_hidden_states': output_hidden_states,
            'return_dict': True,
        }
        
        # Handle different input types
        if input_values is not None:
            encoder_inputs['input_values'] = input_values
        elif input_features is not None:
            encoder_inputs['input_features'] = input_features
        else:
            raise ValueError("Must provide either input_values or input_features")
        
        if attention_mask is not None:
            encoder_inputs['attention_mask'] = attention_mask
        
        # Encode audio
        encoder_outputs = self.encoder(**encoder_inputs)
        hidden_states = encoder_outputs.last_hidden_state
        
        # Project to decoder dimension
        projected = self.encoder_projection(hidden_states)
        
        # Apply layer norm
        projected = self.encoder_layer_norm(projected)
        
        # Apply dropout if configured
        if self.encoder_dropout is not None:
            projected = self.encoder_dropout(projected)
        
        # Add start and end tokens
        batch_size = projected.shape[0]
        start_tokens = self.audio_start_token.expand(batch_size, -1, -1)
        end_tokens = self.audio_end_token.expand(batch_size, -1, -1)
        
        # Concatenate: [START] + audio_features + [END]
        audio_features = torch.cat([start_tokens, projected, end_tokens], dim=1)
        
        return audio_features
    
    def _get_audio_attention_mask(
        self,
        audio_features: torch.FloatTensor,
        input_attention_mask: Optional[torch.LongTensor] = None,
    ) -> torch.LongTensor:
        """
        Create attention mask for audio features.
        Handles sequence length reduction from encoders like Wav2Vec2.
        """
        batch_size, audio_seq_len, _ = audio_features.shape
        
        if input_attention_mask is None:
            # No masking - attend to all audio
            return torch.ones(
                (batch_size, audio_seq_len),
                dtype=torch.long,
                device=audio_features.device
            )
        
        # Check if encoder reduces sequence length (Wav2Vec2, etc.)
        if hasattr(self.config.encoder, 'conv_stride'):
            # Calculate output lengths after convolutions
            input_lengths = input_attention_mask.sum(-1)
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)
            
            audio_attention_mask = torch.zeros(
                (batch_size, audio_seq_len),
                dtype=torch.long,
                device=audio_features.device
            )
            
            # +2 for start and end tokens
            for i, length in enumerate(output_lengths):
                audio_attention_mask[i, :length + 2] = 1
            
            return audio_attention_mask
        else:
            # No sequence reduction - expand mask for start/end tokens
            # Assume all audio features are valid
            return torch.ones(
                (batch_size, audio_seq_len),
                dtype=torch.long,
                device=audio_features.device
            )
    
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Compute output lengths after convolutional feature extraction.
        For Wav2Vec2-style encoders.
        """
        def _conv_out_length(input_length, kernel_size, stride):
            return (input_length - kernel_size) // stride + 1
        
        if hasattr(self.config.encoder, 'conv_kernel'):
            for kernel_size, stride in zip(
                self.config.encoder.conv_kernel,
                self.config.encoder.conv_stride,
            ):
                input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
        
        return input_lengths
    
    def forward(
        self,
        input_values: Optional[torch.FloatTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get audio features if provided
        if input_values is not None or input_features is not None:
            audio_features = self.get_audio_features(
                input_values=input_values,
                input_features=input_features,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            
            audio_attention_mask = self._get_audio_attention_mask(
                audio_features,
                attention_mask,
            )
        else:
            audio_features = None
            audio_attention_mask = None
        
        # Robust way to get embed_tokens (works with PEFT/LoRA wrappers)
        embed_tokens = self.decoder.get_input_embeddings()
        
        # Get text embeddings if provided
        text_embeddings = None
        if input_ids is not None:
            text_embeddings = embed_tokens(input_ids)
        
        # Handle labels and combine audio/text
        if audio_features is not None and text_embeddings is not None:
            # Standard case: audio + text prefix
            inputs_embeds = torch.cat([audio_features, text_embeddings], dim=1)
            
            # Concatenate attention masks
            if decoder_attention_mask is None:
                decoder_attention_mask = torch.ones(
                    text_embeddings.shape[:2],
                    dtype=torch.long,
                    device=text_embeddings.device
                )
            
            combined_attention_mask = torch.cat(
                [audio_attention_mask, decoder_attention_mask],
                dim=1
            )
        elif audio_features is not None and input_ids is None and labels is not None:
            # New case: audio-only prompt, use labels for teacher-forcing (continuation/transcription)
            batch_size = audio_features.shape[0]
            # Embed the labels (full sequence)
            text_embeddings = embed_tokens(labels)
            
            # Shift for teacher-forcing: use embeds[:-1]
            shifted_text_embeddings = text_embeddings[:, :-1, :]
            
            # Concat to audio
            inputs_embeds = torch.cat([audio_features, shifted_text_embeddings], dim=1)
            
            # Attention mask for shifted text
            shifted_text_mask = torch.ones(
                (batch_size, labels.shape[1] - 1),
                dtype=torch.long,
                device=labels.device
            )
            
            combined_attention_mask = torch.cat(
                [audio_attention_mask, shifted_text_mask],
                dim=1
            )
        elif audio_features is not None:
            # Audio only (no text or labels)
            inputs_embeds = audio_features
            combined_attention_mask = audio_attention_mask
        elif text_embeddings is not None:
            # Text only
            inputs_embeds = text_embeddings
            combined_attention_mask = decoder_attention_mask
        else:
            raise ValueError("Must provide either audio or text inputs")
        
        # Prepare labels if provided
        if labels is not None and audio_features is not None:
            audio_seq_len = audio_features.shape[1]
            audio_labels = torch.full(
                (labels.shape[0], audio_seq_len),
                -100,
                dtype=labels.dtype,
                device=labels.device
            )
            if input_ids is None:
                # For continuation: cat with labels[1:]
                shifted_labels = labels[:, 1:]
                labels = torch.cat([audio_labels, shifted_labels], dim=1)
            else:
                # Standard: cat with full labels
                labels = torch.cat([audio_labels, labels], dim=1)
        
        # Forward through decoder
        decoder_outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        return decoder_outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        """Prepare inputs for generation."""
        # If past_key_values is used, we only need the last token
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        
        # If inputs_embeds are passed, we only want to use them on the first pass
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        
        model_inputs.update({
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        })
        
        return model_inputs
    
    @torch.no_grad()
    def generate(
        self,
        input_values: Optional[torch.FloatTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs
    ):
        """
        Generate text from audio input.
        
        Args:
            input_values: Audio input (Wav2Vec2-style)
            input_features: Audio input (Whisper-style)
            attention_mask: Audio attention mask
            **generate_kwargs: Arguments for generation (max_length, temperature, etc.)
        """
        # Encode audio
        audio_features = self.get_audio_features(
            input_values=input_values,
            input_features=input_features,
            attention_mask=attention_mask,
        )
        
        audio_attention_mask = self._get_audio_attention_mask(
            audio_features,
            attention_mask,
        )
        
        # Generate using decoder
        return self.decoder.generate(
            inputs_embeds=audio_features,
            attention_mask=audio_attention_mask,
            **generate_kwargs
        )
    
    def save_pretrained(self, save_directory, **kwargs):
        """Save model and configuration."""
        super().save_pretrained(save_directory, **kwargs)
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load pretrained model."""
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

