# SpeechLM

A PyTorch implementation of a Speech Language Model that combines any audio encoder with any decoder-only LLM for speech-to-text and audio understanding tasks.

## Overview

SpeechLM bridges pretrained audio encoders (e.g., Wav2Vec2, MMS) with decoder-only language models (e.g., Qwen, Llama) through a learned projection layer. This architecture enables the LLM to process audio features alongside text tokens, allowing for seamless speech-to-text generation.

### Architecture

```
Audio Input → Audio Encoder → Projection Layer → [Audio Features | Text Embeddings] → Decoder LLM → Text Output
```

Key components:
- **Audio Encoder**: Extracts features from raw audio (supports Wav2Vec2, MMS, Whisper-style encoders)
- **Projection Layer**: Multi-layer perceptron that maps encoder hidden states to decoder embedding space
- **Decoder LLM**: Generates text conditioned on audio features
- **Special Tokens**: Learned audio start/end tokens mark audio boundaries

## Installation

```bash
pip install torch transformers
```

## Quick Start

```python
from main import create_speech_lm_model

# Create a model combining MMS encoder and Qwen decoder
feature_extractor, tokenizer, model = create_speech_lm_model(
    encoder_name="facebook/mms-300m",
    decoder_name="Qwen/Qwen3-0.6B",
)

# Generate text from audio
audio_input = ...  # Load audio waveform
inputs = feature_extractor(audio_input, return_tensors="pt")

# Generate
output_ids = model.generate(
    input_values=inputs.input_values,
    max_length=100,
    temperature=0.7
)
text = tokenizer.decode(output_ids[0])
```

## Usage

### Creating a Model

```python
from main import create_speech_lm_model

# Basic usage
feature_extractor, tokenizer, model = create_speech_lm_model(
    encoder_name="facebook/wav2vec2-base",
    decoder_name="Qwen/Qwen2.5-0.5B-Instruct",
)

# With custom projection size and dropout
feature_extractor, tokenizer, model = create_speech_lm_model(
    encoder_name="facebook/mms-300m",
    decoder_name="Qwen/Qwen3-0.6B",
    projection_hidden_size=1024,  # Intermediate projection size
    encoder_dropout=0.1,          # Dropout on encoder features
)
```

### Training Setup

Freeze specific components during training:

```python
# Freeze encoder (common for transfer learning)
model.freeze_encoder()

# Freeze decoder (train only projection)
model.freeze_decoder()

# Freeze everything except LM head
model.freeze_decoder_except_lm_head()

# Freeze projection layers only
model.freeze_projection()
```

### Forward Pass

```python
import torch

# Audio + text input
outputs = model(
    input_values=audio_waveform,      # Audio for Wav2Vec2-style encoders
    input_ids=text_token_ids,          # Text tokens
    attention_mask=audio_attention_mask,
    decoder_attention_mask=text_attention_mask,
    labels=labels,                     # For training (loss computed on text only)
)

loss = outputs.loss
logits = outputs.logits
```

### Generation

```python
# From audio only
output_ids = model.generate(
    input_values=audio_input,
    max_length=200,
    num_beams=4,
    temperature=0.8,
)

# From audio + prompt text
output_ids = model.generate(
    input_values=audio_input,
    input_ids=prompt_token_ids,
    max_length=200,
)

text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

### Saving and Loading

```python
# Save to Hugging Face Hub
model.push_to_hub("username/speech-lm-model")
feature_extractor.push_to_hub("username/speech-lm-model")
tokenizer.push_to_hub("username/speech-lm-model")

# Save locally
model.save_pretrained("./my-model")

# Load from pretrained
from model import SpeechLMModel
model = SpeechLMModel.from_pretrained("username/speech-lm-model")
```

## File Structure

```
speechlm/
├── config.py    # SpeechLMConfig - model configuration
├── model.py     # SpeechLMModel - core model implementation
└── main.py      # create_speech_lm_model() factory function
```

### Key Classes

- **`SpeechLMConfig`** (`config.py`): Configuration class combining encoder and decoder configs
- **`SpeechLMModel`** (`model.py`): Main model class inheriting from `PreTrainedModel` and `GenerationMixin`

## Supported Encoders

- Wav2Vec2 (e.g., `facebook/wav2vec2-base`, `facebook/wav2vec2-large`)
- MMS (e.g., `facebook/mms-300m`, `facebook/mms-1b`)
- Whisper-style encoders
- Any Hugging Face `AutoModel` compatible audio encoder

## Supported Decoders

- Qwen (e.g., `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen3-0.6B`)
- Llama (e.g., `meta-llama/Llama-2-7b`)
- Mistral
- GPT-2
- Any Hugging Face `AutoModelForCausalLM` compatible model

## Features

- **Modular Design**: Mix and match any encoder with any decoder
- **Flexible Input**: Supports both `input_values` (Wav2Vec2) and `input_features` (Whisper)
- **Efficient Training**: Gradient checkpointing support, selective freezing
- **Generation Ready**: Full support for Hugging Face generation methods (beam search, sampling, etc.)
- **Attention Masking**: Proper handling of audio sequence length reduction from convolutional encoders
- **Label Padding**: Ignores audio tokens in loss computation (padded with -100)

## Example: Full Training Loop

```python
from transformers import AdamW
from torch.utils.data import DataLoader

# Setup
feature_extractor, tokenizer, model = create_speech_lm_model(
    encoder_name="facebook/mms-300m",
    decoder_name="Qwen/Qwen2.5-0.5B-Instruct",
)
model.freeze_encoder()  # Freeze pretrained encoder

optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
model.train()
for batch in dataloader:
    optimizer.zero_grad()

    outputs = model(
        input_values=batch["input_values"],
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )

    loss = outputs.loss
    loss.backward()
    optimizer.step()
```

## License

MIT License - See LICENSE file for details.
