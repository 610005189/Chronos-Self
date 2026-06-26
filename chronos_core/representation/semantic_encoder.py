"""
Semantic Intent Encoder for Chronos-Self
==========================================

This module implements the semantic intent encoder that processes symbolic inputs
(text or other symbolic representations) and produces semantic intent vectors.

Components:
- SemanticEncoder: Main transformer-based encoder
- SentimentExtractor: Extracts emotional polarity and intensity
- IntentExtractor: Extracts pragmatic intents and goals
- IntentVector: Combined output representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from loguru import logger
import math

from chronos_core.utils.config import EncoderConfig


@dataclass
class IntentVector:
    """
    Output intent vector containing semantic, sentiment, and pragmatic information.

    Attributes:
        semantic_vector: Main semantic representation (dim: output_dim)
        sentiment_polarity: Sentiment polarity (-1.0 to 1.0)
        sentiment_intensity: Sentiment strength (0.0 to 1.0)
        intent_type: Pragmatic intent type (inform/request/promise/question, etc.)
        intent_confidence: Confidence score for intent classification (0.0 to 1.0)
        goal_vector: Goal representation (dim: output_dim // 2)
        combined_vector: Fused vector combining all components (dim: output_dim)
    """
    semantic_vector: torch.Tensor
    sentiment_polarity: float
    sentiment_intensity: float
    intent_type: str
    intent_confidence: float
    goal_vector: torch.Tensor
    combined_vector: torch.Tensor

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'semantic_vector': self.semantic_vector.cpu().numpy(),
            'sentiment_polarity': self.sentiment_polarity,
            'sentiment_intensity': self.sentiment_intensity,
            'intent_type': self.intent_type,
            'intent_confidence': self.intent_confidence,
            'goal_vector': self.goal_vector.cpu().numpy(),
            'combined_vector': self.combined_vector.cpu().numpy()
        }


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer inputs.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Output tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LightweightTransformerEncoder(nn.Module):
    """
    Lightweight Transformer encoder with 2-4 layers.

    This is a custom implementation optimized for semantic encoding tasks.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True  # Pre-norm for better gradient flow
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim)
        )

        # Output projection for attention pooling
        self.attention_proj = nn.Linear(hidden_dim, 1)

        logger.info(f"Initialized LightweightTransformerEncoder: {num_layers} layers, {num_heads} heads, hidden_dim={hidden_dim}")

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the transformer encoder.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            attention_mask: Optional mask of shape (batch_size, seq_len)

        Returns:
            encoded_output: Encoded sequence (batch_size, seq_len, hidden_dim)
            pooled_output: Pooled representation (batch_size, hidden_dim)
        """
        # Add positional encoding
        x = self.pos_encoder(x)

        # Create padding mask for transformer
        src_key_padding_mask = None
        if attention_mask is not None:
            # Convert attention mask to padding mask (True = masked position)
            src_key_padding_mask = (attention_mask == 0)

        # Pass through transformer
        encoded_output = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )

        # Attention-based pooling
        attention_scores = self.attention_proj(encoded_output).squeeze(-1)  # (batch_size, seq_len)

        if attention_mask is not None:
            # Mask out padding positions
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))

        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, seq_len)
        pooled_output = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch_size, 1, seq_len)
            encoded_output  # (batch_size, seq_len, hidden_dim)
        ).squeeze(1)  # (batch_size, hidden_dim)

        return encoded_output, pooled_output


class SentimentExtractor(nn.Module):
    """
    Extracts emotional polarity and intensity from semantic representations.

    This module analyzes the semantic vector to determine:
    - Sentiment polarity: negative (-1.0) to positive (1.0)
    - Sentiment intensity: weak (0.0) to strong (1.0)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Sentiment polarity classifier
        self.polarity_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Output in [-1, 1]
        )

        # Sentiment intensity regressor
        self.intensity_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

        # Sentiment embedding for combined vector
        self.sentiment_embedding = nn.Linear(2, input_dim // 4)

        logger.info(f"Initialized SentimentExtractor: input_dim={input_dim}, hidden_dim={hidden_dim}")

    def forward(self, semantic_vector: torch.Tensor) -> Tuple[float, float, torch.Tensor]:
        """
        Extract sentiment information from semantic vector.

        Args:
            semantic_vector: Input semantic representation (batch_size, input_dim)

        Returns:
            polarity: Sentiment polarity score (-1.0 to 1.0)
            intensity: Sentiment intensity score (0.0 to 1.0)
            sentiment_embedding: Embedded sentiment features (batch_size, input_dim // 4)
        """
        # Ensure input is valid
        if torch.isnan(semantic_vector).any() or torch.isinf(semantic_vector).any():
            logger.warning("Detected NaN or Inf in semantic vector, applying clipping")
            semantic_vector = torch.nan_to_num(semantic_vector, nan=0.0, posinf=1.0, neginf=-1.0)

        # Predict polarity and intensity
        polarity = self.polarity_net(semantic_vector).squeeze(-1)  # (batch_size,)
        intensity = self.intensity_net(semantic_vector).squeeze(-1)  # (batch_size,)

        # Combine into sentiment features
        sentiment_features = torch.stack([polarity, intensity], dim=-1)  # (batch_size, 2)
        sentiment_embedding = self.sentiment_embedding(sentiment_features)  # (batch_size, input_dim // 4)

        # Extract scalar values (take mean over batch for aggregation)
        polarity_value = polarity.mean().item()
        intensity_value = intensity.mean().item()

        return polarity_value, intensity_value, sentiment_embedding


class IntentExtractor(nn.Module):
    """
    Extracts pragmatic intent and goals from semantic representations.

    This module identifies:
    - Intent type: inform, request, promise, question, etc.
    - Intent confidence: confidence score for the classification
    - Goal vector: representation of the underlying goal
    """

    # Intent type definitions
    INTENT_TYPES = [
        'inform',      # Providing information
        'request',     # Asking for something
        'promise',     # Committing to an action
        'question',    # Asking a question
        'command',     # Direct instruction
        'acknowledge', # Acknowledging receipt
        'apologize',   # Expressing apology
        'greet',       # Social greeting
        'other'        # Catch-all category
    ]

    def __init__(
        self,
        input_dim: int,
        num_intent_types: int = None,
        hidden_dim: int = 256,
        goal_dim: int = 128
    ):
        super().__init__()

        if num_intent_types is None:
            num_intent_types = len(self.INTENT_TYPES)

        self.num_intent_types = num_intent_types
        self.goal_dim = goal_dim

        # Intent type classifier
        self.intent_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_intent_types)
        )

        # Goal extraction network
        self.goal_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, goal_dim),
            nn.LayerNorm(goal_dim)
        )

        # Intent embedding for combined vector
        self.intent_embedding = nn.Linear(num_intent_types + goal_dim, input_dim // 4)

        logger.info(f"Initialized IntentExtractor: input_dim={input_dim}, num_intents={num_intent_types}, goal_dim={goal_dim}")

    def forward(
        self,
        semantic_vector: torch.Tensor
    ) -> Tuple[str, float, torch.Tensor, torch.Tensor]:
        """
        Extract intent information from semantic vector.

        Args:
            semantic_vector: Input semantic representation (batch_size, input_dim)

        Returns:
            intent_type: Predicted intent type string
            confidence: Confidence score for the prediction (0.0 to 1.0)
            goal_vector: Extracted goal representation (batch_size, goal_dim)
            intent_embedding: Embedded intent features (batch_size, input_dim // 4)
        """
        # Ensure input is valid
        if torch.isnan(semantic_vector).any() or torch.isinf(semantic_vector).any():
            logger.warning("Detected NaN or Inf in semantic vector, applying clipping")
            semantic_vector = torch.nan_to_num(semantic_vector, nan=0.0, posinf=1.0, neginf=-1.0)

        # Classify intent type
        intent_logits = self.intent_classifier(semantic_vector)  # (batch_size, num_intent_types)
        intent_probs = F.softmax(intent_logits, dim=-1)

        # Get predicted intent
        confidence, predicted_idx = intent_probs.max(dim=-1)
        confidence_value = confidence.mean().item()
        predicted_idx_value = predicted_idx[0].item()  # Use first batch item

        intent_type = self.INTENT_TYPES[predicted_idx_value]

        # Extract goal vector
        goal_vector = self.goal_extractor(semantic_vector)  # (batch_size, goal_dim)

        # Combine intent features
        intent_features = torch.cat([
            intent_probs,
            goal_vector.mean(dim=0, keepdim=True).expand(intent_probs.size(0), -1)
        ], dim=-1)
        intent_embedding = self.intent_embedding(intent_features)  # (batch_size, input_dim // 4)

        return intent_type, confidence_value, goal_vector, intent_embedding


class SemanticEncoder(nn.Module):
    """
    Main semantic intent encoder that combines transformer encoding,
    sentiment extraction, and intent extraction.

    This encoder processes text or symbolic inputs and produces a comprehensive
    intent vector representation suitable for the Chronos-Self system.
    """

    def __init__(
        self,
        config: EncoderConfig,
        output_dim: int = 256,
        device: str = 'cuda'
    ):
        super().__init__()

        self.config = config
        self.output_dim = output_dim
        self.device = device

        # Load pre-trained tokenizer and base model
        logger.info(f"Loading pre-trained model: {config.semantic_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.semantic_model_name)
        self.base_model = AutoModel.from_pretrained(config.semantic_model_name)

        # Get the output dimension of the base model
        base_hidden_dim = self.base_model.config.hidden_size

        # Freeze base model parameters (optional, can be fine-tuned later)
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Projection to hidden dimension
        self.input_projection = nn.Linear(base_hidden_dim, config.semantic_hidden_dim)

        # Lightweight transformer encoder
        self.transformer_encoder = LightweightTransformerEncoder(
            hidden_dim=config.semantic_hidden_dim,
            num_layers=config.semantic_num_layers,
            num_heads=config.semantic_num_heads,
            feedforward_dim=config.semantic_hidden_dim * 4,
            dropout=config.cross_attention_dropout
        )

        # Sentiment extractor
        self.sentiment_extractor = SentimentExtractor(
            input_dim=config.semantic_hidden_dim,
            hidden_dim=config.semantic_hidden_dim // 2
        )

        # Intent extractor
        self.intent_extractor = IntentExtractor(
            input_dim=config.semantic_hidden_dim,
            num_intent_types=len(IntentExtractor.INTENT_TYPES),
            hidden_dim=config.semantic_hidden_dim // 2,
            goal_dim=output_dim // 2
        )

        # Final projection to output dimension
        # Combines: semantic (hidden_dim) + sentiment (hidden_dim//4) + intent (hidden_dim//4)
        combined_dim = config.semantic_hidden_dim + config.semantic_hidden_dim // 4 + config.semantic_hidden_dim // 4
        self.output_projection = nn.Sequential(
            nn.Linear(combined_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )

        # Move to device
        self.to(device)

        logger.info(f"Initialized SemanticEncoder: output_dim={output_dim}, device={device}")
        logger.info(f"Base model: {config.semantic_model_name}, frozen=True")
        logger.info(f"Transformer layers: {config.semantic_num_layers}, heads: {config.semantic_num_heads}")

    def preprocess_text(
        self,
        text: Union[str, List[str]],
        max_length: int = 512
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess text inputs for the encoder.

        Args:
            text: Input text or list of texts
            max_length: Maximum sequence length

        Returns:
            Dictionary containing tokenized inputs
        """
        if isinstance(text, str):
            text = [text]

        # Tokenize
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        # Move to device
        return {k: v.to(self.device) for k, v in encoded.items()}

    def encode_text(
        self,
        text: Union[str, List[str]],
        max_length: int = 512
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text through the base model and transformer encoder.

        Args:
            text: Input text or list of texts
            max_length: Maximum sequence length

        Returns:
            encoded_output: Encoded sequence (batch_size, seq_len, hidden_dim)
            pooled_output: Pooled representation (batch_size, hidden_dim)
        """
        # Preprocess text
        inputs = self.preprocess_text(text, max_length)

        # Pass through base model
        with torch.no_grad():
            base_outputs = self.base_model(**inputs)

        # Get base model output (last hidden state)
        base_hidden = base_outputs.last_hidden_state  # (batch_size, seq_len, base_hidden_dim)

        # Project to hidden dimension
        projected = self.input_projection(base_hidden)  # (batch_size, seq_len, hidden_dim)

        # Pass through transformer encoder
        attention_mask = inputs.get('attention_mask', None)
        encoded_output, pooled_output = self.transformer_encoder(projected, attention_mask)

        return encoded_output, pooled_output

    def forward(
        self,
        text: Union[str, List[str]],
        max_length: int = 512
    ) -> IntentVector:
        """
        Full forward pass through the semantic encoder.

        Args:
            text: Input text or list of texts
            max_length: Maximum sequence length

        Returns:
            IntentVector containing semantic, sentiment, and intent information
        """
        # Encode text
        _, pooled_output = self.encode_text(text, max_length)

        # Numerical stability check
        if torch.isnan(pooled_output).any() or torch.isinf(pooled_output).any():
            logger.warning("Detected NaN or Inf in pooled output, applying clipping")
            pooled_output = torch.nan_to_num(pooled_output, nan=0.0, posinf=1.0, neginf=-1.0)

        # Extract sentiment
        sentiment_polarity, sentiment_intensity, sentiment_embedding = \
            self.sentiment_extractor(pooled_output)

        # Extract intent
        intent_type, intent_confidence, goal_vector, intent_embedding = \
            self.intent_extractor(pooled_output)

        # Combine all representations
        combined_features = torch.cat([
            pooled_output,
            sentiment_embedding,
            intent_embedding
        ], dim=-1)

        # Project to output dimension
        combined_vector = self.output_projection(combined_features)

        # Numerical stability check for final output
        if torch.isnan(combined_vector).any() or torch.isinf(combined_vector).any():
            logger.warning("Detected NaN or Inf in combined vector, applying clipping")
            combined_vector = torch.nan_to_num(combined_vector, nan=0.0, posinf=1.0, neginf=-1.0)

        # Create IntentVector
        intent_vector = IntentVector(
            semantic_vector=pooled_output,
            sentiment_polarity=sentiment_polarity,
            sentiment_intensity=sentiment_intensity,
            intent_type=intent_type,
            intent_confidence=intent_confidence,
            goal_vector=goal_vector,
            combined_vector=combined_vector
        )

        return intent_vector

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 512
    ) -> List[IntentVector]:
        """
        Encode a batch of texts with automatic batching.

        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            max_length: Maximum sequence length

        Returns:
            List of IntentVector objects
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            intent_vector = self.forward(batch_texts, max_length)
            results.append(intent_vector)

            logger.debug(f"Processed batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")

        logger.info(f"Encoded {len(texts)} texts in {(len(texts) + batch_size - 1) // batch_size} batches")
        return results

    def get_semantic_vector(self, text: Union[str, List[str]], max_length: int = 512) -> torch.Tensor:
        """
        Get only the semantic vector (without sentiment/intent extraction).

        This is a convenience method for cases where only the semantic
        representation is needed.

        Args:
            text: Input text or list of texts
            max_length: Maximum sequence length

        Returns:
            Semantic vector of shape (batch_size, hidden_dim)
        """
        _, pooled_output = self.encode_text(text, max_length)
        return pooled_output

    def save(self, path: str):
        """Save model state to file."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'output_dim': self.output_dim
        }, path)
        logger.info(f"Saved SemanticEncoder to {path}")

    def load(self, path: str):
        """Load model state from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded SemanticEncoder from {path}")


def create_semantic_encoder(
    output_dim: int = 256,
    device: str = 'cuda',
    config: Optional[EncoderConfig] = None
) -> SemanticEncoder:
    """
    Factory function to create a semantic encoder.

    Args:
        output_dim: Output dimension for intent vectors
        device: Device to run the model on
        config: EncoderConfig instance (uses default if None)

    Returns:
        Initialized SemanticEncoder instance
    """
    if config is None:
        config = EncoderConfig()

    return SemanticEncoder(config=config, output_dim=output_dim, device=device)