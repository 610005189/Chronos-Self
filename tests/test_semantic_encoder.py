"""
Tests for Semantic Intent Encoder
===================================

This test file verifies the implementation of the semantic intent encoder
and its components.
"""

import pytest
import torch
from chronos_core.representation.semantic_encoder import (
    SemanticEncoder,
    SentimentExtractor,
    IntentExtractor,
    IntentVector,
    LightweightTransformerEncoder,
    PositionalEncoding,
    create_semantic_encoder
)
from chronos_core.utils.config import EncoderConfig


class TestPositionalEncoding:
    """Test positional encoding implementation."""

    def test_positional_encoding_shape(self):
        """Test that positional encoding produces correct output shape."""
        d_model = 512
        max_len = 100
        batch_size = 4
        seq_len = 20

        pos_encoder = PositionalEncoding(d_model, max_len)
        x = torch.randn(batch_size, seq_len, d_model)
        output = pos_encoder(x)

        assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"

    def test_positional_encoding_deterministic(self):
        """Test that positional encoding is deterministic."""
        d_model = 512
        pos_encoder = PositionalEncoding(d_model)
        x = torch.randn(2, 10, d_model)

        output1 = pos_encoder(x)
        output2 = pos_encoder(x)

        assert torch.allclose(output1, output2), "Positional encoding should be deterministic"


class TestLightweightTransformerEncoder:
    """Test lightweight transformer encoder implementation."""

    def test_transformer_encoder_shape(self):
        """Test that transformer encoder produces correct output shapes."""
        hidden_dim = 512
        num_layers = 4
        num_heads = 8
        batch_size = 4
        seq_len = 20

        encoder = LightweightTransformerEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads
        )

        x = torch.randn(batch_size, seq_len, hidden_dim)
        encoded_output, pooled_output = encoder(x)

        assert encoded_output.shape == (batch_size, seq_len, hidden_dim)
        assert pooled_output.shape == (batch_size, hidden_dim)

    def test_transformer_encoder_with_mask(self):
        """Test that transformer encoder handles attention masks correctly."""
        hidden_dim = 512
        batch_size = 4
        seq_len = 20

        encoder = LightweightTransformerEncoder(hidden_dim=hidden_dim)

        x = torch.randn(batch_size, seq_len, hidden_dim)
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, 15:] = 0  # Mask out last 5 tokens

        encoded_output, pooled_output = encoder(x, attention_mask)

        assert encoded_output.shape == (batch_size, seq_len, hidden_dim)
        assert pooled_output.shape == (batch_size, hidden_dim)


class TestSentimentExtractor:
    """Test sentiment extractor implementation."""

    def test_sentiment_extractor_shape(self):
        """Test that sentiment extractor produces correct output shapes."""
        input_dim = 512
        batch_size = 4

        extractor = SentimentExtractor(input_dim=input_dim)

        semantic_vector = torch.randn(batch_size, input_dim)
        polarity, intensity, embedding = extractor(semantic_vector)

        assert isinstance(polarity, float)
        assert isinstance(intensity, float)
        assert embedding.shape == (batch_size, input_dim // 4)

    def test_sentiment_polarity_range(self):
        """Test that sentiment polarity is in [-1, 1]."""
        input_dim = 512
        extractor = SentimentExtractor(input_dim=input_dim)

        # Test with multiple random inputs
        for _ in range(10):
            semantic_vector = torch.randn(2, input_dim)
            polarity, _, _ = extractor(semantic_vector)
            assert -1.0 <= polarity <= 1.0, f"Polarity {polarity} out of range [-1, 1]"

    def test_sentiment_intensity_range(self):
        """Test that sentiment intensity is in [0, 1]."""
        input_dim = 512
        extractor = SentimentExtractor(input_dim=input_dim)

        # Test with multiple random inputs
        for _ in range(10):
            semantic_vector = torch.randn(2, input_dim)
            _, intensity, _ = extractor(semantic_vector)
            assert 0.0 <= intensity <= 1.0, f"Intensity {intensity} out of range [0, 1]"


class TestIntentExtractor:
    """Test intent extractor implementation."""

    def test_intent_extractor_shape(self):
        """Test that intent extractor produces correct output shapes."""
        input_dim = 512
        goal_dim = 128
        batch_size = 4

        extractor = IntentExtractor(
            input_dim=input_dim,
            goal_dim=goal_dim
        )

        semantic_vector = torch.randn(batch_size, input_dim)
        intent_type, confidence, goal_vector, intent_embedding = extractor(semantic_vector)

        assert isinstance(intent_type, str)
        assert isinstance(confidence, float)
        assert goal_vector.shape == (batch_size, goal_dim)
        assert intent_embedding.shape == (batch_size, input_dim // 4)

    def test_intent_type_valid(self):
        """Test that predicted intent type is valid."""
        input_dim = 512
        extractor = IntentExtractor(input_dim=input_dim)

        semantic_vector = torch.randn(2, input_dim)
        intent_type, _, _, _ = extractor(semantic_vector)

        assert intent_type in IntentExtractor.INTENT_TYPES, \
            f"Intent type '{intent_type}' not in valid types: {IntentExtractor.INTENT_TYPES}"

    def test_intent_confidence_range(self):
        """Test that intent confidence is in [0, 1]."""
        input_dim = 512
        extractor = IntentExtractor(input_dim=input_dim)

        # Test with multiple random inputs
        for _ in range(10):
            semantic_vector = torch.randn(2, input_dim)
            _, confidence, _, _ = extractor(semantic_vector)
            assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} out of range [0, 1]"


class TestSemanticEncoder:
    """Test semantic encoder integration."""

    @pytest.fixture
    def encoder_config(self):
        """Create test encoder configuration."""
        return EncoderConfig(
            semantic_model_name="sentence-transformers/all-MiniLM-L6-v2",
            semantic_hidden_dim=512,
            semantic_num_layers=4,
            semantic_num_heads=8,
            cross_attention_dropout=0.1
        )

    @pytest.fixture
    def encoder(self, encoder_config):
        """Create test semantic encoder."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return SemanticEncoder(
            config=encoder_config,
            output_dim=256,
            device=device
        )

    def test_encoder_initialization(self, encoder):
        """Test that encoder initializes correctly."""
        assert encoder is not None
        assert encoder.output_dim == 256

    def test_encoder_forward_single_text(self, encoder):
        """Test encoding a single text."""
        text = "Hello, this is a test sentence."

        intent_vector = encoder(text)

        assert isinstance(intent_vector, IntentVector)
        assert intent_vector.semantic_vector.shape[0] == 1
        assert intent_vector.combined_vector.shape == (1, 256)
        assert -1.0 <= intent_vector.sentiment_polarity <= 1.0
        assert 0.0 <= intent_vector.sentiment_intensity <= 1.0
        assert 0.0 <= intent_vector.intent_confidence <= 1.0
        assert intent_vector.intent_type in IntentExtractor.INTENT_TYPES

    def test_encoder_forward_batch_texts(self, encoder):
        """Test encoding a batch of texts."""
        texts = [
            "Hello, world!",
            "This is a test.",
            "How are you today?",
            "I love this product."
        ]

        intent_vector = encoder(texts)

        assert isinstance(intent_vector, IntentVector)
        assert intent_vector.semantic_vector.shape[0] == len(texts)
        assert intent_vector.combined_vector.shape == (len(texts), 256)

    def test_encoder_numerical_stability(self, encoder):
        """Test that encoder handles extreme inputs without NaN/Inf."""
        # Test with very long text
        long_text = "This is a very long text. " * 100

        intent_vector = encoder(long_text)

        assert not torch.isnan(intent_vector.semantic_vector).any()
        assert not torch.isinf(intent_vector.semantic_vector).any()
        assert not torch.isnan(intent_vector.combined_vector).any()
        assert not torch.isinf(intent_vector.combined_vector).any()

    def test_encoder_get_semantic_vector(self, encoder):
        """Test getting only semantic vector."""
        text = "This is a test sentence."

        semantic_vector = encoder.get_semantic_vector(text)

        assert semantic_vector.shape[0] == 1
        assert not torch.isnan(semantic_vector).any()
        assert not torch.isinf(semantic_vector).any()

    def test_encoder_output_dimension(self, encoder):
        """Test that output dimension is correct."""
        text = "Test sentence."
        output_dim = encoder.output_dim

        intent_vector = encoder(text)

        assert intent_vector.combined_vector.shape[1] == output_dim
        assert intent_vector.goal_vector.shape[1] == output_dim // 2

    def test_create_semantic_encoder_factory(self):
        """Test factory function for creating semantic encoder."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        encoder = create_semantic_encoder(
            output_dim=512,
            device=device
        )

        assert encoder.output_dim == 512

        text = "Test sentence."
        intent_vector = encoder(text)

        assert intent_vector.combined_vector.shape[1] == 512


class TestIntentVector:
    """Test IntentVector dataclass."""

    def test_intent_vector_to_dict(self):
        """Test IntentVector serialization to dictionary."""
        intent_vector = IntentVector(
            semantic_vector=torch.randn(2, 512),
            sentiment_polarity=0.5,
            sentiment_intensity=0.8,
            intent_type='inform',
            intent_confidence=0.9,
            goal_vector=torch.randn(2, 128),
            combined_vector=torch.randn(2, 256)
        )

        vector_dict = intent_vector.to_dict()

        assert isinstance(vector_dict, dict)
        assert 'semantic_vector' in vector_dict
        assert 'sentiment_polarity' in vector_dict
        assert 'sentiment_intensity' in vector_dict
        assert 'intent_type' in vector_dict
        assert 'intent_confidence' in vector_dict
        assert 'goal_vector' in vector_dict
        assert 'combined_vector' in vector_dict


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])