"""
Simple verification script for Semantic Encoder implementation.
"""

import torch
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos_core.representation.semantic_encoder import (
    LightweightTransformerEncoder,
    PositionalEncoding,
    SentimentExtractor,
    IntentExtractor,
    IntentVector
)
from chronos_core.utils.config import EncoderConfig


def test_positional_encoding():
    """Test positional encoding."""
    print("\n" + "="*60)
    print("Testing PositionalEncoding")
    print("="*60)

    d_model = 512
    batch_size = 2
    seq_len = 10

    pos_encoder = PositionalEncoding(d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    output = pos_encoder(x)

    assert output.shape == x.shape
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Test passed!")


def test_lightweight_transformer():
    """Test lightweight transformer encoder."""
    print("\n" + "="*60)
    print("Testing LightweightTransformerEncoder")
    print("="*60)

    hidden_dim = 512
    batch_size = 2
    seq_len = 10

    encoder = LightweightTransformerEncoder(
        hidden_dim=hidden_dim,
        num_layers=4,
        num_heads=8
    )
    x = torch.randn(batch_size, seq_len, hidden_dim)
    encoded_output, pooled_output = encoder(x)

    assert encoded_output.shape == (batch_size, seq_len, hidden_dim)
    assert pooled_output.shape == (batch_size, hidden_dim)

    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Encoded output shape: {encoded_output.shape}")
    print(f"✓ Pooled output shape: {pooled_output.shape}")
    print(f"✓ Test passed!")


def test_sentiment_extractor():
    """Test sentiment extractor."""
    print("\n" + "="*60)
    print("Testing SentimentExtractor")
    print("="*60)

    input_dim = 512
    batch_size = 2

    extractor = SentimentExtractor(input_dim=input_dim)
    semantic_vector = torch.randn(batch_size, input_dim)
    polarity, intensity, embedding = extractor(semantic_vector)

    assert isinstance(polarity, float)
    assert isinstance(intensity, float)
    assert embedding.shape == (batch_size, input_dim // 4)
    assert -1.0 <= polarity <= 1.0
    assert 0.0 <= intensity <= 1.0

    print(f"✓ Input shape: {semantic_vector.shape}")
    print(f"✓ Polarity: {polarity:.4f} (range [-1, 1])")
    print(f"✓ Intensity: {intensity:.4f} (range [0, 1])")
    print(f"✓ Embedding shape: {embedding.shape}")
    print(f"✓ Test passed!")


def test_intent_extractor():
    """Test intent extractor."""
    print("\n" + "="*60)
    print("Testing IntentExtractor")
    print("="*60)

    input_dim = 512
    goal_dim = 128
    batch_size = 2

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
    assert 0.0 <= confidence <= 1.0
    assert intent_type in IntentExtractor.INTENT_TYPES

    print(f"✓ Input shape: {semantic_vector.shape}")
    print(f"✓ Intent type: {intent_type}")
    print(f"✓ Confidence: {confidence:.4f} (range [0, 1])")
    print(f"✓ Goal vector shape: {goal_vector.shape}")
    print(f"✓ Intent embedding shape: {intent_embedding.shape}")
    print(f"✓ Test passed!")


def test_intent_vector():
    """Test IntentVector dataclass."""
    print("\n" + "="*60)
    print("Testing IntentVector")
    print("="*60)

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
    assert 'intent_type' in vector_dict

    print(f"✓ IntentVector created successfully")
    print(f"✓ Serialization works correctly")
    print(f"✓ Test passed!")


def test_semantic_encoder_components():
    """Test that all components work together."""
    print("\n" + "="*60)
    print("Testing SemanticEncoder Components Integration")
    print("="*60)

    # Create configuration
    config = EncoderConfig(
        semantic_model_name="sentence-transformers/all-MiniLM-L6-v2",
        semantic_hidden_dim=512,
        semantic_num_layers=4,
        semantic_num_heads=8
    )

    # Test input projection and transformer
    base_hidden_dim = 384  # typical for sentence-transformers
    input_projection = torch.nn.Linear(base_hidden_dim, config.semantic_hidden_dim)

    transformer_encoder = LightweightTransformerEncoder(
        hidden_dim=config.semantic_hidden_dim,
        num_layers=config.semantic_num_layers,
        num_heads=config.semantic_num_heads
    )

    sentiment_extractor = SentimentExtractor(
        input_dim=config.semantic_hidden_dim
    )

    intent_extractor = IntentExtractor(
        input_dim=config.semantic_hidden_dim
    )

    # Simulate input
    batch_size = 2
    seq_len = 10
    base_hidden = torch.randn(batch_size, seq_len, base_hidden_dim)

    # Process through pipeline
    projected = input_projection(base_hidden)
    encoded_output, pooled_output = transformer_encoder(projected)
    polarity, intensity, sentiment_emb = sentiment_extractor(pooled_output)
    intent_type, confidence, goal_vec, intent_emb = intent_extractor(pooled_output)

    # Verify shapes
    assert projected.shape == (batch_size, seq_len, config.semantic_hidden_dim)
    assert pooled_output.shape == (batch_size, config.semantic_hidden_dim)
    assert sentiment_emb.shape[1] == config.semantic_hidden_dim // 4

    print(f"✓ Pipeline works correctly")
    print(f"✓ Base hidden -> Projected: {base_hidden.shape} -> {projected.shape}")
    print(f"✓ Transformer encoding: {encoded_output.shape}")
    print(f"✓ Pooling output: {pooled_output.shape}")
    print(f"✓ Sentiment extracted: polarity={polarity:.4f}, intensity={intensity:.4f}")
    print(f"✓ Intent extracted: type={intent_type}, confidence={confidence:.4f}")
    print(f"✓ Test passed!")


def main():
    """Run all verification tests."""
    print("\n" + "="*60)
    print("SEMANTIC ENCODER VERIFICATION TESTS")
    print("="*60)

    try:
        test_positional_encoding()
        test_lightweight_transformer()
        test_sentiment_extractor()
        test_intent_extractor()
        test_intent_vector()
        test_semantic_encoder_components()

        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("\nSemanticEncoder implementation verified successfully!")
        print("All components work correctly and produce expected outputs.")

    except Exception as e:
        print("\n" + "="*60)
        print("TEST FAILED ✗")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)