"""
SemanticEncoder Usage Example
==============================

This example demonstrates how to use the SemanticEncoder for processing
text inputs and extracting semantic intent vectors.
"""

import torch
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos_core.representation.semantic_encoder import SemanticEncoder, IntentVector
from chronos_core.utils.config import EncoderConfig


def example_basic_usage():
    """Basic usage example: encoding a single text."""
    print("\n" + "="*60)
    print("Basic Usage: Encoding Single Text")
    print("="*60)

    # Create encoder
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = EncoderConfig()
    encoder = SemanticEncoder(config=config, output_dim=256, device=device)

    # Encode text
    text = "I'm feeling very happy about this new project!"
    intent_vector = encoder(text)

    # Display results
    print(f"\nInput text: {text}")
    print(f"\nResults:")
    print(f"  - Sentiment polarity: {intent_vector.sentiment_polarity:.4f}")
    print(f"  - Sentiment intensity: {intent_vector.sentiment_intensity:.4f}")
    print(f"  - Intent type: {intent_vector.intent_type}")
    print(f"  - Intent confidence: {intent_vector.intent_confidence:.4f}")
    print(f"  - Combined vector shape: {intent_vector.combined_vector.shape}")
    print(f"  - Goal vector shape: {intent_vector.goal_vector.shape}")

    return intent_vector


def example_batch_processing():
    """Batch processing example: encoding multiple texts."""
    print("\n" + "="*60)
    print("Batch Processing: Encoding Multiple Texts")
    print("="*60)

    # Create encoder
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = EncoderConfig()
    encoder = SemanticEncoder(config=config, output_dim=256, device=device)

    # Encode multiple texts
    texts = [
        "I need help with this task.",
        "This is an excellent product!",
        "Can you explain how this works?",
        "I apologize for the delay.",
        "Good morning, everyone!"
    ]

    intent_vector = encoder(texts)

    # Display results
    print(f"\nInput texts:")
    for i, text in enumerate(texts):
        print(f"  {i+1}. {text}")

    print(f"\nResults:")
    print(f"  - Batch size: {intent_vector.semantic_vector.shape[0]}")
    print(f"  - Combined vector shape: {intent_vector.combined_vector.shape}")
    print(f"  - Sentiment polarity: {intent_vector.sentiment_polarity:.4f}")
    print(f"  - Intent type: {intent_vector.intent_type}")
    print(f"  - Intent confidence: {intent_vector.intent_confidence:.4f}")

    return intent_vector


def example_different_outputs():
    """Example showing different output dimensions."""
    print("\n" + "="*60)
    print("Different Output Dimensions")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = EncoderConfig()

    # Test different output dimensions
    output_dims = [256, 512]

    for output_dim in output_dims:
        print(f"\nTesting output_dim = {output_dim}")
        encoder = SemanticEncoder(config=config, output_dim=output_dim, device=device)

        text = "This is a test sentence."
        intent_vector = encoder(text)

        print(f"  ✓ Combined vector shape: {intent_vector.combined_vector.shape}")
        print(f"  ✓ Goal vector shape: {intent_vector.goal_vector.shape}")
        print(f"  ✓ Expected: combined ({output_dim}), goal ({output_dim // 2})")


def example_intent_analysis():
    """Example analyzing different intent types."""
    print("\n" + "="*60)
    print("Intent Type Analysis")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = EncoderConfig()
    encoder = SemanticEncoder(config=config, output_dim=256, device=device)

    # Test different types of intents
    intent_texts = {
        'inform': "The meeting is scheduled for tomorrow at 2pm.",
        'request': "Could you please send me the report?",
        'question': "What time does the event start?",
        'command': "Complete the assignment by Friday.",
        'apologize': "I'm sorry for the inconvenience.",
        'greet': "Hello, nice to meet you!",
        'promise': "I will finish the project next week.",
        'acknowledge': "Thank you for your email."
    }

    print(f"\nAnalyzing intent types:")
    for intent_type, text in intent_texts.items():
        vector = encoder(text)
        predicted_intent = vector.intent_type
        confidence = vector.intent_confidence
        print(f"  {intent_type:12} -> Predicted: {predicted_intent:12} (confidence: {confidence:.4f})")


def example_sentiment_analysis():
    """Example analyzing sentiment in different texts."""
    print("\n" + "="*60)
    print("Sentiment Analysis")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = EncoderConfig()
    encoder = SemanticEncoder(config=config, output_dim=256, device=device)

    # Test different sentiment texts
    sentiment_texts = {
        'positive': "I love this product! It's absolutely amazing.",
        'negative': "This service is terrible. I'm very disappointed.",
        'neutral': "The package arrived on time as scheduled."
    }

    print(f"\nAnalyzing sentiment:")
    for sentiment_type, text in sentiment_texts.items():
        vector = encoder(text)
        polarity = vector.sentiment_polarity
        intensity = vector.sentiment_intensity
        print(f"  {sentiment_type:8} -> Polarity: {polarity:6.3f}, Intensity: {intensity:.3f}")


def example_serialization():
    """Example showing IntentVector serialization."""
    print("\n" + "="*60)
    print("IntentVector Serialization")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = EncoderConfig()
    encoder = SemanticEncoder(config=config, output_dim=256, device=device)

    # Encode and serialize
    text = "This is an example for serialization."
    intent_vector = encoder(text)

    # Convert to dictionary
    vector_dict = intent_vector.to_dict()

    print(f"\nSerialized IntentVector:")
    print(f"  Keys: {list(vector_dict.keys())}")
    print(f"  Sentiment polarity: {vector_dict['sentiment_polarity']}")
    print(f"  Sentiment intensity: {vector_dict['sentiment_intensity']}")
    print(f"  Intent type: {vector_dict['intent_type']}")
    print(f"  Intent confidence: {vector_dict['intent_confidence']}")
    print(f"  Semantic vector shape: {vector_dict['semantic_vector'].shape}")
    print(f"  Combined vector shape: {vector_dict['combined_vector'].shape}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("SEMANTIC ENCODER USAGE EXAMPLES")
    print("="*60)

    try:
        # Run examples
        example_basic_usage()
        example_batch_processing()
        example_different_outputs()
        example_intent_analysis()
        example_sentiment_analysis()
        example_serialization()

        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED ✓")
        print("="*60)
        print("\nSemanticEncoder is ready for use in Chronos-Self system!")

    except Exception as e:
        print("\n" + "="*60)
        print("EXAMPLE FAILED ✗")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()