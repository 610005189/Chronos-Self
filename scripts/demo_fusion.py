"""
Demonstration Script for Cross-Attention Fusion Mechanism
===========================================================

This script demonstrates how to use the fusion module to combine
semantic intent and physical state representations in Chronos-Self.

Usage:
    python scripts/demo_fusion.py
"""

import torch
from chronos_core.representation import (
    FusionModule,
    create_fusion_module
)
from chronos_core.utils.config import EncoderConfig


def demo_basic_fusion():
    """Demonstrate basic fusion functionality."""
    print("=" * 80)
    print("Basic Fusion Demonstration")
    print("=" * 80)

    # Create fusion module with default dimensions
    print("\n1. Creating FusionModule...")
    fusion_module = FusionModule(
        sem_dim=256,  # Semantic encoder output dimension
        log_dim=512,  # Logical encoder output dimension
        fusion_dim=768,  # Fusion dimension (256 + 512)
        num_heads=8,
        dropout=0.0
    )

    print(f"   - Semantic dimension: {fusion_module.sem_dim}")
    print(f"   - Physical dimension: {fusion_module.log_dim}")
    print(f"   - Fusion dimension: {fusion_module.fusion_dim}")
    print(f"   - Number of attention heads: {fusion_module.num_heads}")

    # Create sample inputs (simulating encoder outputs)
    print("\n2. Creating sample inputs...")
    batch_size = 2
    seq_len = 10

    # Simulate semantic encoder output
    X_sem = torch.randn(batch_size, seq_len, 256)
    print(f"   - Semantic input shape: {X_sem.shape}")

    # Simulate logical encoder output
    X_log = torch.randn(batch_size, seq_len, 512)
    print(f"   - Physical input shape: {X_log.shape}")

    # Perform fusion
    print("\n3. Performing fusion...")
    fusion_output = fusion_module(
        X_sem, X_log,
        return_enriched=True,
        need_attention_weights=True
    )

    print(f"   - Fused output shape: {fusion_output.X_fused.shape}")
    print(f"   - Enriched semantic shape: {fusion_output.X_sem_enriched.shape}")
    print(f"   - Enriched physical shape: {fusion_output.X_log_enriched.shape}")

    # Check metadata
    print("\n4. Fusion metadata:")
    metadata = fusion_output.metadata
    print(f"   - Semantic input norm: {metadata['X_sem_norm']:.4f}")
    print(f"   - Physical input norm: {metadata['X_log_norm']:.4f}")
    print(f"   - Fused output norm: {metadata['X_fused_norm']:.4f}")
    print(f"   - Fusion strategy: {metadata['fusion_strategy']}")

    # Check attention weights
    if fusion_output.sem_to_phys_attention is not None:
        print("\n5. Attention weights:")
        print(f"   - Semantic→Physical attention shape: {fusion_output.sem_to_phys_attention.shape}")
        print(f"   - Physical→Semantic attention shape: {fusion_output.phys_to_sem_attention.shape}")

    print("\n✓ Basic fusion demonstration completed successfully!")


def demo_attention_fusion():
    """Demonstrate attention-based fusion mode."""
    print("\n" + "=" * 80)
    print("Attention-Based Fusion Demonstration")
    print("=" * 80)

    # Create fusion module with attention fusion
    print("\n1. Creating FusionModule with attention fusion...")
    fusion_module = FusionModule(
        sem_dim=256,
        log_dim=512,
        fusion_dim=768,
        num_heads=8,
        dropout=0.0,
        use_attention_fusion=True  # Enable attention-based fusion
    )

    print(f"   - Using attention-based fusion: {fusion_module.use_attention_fusion}")

    # Create sample inputs
    X_sem = torch.randn(2, 10, 256)
    X_log = torch.randn(2, 10, 512)

    # Perform fusion
    print("\n2. Performing attention-based fusion...")
    fusion_output = fusion_module(X_sem, X_log, return_enriched=True)

    print(f"   - Fused output shape: {fusion_output.X_fused.shape}")
    print(f"   - Fusion strategy: {fusion_output.metadata['fusion_strategy']}")

    print("\n✓ Attention-based fusion demonstration completed successfully!")


def demo_batch_fusion():
    """Demonstrate batch fusion for large-scale processing."""
    print("\n" + "=" * 80)
    print("Batch Fusion Demonstration")
    print("=" * 80)

    # Create fusion module
    print("\n1. Creating FusionModule...")
    fusion_module = FusionModule(
        sem_dim=256,
        log_dim=512,
        fusion_dim=768,
        num_heads=8,
        dropout=0.0
    )

    # Create large number of samples
    total_samples = 1000
    print(f"\n2. Creating {total_samples} samples...")

    sem_vectors = torch.randn(total_samples, 256)
    log_vectors = torch.randn(total_samples, 512)

    print(f"   - Semantic vectors shape: {sem_vectors.shape}")
    print(f"   - Physical vectors shape: {log_vectors.shape}")

    # Perform batch fusion
    print("\n3. Performing batch fusion...")
    batch_size = 64
    fused_vectors = fusion_module.fuse_batch(sem_vectors, log_vectors, batch_size=batch_size)

    print(f"   - Fused vectors shape: {fused_vectors.shape}")
    print(f"   - Batch size used: {batch_size}")

    print("\n✓ Batch fusion demonstration completed successfully!")


def demo_custom_dimensions():
    """Demonstrate custom fusion dimensions."""
    print("\n" + "=" * 80)
    print("Custom Fusion Dimensions Demonstration")
    print("=" * 80)

    # Create fusion module with custom output dimension
    print("\n1. Creating FusionModule with custom dimension...")
    fusion_module = FusionModule(
        sem_dim=256,
        log_dim=512,
        fusion_dim=512,  # Custom: smaller than 256+512=768
        num_heads=8,
        dropout=0.0
    )

    print(f"   - Output dimension: {fusion_module.fusion_dim} (projected from {256+512})")
    print(f"   - Has projection layer: {fusion_module.output_projection is not None}")

    # Create sample inputs
    X_sem = torch.randn(2, 10, 256)
    X_log = torch.randn(2, 10, 512)

    # Perform fusion
    print("\n2. Performing fusion with custom dimension...")
    X_fused = fusion_module(X_sem, X_log, return_enriched=False)

    print(f"   - Fused output shape: {X_fused.shape}")

    print("\n✓ Custom dimensions demonstration completed successfully!")


def demo_with_config():
    """Demonstrate using fusion module with EncoderConfig."""
    print("\n" + "=" * 80)
    print("Fusion with EncoderConfig Demonstration")
    print("=" * 80)

    # Create config
    print("\n1. Creating EncoderConfig...")
    config = EncoderConfig()
    print(f"   - Cross-attention heads: {config.cross_attention_heads}")
    print(f"   - Cross-attention dropout: {config.cross_attention_dropout}")

    # Create fusion module with config
    print("\n2. Creating FusionModule with config...")
    fusion_module = create_fusion_module(
        config=config,
        sem_dim=256,
        log_dim=512,
        fusion_dim=768
    )

    print(f"   - Number of heads (from config): {fusion_module.num_heads}")

    # Create sample inputs
    X_sem = torch.randn(2, 10, 256)
    X_log = torch.randn(2, 10, 512)

    # Perform fusion
    print("\n3. Performing fusion...")
    fusion_output = fusion_module(X_sem, X_log, return_enriched=True)

    print(f"   - Fused output shape: {fusion_output.X_fused.shape}")

    print("\n✓ Config-based fusion demonstration completed successfully!")


def run_all_demos():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("Chronos-Self Cross-Attention Fusion Mechanism Demonstration")
    print("=" * 80)

    demo_basic_fusion()
    demo_attention_fusion()
    demo_batch_fusion()
    demo_custom_dimensions()
    demo_with_config()

    print("\n" + "=" * 80)
    print("All demonstrations completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    run_all_demos()