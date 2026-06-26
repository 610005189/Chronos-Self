"""
Test suite for Cross-Attention Fusion Mechanism
================================================

This test suite validates the correct implementation of the fusion mechanism
components:
- SemanticToPhysicalCrossAttention
- PhysicalToSemanticCrossAttention
- FusionModule

Tests cover:
- Dimension correctness
- Numerical stability
- Batch processing
- Integration with existing encoders
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any

from chronos_core.representation.fusion import (
    SemanticToPhysicalCrossAttention,
    PhysicalToSemanticCrossAttention,
    FusionModule,
    FusionOutput,
    ScaledDotProductAttention,
    create_fusion_module
)

from chronos_core.utils.config import EncoderConfig


class TestScaledDotProductAttention:
    """Test ScaledDotProductAttention functionality."""

    def test_basic_attention(self):
        """Test basic attention computation."""
        attention = ScaledDotProductAttention(dropout=0.0, scale_by_dim=True)

        batch_size = 2
        num_queries = 4
        num_keys = 6
        query_dim = 32

        query = torch.randn(batch_size, num_queries, query_dim)
        key = torch.randn(batch_size, num_keys, query_dim)
        value = torch.randn(batch_size, num_keys, query_dim)

        output, weights = attention(query, key, value, need_weights=True)

        # Check output dimensions
        assert output.shape == (batch_size, num_queries, query_dim)
        assert weights.shape == (batch_size, num_queries, num_keys)

        # Check weights sum to 1
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, num_queries), atol=1e-5)

    def test_attention_with_mask(self):
        """Test attention with masking."""
        attention = ScaledDotProductAttention(dropout=0.0, scale_by_dim=True)

        batch_size = 2
        num_queries = 4
        num_keys = 6
        query_dim = 32

        query = torch.randn(batch_size, num_queries, query_dim)
        key = torch.randn(batch_size, num_keys, query_dim)
        value = torch.randn(batch_size, num_keys, query_dim)

        # Create mask: mask last 2 keys for each query
        mask = torch.zeros(batch_size, num_queries, num_keys, dtype=torch.bool)
        mask[:, :, -2:] = True

        output, weights = attention(query, key, value, attention_mask=mask, need_weights=True)

        # Check that masked positions have zero attention weight
        assert torch.allclose(weights[:, :, -2:], torch.zeros(batch_size, num_queries, 2), atol=1e-6)

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        attention = ScaledDotProductAttention(dropout=0.0, scale_by_dim=True)

        # Create inputs with extreme values
        query = torch.randn(2, 4, 32) * 1000
        key = torch.randn(2, 6, 32) * 1000
        value = torch.randn(2, 6, 32)

        output, weights = attention(query, key, value, need_weights=True)

        # Check no NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert not torch.isnan(weights).any()
        assert not torch.isinf(weights).any()


class TestSemanticToPhysicalCrossAttention:
    """Test Semantic to Physical Cross-Attention."""

    def test_dimension_correctness(self):
        """Test output dimension correctness."""
        cross_attention = SemanticToPhysicalCrossAttention(
            sem_dim=256,
            log_dim=512,
            num_heads=8,
            dropout=0.0
        )

        batch_size = 2
        seq_len = 10
        sem_dim = 256
        log_dim = 512

        X_sem = torch.randn(batch_size, seq_len, sem_dim)
        X_log = torch.randn(batch_size, seq_len, log_dim)

        X_sem_enriched, attention_weights = cross_attention(X_sem, X_log, need_weights=True)

        # Check output dimensions
        assert X_sem_enriched.shape == (batch_size, seq_len, sem_dim), \
            f"Expected shape {(batch_size, seq_len, sem_dim)}, got {X_sem_enriched.shape}"

        # Check attention weights dimensions
        if attention_weights is not None:
            assert attention_weights.shape == (batch_size, cross_attention.num_heads, seq_len, seq_len), \
                f"Expected shape {(batch_size, cross_attention.num_heads, seq_len, seq_len)}, got {attention_weights.shape}"

    def test_numerical_stability(self):
        """Test numerical stability with various inputs."""
        cross_attention = SemanticToPhysicalCrossAttention(
            sem_dim=256,
            log_dim=512,
            num_heads=8,
            dropout=0.0
        )

        batch_size = 2
        seq_len = 10

        # Test with normal inputs
        X_sem = torch.randn(batch_size, seq_len, 256)
        X_log = torch.randn(batch_size, seq_len, 512)

        X_sem_enriched, _ = cross_attention(X_sem, X_log)

        assert not torch.isnan(X_sem_enriched).any()
        assert not torch.isinf(X_sem_enriched).any()

        # Test with extreme inputs
        X_sem_extreme = torch.randn(batch_size, seq_len, 256) * 100
        X_log_extreme = torch.randn(batch_size, seq_len, 512) * 100

        X_sem_enriched_extreme, _ = cross_attention(X_sem_extreme, X_log_extreme)

        assert not torch.isnan(X_sem_enriched_extreme).any()
        assert not torch.isinf(X_sem_enriched_extreme).any()

    def test_different_head_configurations(self):
        """Test with different number of heads."""
        for num_heads in [4, 8]:
            cross_attention = SemanticToPhysicalCrossAttention(
                sem_dim=256,
                log_dim=512,
                num_heads=num_heads,
                dropout=0.0
            )

            X_sem = torch.randn(2, 10, 256)
            X_log = torch.randn(2, 10, 512)

            X_sem_enriched, attention_weights = cross_attention(X_sem, X_log, need_weights=True)

            assert X_sem_enriched.shape == (2, 10, 256)
            if attention_weights is not None:
                assert attention_weights.shape[1] == num_heads


class TestPhysicalToSemanticCrossAttention:
    """Test Physical to Semantic Cross-Attention."""

    def test_dimension_correctness(self):
        """Test output dimension correctness."""
        cross_attention = PhysicalToSemanticCrossAttention(
            log_dim=512,
            sem_dim=256,
            num_heads=8,
            dropout=0.0
        )

        batch_size = 2
        seq_len = 10
        log_dim = 512
        sem_dim = 256

        X_log = torch.randn(batch_size, seq_len, log_dim)
        X_sem = torch.randn(batch_size, seq_len, sem_dim)

        X_log_enriched, attention_weights = cross_attention(X_log, X_sem, need_weights=True)

        # Check output dimensions
        assert X_log_enriched.shape == (batch_size, seq_len, log_dim), \
            f"Expected shape {(batch_size, seq_len, log_dim)}, got {X_log_enriched.shape}"

        # Check attention weights dimensions
        if attention_weights is not None:
            assert attention_weights.shape == (batch_size, cross_attention.num_heads, seq_len, seq_len), \
                f"Expected shape {(batch_size, cross_attention.num_heads, seq_len, seq_len)}, got {attention_weights.shape}"

    def test_numerical_stability(self):
        """Test numerical stability."""
        cross_attention = PhysicalToSemanticCrossAttention(
            log_dim=512,
            sem_dim=256,
            num_heads=8,
            dropout=0.0
        )

        batch_size = 2
        seq_len = 10

        X_log = torch.randn(batch_size, seq_len, 512)
        X_sem = torch.randn(batch_size, seq_len, 256)

        X_log_enriched, _ = cross_attention(X_log, X_sem)

        assert not torch.isnan(X_log_enriched).any()
        assert not torch.isinf(X_log_enriched).any()

    def test_dimension_adjustment(self):
        """Test automatic dimension adjustment when log_dim not divisible by num_heads."""
        # log_dim=512 is divisible by 8, so no adjustment needed
        cross_attention = PhysicalToSemanticCrossAttention(
            log_dim=512,
            sem_dim=256,
            num_heads=8,
            dropout=0.0
        )

        assert cross_attention.effective_query_dim == 512
        assert cross_attention.head_dim == 64


class TestFusionModule:
    """Test complete Fusion Module."""

    def test_fusion_output_dimensions(self):
        """Test fusion output dimensions."""
        fusion_module = FusionModule(
            sem_dim=256,
            log_dim=512,
            fusion_dim=768,
            num_heads=8,
            dropout=0.0
        )

        batch_size = 2
        seq_len = 10

        X_sem = torch.randn(batch_size, seq_len, 256)
        X_log = torch.randn(batch_size, seq_len, 512)

        # Test returning only fused output
        X_fused = fusion_module(X_sem, X_log, return_enriched=False)
        assert X_fused.shape == (batch_size, seq_len, 768), \
            f"Expected shape {(batch_size, seq_len, 768)}, got {X_fused.shape}"

        # Test returning enriched representations
        fusion_output = fusion_module(X_sem, X_log, return_enriched=True)

        assert isinstance(fusion_output, FusionOutput)
        assert fusion_output.X_fused.shape == (batch_size, seq_len, 768)
        assert fusion_output.X_sem_enriched.shape == (batch_size, seq_len, 256)
        assert fusion_output.X_log_enriched.shape == (batch_size, seq_len, 512)

    def test_fusion_metadata(self):
        """Test fusion metadata."""
        fusion_module = FusionModule(
            sem_dim=256,
            log_dim=512,
            fusion_dim=768,
            num_heads=8,
            dropout=0.0
        )

        X_sem = torch.randn(2, 10, 256)
        X_log = torch.randn(2, 10, 512)

        fusion_output = fusion_module(X_sem, X_log, return_enriched=True)

        assert fusion_output.metadata is not None
        assert 'X_sem_norm' in fusion_output.metadata
        assert 'X_log_norm' in fusion_output.metadata
        assert 'X_fused_norm' in fusion_output.metadata
        assert 'dimensions' in fusion_output.metadata

        # Check dimension metadata
        dims = fusion_output.metadata['dimensions']
        assert dims['sem_dim'] == 256
        assert dims['log_dim'] == 512
        assert dims['fusion_dim'] == 768

    def test_attention_fusion_mode(self):
        """Test attention-based fusion mode."""
        fusion_module = FusionModule(
            sem_dim=256,
            log_dim=512,
            fusion_dim=768,
            num_heads=8,
            dropout=0.0,
            use_attention_fusion=True
        )

        X_sem = torch.randn(2, 10, 256)
        X_log = torch.randn(2, 10, 512)

        fusion_output = fusion_module(X_sem, X_log, return_enriched=True)

        assert fusion_output.X_fused.shape == (2, 10, 768)
        assert fusion_output.metadata['fusion_strategy'] == 'attention'

    def test_numerical_stability(self):
        """Test numerical stability."""
        fusion_module = FusionModule(
            sem_dim=256,
            log_dim=512,
            fusion_dim=768,
            num_heads=8,
            dropout=0.0
        )

        # Test with normal inputs
        X_sem = torch.randn(2, 10, 256)
        X_log = torch.randn(2, 10, 512)

        fusion_output = fusion_module(X_sem, X_log, return_enriched=True)

        assert not torch.isnan(fusion_output.X_fused).any()
        assert not torch.isinf(fusion_output.X_fused).any()
        assert not torch.isnan(fusion_output.X_sem_enriched).any()
        assert not torch.isinf(fusion_output.X_sem_enriched).any()
        assert not torch.isnan(fusion_output.X_log_enriched).any()
        assert not torch.isinf(fusion_output.X_log_enriched).any()

        # Test with extreme inputs
        X_sem_extreme = torch.randn(2, 10, 256) * 100
        X_log_extreme = torch.randn(2, 10, 512) * 100

        fusion_output_extreme = fusion_module(X_sem_extreme, X_log_extreme, return_enriched=True)

        assert not torch.isnan(fusion_output_extreme.X_fused).any()
        assert not torch.isinf(fusion_output_extreme.X_fused).any()

    def test_batch_fusion(self):
        """Test batch fusion method."""
        fusion_module = FusionModule(
            sem_dim=256,
            log_dim=512,
            fusion_dim=768,
            num_heads=8,
            dropout=0.0
        )

        total_samples = 100
        sem_vectors = torch.randn(total_samples, 256)
        log_vectors = torch.randn(total_samples, 512)

        fused_vectors = fusion_module.fuse_batch(sem_vectors, log_vectors, batch_size=32)

        assert fused_vectors.shape == (total_samples, 768), \
            f"Expected shape {(total_samples, 768)}, got {fused_vectors.shape}"

        assert not torch.isnan(fused_vectors).any()
        assert not torch.isinf(fused_vectors).any()

    def test_custom_fusion_dim(self):
        """Test custom fusion dimension."""
        # Test projection when fusion_dim != sem_dim + log_dim
        fusion_module = FusionModule(
            sem_dim=256,
            log_dim=512,
            fusion_dim=512,  # Custom dimension (different from 256+512=768)
            num_heads=8,
            dropout=0.0
        )

        X_sem = torch.randn(2, 10, 256)
        X_log = torch.randn(2, 10, 512)

        X_fused = fusion_module(X_sem, X_log, return_enriched=False)

        assert X_fused.shape == (2, 10, 512)
        assert fusion_module.output_projection is not None


class TestIntegration:
    """Integration tests with configuration and existing components."""

    def test_fusion_with_config(self):
        """Test fusion module with EncoderConfig."""
        config = EncoderConfig()
        fusion_module = create_fusion_module(config)

        X_sem = torch.randn(2, 10, 256)
        X_log = torch.randn(2, 10, 512)

        fusion_output = fusion_module(X_sem, X_log, return_enriched=True)

        # Check dimensions match config expectations
        assert fusion_output.X_fused.shape[-1] == 256 + 512  # 768

    def test_attention_weights_return(self):
        """Test returning attention weights."""
        fusion_module = FusionModule(
            sem_dim=256,
            log_dim=512,
            fusion_dim=768,
            num_heads=8,
            dropout=0.0
        )

        X_sem = torch.randn(2, 10, 256)
        X_log = torch.randn(2, 10, 512)

        fusion_output = fusion_module(
            X_sem, X_log,
            return_enriched=True,
            need_attention_weights=True
        )

        # Check attention weights are returned
        assert fusion_output.sem_to_phys_attention is not None
        assert fusion_output.phys_to_sem_attention is not None

        # Check attention weights dimensions
        assert fusion_output.sem_to_phys_attention.shape == (2, 8, 10, 10)
        assert fusion_output.phys_to_sem_attention.shape == (2, 8, 10, 10)

    def test_gpu_acceleration(self):
        """Test GPU acceleration if available."""
        if torch.cuda.is_available():
            device = 'cuda'

            fusion_module = FusionModule(
                sem_dim=256,
                log_dim=512,
                fusion_dim=768,
                num_heads=8,
                dropout=0.0
            ).to(device)

            X_sem = torch.randn(2, 10, 256).to(device)
            X_log = torch.randn(2, 10, 512).to(device)

            fusion_output = fusion_module(X_sem, X_log, return_enriched=True)

            # Check outputs are on GPU
            assert fusion_output.X_fused.device.type == 'cuda'
            assert fusion_output.X_sem_enriched.device.type == 'cuda'
            assert fusion_output.X_log_enriched.device.type == 'cuda'

            # Check numerical stability on GPU
            assert not torch.isnan(fusion_output.X_fused).any()
            assert not torch.isinf(fusion_output.X_fused).any()
        else:
            pytest.skip("CUDA not available")

    def test_sequential_processing(self):
        """Test sequential processing across time steps."""
        fusion_module = FusionModule(
            sem_dim=256,
            log_dim=512,
            fusion_dim=768,
            num_heads=8,
            dropout=0.0
        )

        # Simulate sequence of 20 time steps
        batch_size = 1
        seq_len = 20

        X_sem_sequence = torch.randn(batch_size, seq_len, 256)
        X_log_sequence = torch.randn(batch_size, seq_len, 512)

        fusion_output = fusion_module(
            X_sem_sequence, X_log_sequence,
            return_enriched=True,
            need_attention_weights=True
        )

        # Check outputs for entire sequence
        assert fusion_output.X_fused.shape == (batch_size, seq_len, 768)
        assert fusion_output.X_sem_enriched.shape == (batch_size, seq_len, 256)
        assert fusion_output.X_log_enriched.shape == (batch_size, seq_len, 512)

        # Check attention weights shape
        assert fusion_output.sem_to_phys_attention.shape == (batch_size, 8, seq_len, seq_len)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_sample(self):
        """Test with single sample (batch_size=1)."""
        fusion_module = FusionModule(
            sem_dim=256,
            log_dim=512,
            fusion_dim=768,
            num_heads=8,
            dropout=0.0
        )

        X_sem = torch.randn(1, 5, 256)
        X_log = torch.randn(1, 5, 512)

        fusion_output = fusion_module(X_sem, X_log, return_enriched=True)

        assert fusion_output.X_fused.shape == (1, 5, 768)

    def test_long_sequence(self):
        """Test with long sequence."""
        fusion_module = FusionModule(
            sem_dim=256,
            log_dim=512,
            fusion_dim=768,
            num_heads=8,
            dropout=0.0
        )

        batch_size = 2
        seq_len = 100  # Long sequence

        X_sem = torch.randn(batch_size, seq_len, 256)
        X_log = torch.randn(batch_size, seq_len, 512)

        fusion_output = fusion_module(X_sem, X_log, return_enriched=False)

        assert fusion_output.shape == (batch_size, seq_len, 768)
        assert not torch.isnan(fusion_output).any()

    def test_nan_input_handling(self):
        """Test handling of NaN inputs."""
        fusion_module = FusionModule(
            sem_dim=256,
            log_dim=512,
            fusion_dim=768,
            num_heads=8,
            dropout=0.0
        )

        # Create inputs with NaN
        X_sem = torch.randn(2, 10, 256)
        X_sem[0, 0, 0] = float('nan')

        X_log = torch.randn(2, 10, 512)

        # Module should handle NaN gracefully
        fusion_output = fusion_module(X_sem, X_log, return_enriched=True)

        # Check NaN is corrected
        assert not torch.isnan(fusion_output.X_fused).any()


def run_tests():
    """Run all tests manually."""
    print("=" * 80)
    print("Running Fusion Module Tests")
    print("=" * 80)

    # Test basic attention
    print("\n1. Testing ScaledDotProductAttention...")
    test_attention = TestScaledDotProductAttention()
    test_attention.test_basic_attention()
    test_attention.test_attention_with_mask()
    test_attention.test_numerical_stability()
    print("✓ ScaledDotProductAttention tests passed")

    # Test semantic to physical cross-attention
    print("\n2. Testing SemanticToPhysicalCrossAttention...")
    test_sem_to_phys = TestSemanticToPhysicalCrossAttention()
    test_sem_to_phys.test_dimension_correctness()
    test_sem_to_phys.test_numerical_stability()
    test_sem_to_phys.test_different_head_configurations()
    print("✓ SemanticToPhysicalCrossAttention tests passed")

    # Test physical to semantic cross-attention
    print("\n3. Testing PhysicalToSemanticCrossAttention...")
    test_phys_to_sem = TestPhysicalToSemanticCrossAttention()
    test_phys_to_sem.test_dimension_correctness()
    test_phys_to_sem.test_numerical_stability()
    test_phys_to_sem.test_dimension_adjustment()
    print("✓ PhysicalToSemanticCrossAttention tests passed")

    # Test fusion module
    print("\n4. Testing FusionModule...")
    test_fusion = TestFusionModule()
    test_fusion.test_fusion_output_dimensions()
    test_fusion.test_fusion_metadata()
    test_fusion.test_attention_fusion_mode()
    test_fusion.test_numerical_stability()
    test_fusion.test_batch_fusion()
    test_fusion.test_custom_fusion_dim()
    print("✓ FusionModule tests passed")

    # Test integration
    print("\n5. Testing Integration...")
    test_integration = TestIntegration()
    test_integration.test_fusion_with_config()
    test_integration.test_attention_weights_return()
    test_integration.test_sequential_processing()
    print("✓ Integration tests passed")

    # Test edge cases
    print("\n6. Testing Edge Cases...")
    test_edge = TestEdgeCases()
    test_edge.test_single_sample()
    test_edge.test_long_sequence()
    test_edge.test_nan_input_handling()
    print("✓ Edge case tests passed")

    print("\n" + "=" * 80)
    print("All tests passed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    run_tests()