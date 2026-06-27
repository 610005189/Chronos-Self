"""
Tests for Chronos-Self Configuration System
============================================

This module tests the configuration management system, including:
- Configuration initialization
- Parameter validation
- Save/load functionality
- Nested updates
"""

import pytest
from pathlib import Path
import json
import tempfile

from chronos_core.utils.config import (
    ChronosConfig,
    DimensionalityConfig,
    MemoryTemporalConfig,
    CouplingStabilityConfig,
    ChaosInjectionConfig,
    TrainingConfig,
    NeuralODEConfig,
    EncoderConfig,
    MetaCognitiveConfig,
    ValidationConfig,
    LoggingConfig,
    PathsConfig,
    get_config,
    set_config,
    init_config,
)


class TestDimensionalityConfig:
    """Test dimensionality configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = DimensionalityConfig()
        assert config.fast_variable_dim == 2048
        assert config.slow_variable_dim == 512
        assert config.core_subspace_dim == 64
        assert config.semantic_dim == 512
        assert config.physical_dim == 512
        assert config.fusion_dim == 1024
        assert config.working_memory_dim == 256
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = DimensionalityConfig(
            fast_variable_dim=4096,
            slow_variable_dim=1024,
            core_subspace_dim=128
        )
        assert config.fast_variable_dim == 4096
        assert config.slow_variable_dim == 1024
        assert config.core_subspace_dim == 128


class TestMemoryTemporalConfig:
    """Test memory and temporal configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = MemoryTemporalConfig()
        assert config.working_memory_chunks == 7
        assert config.reflection_window == 1000
        assert config.sleep_replay_interval_hours == 24.0
        assert config.sleep_duration_minutes == 5.0
        assert config.slow_update_frequency == 100
    
    def test_millers_law(self):
        """Test that working memory chunks satisfy Miller's law."""
        config = MemoryTemporalConfig()
        # Miller's law: 7±2, so should be between 5 and 9
        assert 5 <= config.working_memory_chunks <= 9


class TestCouplingStabilityConfig:
    """Test coupling and stability configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = CouplingStabilityConfig()
        assert config.coupling_adaptation_coeff == 0.5
        assert config.elastic_restoration_coeff == 0.05
        assert config.l2_perturbation_noise == 0.05
        assert config.anti_quietus_weight == 0.1
        assert config.inertia_weight == 0.05
    
    def test_positive_values(self):
        """Test that all stability parameters are positive."""
        config = CouplingStabilityConfig()
        assert config.coupling_adaptation_coeff > 0
        assert config.elastic_restoration_coeff > 0
        assert config.l2_perturbation_noise > 0
        assert config.anti_quietus_weight > 0
        assert config.inertia_weight > 0


class TestChaosInjectionConfig:
    """Test chaos injection configuration."""
    
    def test_lorenz_parameters(self):
        """Test Lorenz attractor parameters."""
        config = ChaosInjectionConfig()
        assert config.lorenz_sigma == 10.0
        assert config.lorenz_rho == 28.0
        assert config.lorenz_beta == 8.0 / 3.0
    
    def test_rossler_parameters(self):
        """Test Rössler attractor parameters."""
        config = ChaosInjectionConfig()
        assert config.rossler_a == 0.2
        assert config.rossler_b == 0.2
        assert config.rossler_c == 5.7
    
    def test_chua_parameters(self):
        """Test Chua's circuit parameters."""
        config = ChaosInjectionConfig()
        assert config.chua_alpha == 15.35
        assert config.chua_beta == 28.0


class TestChronosConfig:
    """Test master configuration class."""
    
    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = ChronosConfig()
        assert config.dim is not None
        assert config.memory_temporal is not None
        assert config.coupling_stability is not None
        assert config.chaos_injection is not None
        assert config.training is not None
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = ChronosConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'dim' in config_dict
        assert 'memory_temporal' in config_dict
        assert config_dict['dim']['fast_variable_dim'] == 2048
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            'dim': {
                'fast_variable_dim': 4096,
                'slow_variable_dim': 1024,
            },
            'memory_temporal': {
                'working_memory_chunks': 7,
            },
            'random_seed': 123,
        }
        
        config = ChronosConfig.from_dict(config_dict)
        assert config.dim.fast_variable_dim == 4096
        assert config.dim.slow_variable_dim == 1024
        assert config.random_seed == 123
    
    def test_save_and_load(self):
        """Test save and load functionality."""
        config = ChronosConfig()
        config.dim.fast_variable_dim = 4096
        config.random_seed = 999
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            config.save(temp_path)
            
            loaded_config = ChronosConfig.load(temp_path)
            assert loaded_config.dim.fast_variable_dim == 4096
            assert loaded_config.random_seed == 999
        finally:
            temp_path.unlink()
    
    def test_update_flat(self):
        """Test flat parameter update."""
        config = ChronosConfig()
        config.update(random_seed=123, device='cpu')
        assert config.random_seed == 123
        assert config.device == 'cpu'
    
    def test_update_nested(self):
        """Test nested parameter update."""
        config = ChronosConfig()
        config.update(dim__fast_variable_dim=4096)
        assert config.dim.fast_variable_dim == 4096


class TestGlobalConfig:
    """Test global configuration management."""
    
    def test_get_config(self):
        """Test getting global configuration."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2
    
    def test_set_config(self):
        """Test setting global configuration."""
        new_config = ChronosConfig()
        new_config.random_seed = 123
        
        set_config(new_config)
        retrieved_config = get_config()
        
        assert retrieved_config is new_config
        assert retrieved_config.random_seed == 123
    
    def test_init_config(self):
        """Test initialization with custom parameters."""
        config = init_config(random_seed=456, dim__fast_variable_dim=4096)
        assert config.random_seed == 456
        assert config.dim.fast_variable_dim == 4096
    
    def test_init_config_from_file(self):
        """Test initialization from file."""
        config = ChronosConfig()
        config.random_seed = 789
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            config.save(temp_path)
            
            loaded_config = init_config(config_path=temp_path)
            assert loaded_config.random_seed == 789
        finally:
            temp_path.unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])