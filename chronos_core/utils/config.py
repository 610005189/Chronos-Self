"""
Chronos-Self Configuration Management System
==============================================

This module defines all hyperparameters and configuration settings for the
Chronos-Self self-referential continuous dynamics system.

The configuration is organized into logical sections:
- Dimensionality parameters
- Memory and temporal parameters
- Coupling and stability parameters
- Chaos injection parameters
- Training and optimization parameters
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import json


@dataclass
class DimensionalityConfig:
    """Configuration for state space dimensions."""
    
    # Fast variable dimension (D_f) - represents rapid cognitive dynamics
    fast_variable_dim: int = 2048
    
    # Slow variable dimension (D_s) - represents stable personality/identity
    slow_variable_dim: int = 512
    
    # Core subspace dimension (k) - where chaos injection occurs
    core_subspace_dim: int = 64
    
    # Semantic encoder output dimension
    semantic_dim: int = 512
    
    # Physical encoder output dimension
    physical_dim: int = 512
    
    # Fusion output dimension
    fusion_dim: int = 1024
    
    # Working memory chunk dimension
    working_memory_dim: int = 256


@dataclass
class MemoryTemporalConfig:
    """Configuration for memory and temporal parameters."""
    
    # Number of working memory chunks (Miller's law: 7±2)
    working_memory_chunks: int = 7
    
    # Real-time reflection window (T) - number of steps for gradient computation
    reflection_window: int = 1000
    
    # Sleep replay interval (in hours)
    sleep_replay_interval_hours: float = 24.0
    
    # Sleep duration (in minutes)
    sleep_duration_minutes: float = 5.0
    
    # Slow variable update frequency (every N fast variable steps)
    slow_update_frequency: int = 100
    
    # Keyframe storage interval (in minutes)
    keyframe_interval_minutes: float = 30.0
    
    # Long-term validation window (in hours)
    validation_window_hours: float = 72.0


@dataclass
class CouplingStabilityConfig:
    """Configuration for coupling and stability parameters."""
    
    # Coupling adaptation coefficient (β) - controls coupling strength
    coupling_adaptation_coeff: float = 0.5
    
    # Elastic restoration coefficient (γ) - baseline return strength
    elastic_restoration_coeff: float = 0.05
    
    # L2 perturbation noise (σ_C) - for regularization
    l2_perturbation_noise: float = 0.05
    
    # Anti-quietus weight (λ) - prevents state collapse
    anti_quietus_weight: float = 0.1
    
    # Inertia weight (μ) - maintains state continuity
    inertia_weight: float = 0.05
    
    # Coupling strength upper bound
    coupling_upper_bound: float = 10.0
    
    # Stability threshold for monitoring
    stability_threshold: float = 1e6
    
    # Maximum Lyapunov exponent threshold for edge-of-chaos
    lyapunov_threshold: float = 0.1


@dataclass
class ChaosInjectionConfig:
    """Configuration for chaos injection from attractors."""
    
    # Lorenz attractor parameters
    lorenz_sigma: float = 10.0
    lorenz_rho: float = 28.0
    lorenz_beta: float = 8.0 / 3.0
    
    # Rössler attractor parameters
    rossler_a: float = 0.2
    rossler_b: float = 0.2
    rossler_c: float = 5.7
    
    # Chua's circuit attractor parameters
    chua_alpha: float = 15.35
    chua_beta: float = 28.0
    chua_m0: float = -1.143
    chua_m1: float = -0.714
    
    # Attractor switching interval (in steps)
    attractor_switch_interval: int = 5000
    
    # Chaos injection gain (controls influence on dynamics)
    chaos_injection_gain: float = 0.005
    
    # Transition smoothing factor for attractor switching
    attractor_transition_smoothing: float = 0.95


@dataclass
class TrainingConfig:
    """Configuration for training and optimization."""
    
    # Annealing initial temperature (T_0)
    annealing_initial_temp: float = 10.0
    
    # Annealing rate (τ) - in steps
    annealing_rate: int = 1000
    
    # Learning rate for main dynamics
    learning_rate: float = 1e-4
    
    # Learning rate for encoders (typically smaller)
    encoder_learning_rate: float = 5e-5
    
    # Batch size for training
    batch_size: int = 32
    
    # Gradient clipping threshold
    gradient_clip_threshold: float = 1.0
    
    # Weight decay for regularization
    weight_decay: float = 1e-5
    
    # Number of training epochs
    num_epochs: int = 100
    
    # Validation frequency (in epochs)
    validation_frequency: int = 5
    
    # Checkpoint saving frequency (in epochs)
    checkpoint_frequency: int = 10


@dataclass
class NeuralODEConfig:
    """Configuration for Neural ODE solver."""
    
    # Integration method ('dopri5', 'adams', 'rk4', etc.)
    integration_method: str = "dopri5"
    
    # Absolute tolerance for adaptive stepping
    atol: float = 1e-6
    
    # Relative tolerance for adaptive stepping
    rtol: float = 1e-5
    
    # Maximum integration steps
    max_steps: int = 1000
    
    # Time step for fixed-step methods
    dt: float = 0.01


@dataclass
class EncoderConfig:
    """Configuration for dual-channel encoders."""

    # Semantic encoder configuration
    semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_hidden_dim: int = 512
    semantic_num_layers: int = 4
    semantic_num_heads: int = 8

    # Physical encoder configuration
    physical_hidden_dim: int = 512
    physical_num_layers: int = 4
    physical_state_dim: int = 128  # SSM state dimension

    # Cross-attention configuration
    cross_attention_heads: int = 8
    cross_attention_dropout: float = 0.1


@dataclass
class NumericsConfig:
    """Configuration for numerical solvers and computation optimization."""

    # ODE 求解器类型: euler | rk4 | imex | verlet
    solver_type: str = "imex"

    # 是否启用谱范数约束
    spectral_norm_enabled: bool = True

    # 注意力模式: linear | full
    attention_mode: str = "linear"

    # 是否启用梯度检查点以节省显存
    checkpointing_enabled: bool = False

    # 是否启用傅里叶变换加速
    fourier_enabled: bool = False

    # IMEX 求解器更新间隔（步数）
    imex_update_interval: int = 100

    # IMEX 时间步安全因子
    imex_dt_safety_factor: float = 0.9


@dataclass
class MetaCognitiveConfig:
    """Configuration for meta-cognitive layers."""
    
    # L0 perception layer
    l0_hidden_dim: int = 256
    
    # L1 self-state layer
    l1_hidden_dim: int = 512
    
    # L2 meta-cognitive layer
    l2_hidden_dim: int = 128
    l2_projection_dim: int = 64  # Johnson-Lindenstrauss projection
    control_output_dim: int = 128  # L2 control signal output dimension
    
    # L2 perturbation noise for training
    l2_perturbation_noise: float = 0.05
    
    # L2 ablation threshold (minimum L1 function retention)
    l2_ablation_threshold: float = 0.4
    
    # 好奇心引擎配置
    enable_curiosity: bool = False  # 是否启用好奇心引擎（默认关闭，向后兼容）
    curiosity_novelty_weight: float = 0.4  # 新奇度权重
    curiosity_complexity_weight: float = 0.3  # 复杂度权重
    curiosity_uncertainty_weight: float = 0.3  # 不确定性权重
    curiosity_exploration_rate: float = 0.1  # 探索率（epsilon-greedy）
    curiosity_exploration_decay: float = 0.995  # 探索率衰减
    curiosity_min_exploration_rate: float = 0.01  # 最小探索率
    curiosity_decay_rate: float = 0.9  # 好奇心衰减率
    curiosity_history_window: int = 100  # 历史窗口大小
    
    # 元意识引擎配置（动态层级）
    enable_meta_consciousness: bool = False  # 是否启用元意识引擎（动态层级）
    meta_consciousness_window_time: float = 1.0  # 元意识场窗口时间 τ_M
    meta_consciousness_emergence_threshold: float = 1.0  # 涌现阈值 M_0
    meta_consciousness_level_spacing: float = 0.5  # 层级间距 ΔM
    meta_consciousness_awareness_gate_threshold: float = 0.5  # 觉知梯度门控阈值 G_th


@dataclass
class ValidationConfig:
    """Configuration for validation and emergence testing."""
    
    # P0 core validation parameters
    p0_open_loop_hours: float = 72.0
    p0_max_baseline_drift: float = 0.1
    
    # Dynamics alignment validation
    alignment_num_steps: List[int] = field(default_factory=lambda: [1, 10, 100, 1000])
    alignment_max_error: float = 0.05
    
    # Lyapunov exponent window
    lyapunov_window: int = 1000
    
    # Self-correlation window
    autocorrelation_window: int = 500
    
    # Emergence criteria thresholds
    emergence_intent_entropy_threshold: float = 0.5
    emergence_transfer_score_threshold: float = 0.6
    emergence_recovery_time_threshold: float = 100.0


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""
    
    # Log level (DEBUG, INFO, WARNING, ERROR)
    log_level: str = "INFO"
    
    # Log file path
    log_file: str = "logs/chronos_self.log"
    
    # Log rotation size (in MB)
    log_rotation_size: int = 10
    
    # Number of backup logs
    log_backup_count: int = 5
    
    # Enable console logging
    console_logging: bool = True
    
    # Enable file logging
    file_logging: bool = True
    
    # Metrics logging interval (in steps)
    metrics_interval: int = 100
    
    # Tensorboard log directory
    tensorboard_dir: str = "runs/chronos_self"


@dataclass
class PathsConfig:
    """Configuration for file paths and directories."""
    
    # Root directory for all data
    data_root: str = "data"
    
    # Model checkpoint directory
    checkpoint_dir: str = "checkpoints"
    
    # Vector database directory for keyframes
    vector_db_dir: str = "vector_db"
    
    # Configuration file directory
    config_dir: str = "configs"
    
    # Logs directory
    logs_dir: str = "logs"
    
    # Results directory
    results_dir: str = "results"


@dataclass
class ChronosConfig:
    """
    Master configuration class for Chronos-Self system.
    
    This class aggregates all configuration sections and provides
    methods for loading/saving configurations.
    """
    
    # Configuration sections
    dim: DimensionalityConfig = field(default_factory=DimensionalityConfig)
    memory_temporal: MemoryTemporalConfig = field(default_factory=MemoryTemporalConfig)
    coupling_stability: CouplingStabilityConfig = field(default_factory=CouplingStabilityConfig)
    chaos_injection: ChaosInjectionConfig = field(default_factory=ChaosInjectionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    neural_ode: NeuralODEConfig = field(default_factory=NeuralODEConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    numerics: NumericsConfig = field(default_factory=NumericsConfig)
    meta_cognitive: MetaCognitiveConfig = field(default_factory=MetaCognitiveConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Device configuration
    device: str = "cuda"  # 'cuda' or 'cpu'
    
    # Enable mixed precision training
    use_amp: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to nested dictionary."""
        import dataclasses
        
        def asdict_recursive(obj):
            if dataclasses.is_dataclass(obj):
                return {k: asdict_recursive(v) for k, v in dataclasses.asdict(obj).items()}
            return obj
        
        return asdict_recursive(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ChronosConfig":
        """Create configuration from nested dictionary."""
        def dict_to_dataclass(d: Dict[str, Any], dataclass_type):
            if not dataclasses.is_dataclass(dataclass_type):
                return d
            
            fields = {}
            for field_info in dataclasses.fields(dataclass_type):
                field_name = field_info.name
                if field_name in d:
                    field_value = d[field_name]
                    if dataclasses.is_dataclass(field_info.type):
                        fields[field_name] = dict_to_dataclass(field_value, field_info.type)
                    else:
                        fields[field_name] = field_value
            
            return dataclass_type(**fields)
        
        return dict_to_dataclass(config_dict, cls)
    
    def save(self, filepath: Path) -> None:
        """Save configuration to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: Path) -> "ChronosConfig":
        """Load configuration from JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def update(self, **kwargs) -> None:
        """
        Update configuration parameters.
        
        Supports nested updates using dot notation:
            config.update(dim__fast_variable_dim=4096)
        """
        for key, value in kwargs.items():
            if '__' in key:
                # Handle nested updates
                parts = key.split('__')
                obj = self
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            else:
                setattr(self, key, value)


# Import dataclasses module for the methods above
import dataclasses


# Global configuration instance
_global_config: Optional[ChronosConfig] = None


def get_config() -> ChronosConfig:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = ChronosConfig()
    return _global_config


def set_config(config: ChronosConfig) -> None:
    """Set the global configuration instance."""
    global _global_config
    _global_config = config


def init_config(config_path: Optional[Path] = None, **kwargs) -> ChronosConfig:
    """
    Initialize configuration from file or create new one with updates.
    
    Args:
        config_path: Path to JSON configuration file
        **kwargs: Configuration updates (supports nested updates)
    
    Returns:
        Initialized configuration instance
    """
    global _global_config
    
    if config_path and config_path.exists():
        _global_config = ChronosConfig.load(config_path)
    else:
        _global_config = ChronosConfig()
    
    if kwargs:
        _global_config.update(**kwargs)
    
    return _global_config