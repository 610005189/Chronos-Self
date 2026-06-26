"""
Logical Encoder - Main Class
=============================

This module implements the LogicalEncoder, which is the main encoder for
logical-physical representation in Chronos-Self.

The LogicalEncoder:
- Inputs: Structured physical information (object states, spatial positions, causal relationships)
- Outputs: Logical-physical vector X_log (dimension 512)
- Explicitly separates into proprioceptive flow X_proprio (256) and world flow X_world (256)
- Integrates physical constraints and causal chain encoding (128 dimensions, integrated into X_log)
- Supports sequence modeling and long-range dependencies through SSM architecture
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import logging

from .ssm import check_numerical_stability
from .proprioceptive_encoder import ProprioceptiveEncoder, ProprioceptiveState
from .world_encoder import WorldEncoder, WorldState
from .causal_encoder import CausalEncoder, PhysicalConstraints, CausalChain
from ..utils.config import EncoderConfig

logger = logging.getLogger(__name__)


class LogicalEncoder(nn.Module):
    """
    Main Logical Encoder for logical-physical representation.

    This encoder integrates:
    1. Proprioceptive flow encoder (internal agent state)
    2. World flow encoder (external environment state)
    3. Causal encoder (physical constraints and causal chains)

    Output: X_log (512 dimensions) = X_proprio (256) + X_world (256)
    The causal/constraint features (128) are integrated into X_log through
    attention-based fusion mechanism.
    """

    def __init__(self, config: EncoderConfig):
        """
        Initialize LogicalEncoder.

        Args:
            config: Encoder configuration from ChronosConfig.encoder
        """
        super().__init__()

        self.config = config

        # Output dimensions
        self.proprio_dim = 256  # Proprioceptive flow dimension
        self.world_dim = 256  # World flow dimension
        self.causal_dim = 128  # Causal/constraint dimension
        self.output_dim = 512  # Total X_log dimension

        # Initialize sub-encoders
        self.proprioceptive_encoder = ProprioceptiveEncoder(config)
        self.world_encoder = WorldEncoder(config)
        self.causal_encoder = CausalEncoder(config)

        # Fusion mechanism to integrate causal/constraint features
        # Cross-attention between proprioceptive/world and causal features
        self.causal_attention_proprio = nn.MultiheadAttention(
            embed_dim=self.proprio_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        self.causal_attention_world = nn.MultiheadAttention(
            embed_dim=self.world_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )

        # Projection to expand causal features for attention
        self.causal_proj_proprio = nn.Linear(self.causal_dim, self.proprio_dim)
        self.causal_proj_world = nn.Linear(self.causal_dim, self.world_dim)

        # Final layer normalization
        self.norm_proprio = nn.LayerNorm(self.proprio_dim)
        self.norm_world = nn.LayerNorm(self.world_dim)
        self.norm_log = nn.LayerNorm(self.output_dim)

        logger.info(
            f"Initialized LogicalEncoder with proprio_dim={self.proprio_dim}, "
            f"world_dim={self.world_dim}, causal_dim={self.causal_dim}, "
            f"output_dim={self.output_dim}"
        )

    def forward(
        self,
        proprioceptive_state: Optional[ProprioceptiveState] = None,
        world_state: Optional[WorldState] = None,
        physical_constraints: Optional[PhysicalConstraints] = None,
        causal_chain: Optional[CausalChain] = None,
        proprioceptive_tensor: Optional[torch.Tensor] = None,
        world_tensor: Optional[torch.Tensor] = None,
        causal_tensor: Optional[torch.Tensor] = None,
        hidden_states: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through LogicalEncoder.

        Args:
            proprioceptive_state: Proprioceptive state object
            world_state: World state object
            physical_constraints: Physical constraints object
            causal_chain: Causal chain object
            proprioceptive_tensor: Pre-computed proprioceptive tensor
            world_tensor: Pre-computed world tensor
            causal_tensor: Pre-computed causal tensor
            hidden_states: Optional dictionary of hidden states for SSM sub-encoders

        Returns:
            X_log: Complete logical-physical vector of shape (batch_size, seq_len, 512)
            X_proprio: Proprioceptive flow of shape (batch_size, seq_len, 256)
            X_world: World flow of shape (batch_size, seq_len, 256)
            X_causal: Causal/constraint features of shape (batch_size, seq_len, 128)
            metadata: Dictionary containing encoding metadata
        """
        # Initialize hidden states if not provided
        if hidden_states is None:
            hidden_states = {
                "proprioceptive": None,
                "world": None,
            }

        # Encode proprioceptive flow
        X_proprio, proprio_hidden, proprio_meta = self.proprioceptive_encoder(
            proprioceptive_state=proprioceptive_state,
            proprioceptive_tensor=proprioceptive_tensor,
            hidden_state=hidden_states.get("proprioceptive"),
        )

        # Encode world flow
        X_world, world_hidden, world_meta = self.world_encoder(
            world_state=world_state,
            world_tensor=world_tensor,
            hidden_state=hidden_states.get("world"),
        )

        # Encode causal/constraint features
        X_causal, causal_meta = self.causal_encoder(
            physical_constraints=physical_constraints,
            causal_chain=causal_chain,
            input_tensor=causal_tensor,
        )

        # Ensure dimensions match for fusion
        batch_size, seq_len = X_proprio.shape[:2]

        # Expand causal features to match proprioceptive and world dimensions
        causal_for_proprio = self.causal_proj_proprio(X_causal)
        causal_for_world = self.causal_proj_world(X_causal)

        # Apply cross-attention: proprioceptive features attend to causal features
        proprio_with_causal, proprio_attn_weights = self.causal_attention_proprio(
            X_proprio, causal_for_proprio, causal_for_proprio,
            need_weights=True,
        )
        X_proprio_enhanced = self.norm_proprio(X_proprio + proprio_with_causal)

        # Apply cross-attention: world features attend to causal features
        world_with_causal, world_attn_weights = self.causal_attention_world(
            X_world, causal_for_world, causal_for_world,
            need_weights=True,
        )
        X_world_enhanced = self.norm_world(X_world + world_with_causal)

        # Concatenate enhanced proprioceptive and world flows
        X_log = torch.cat([X_proprio_enhanced, X_world_enhanced], dim=-1)

        # Apply final layer normalization
        X_log = self.norm_log(X_log)

        # Numerical stability checks
        is_stable_log, msg_log = check_numerical_stability(X_log, "X_log")
        if not is_stable_log:
            logger.warning(msg_log)

        # Prepare comprehensive metadata
        metadata = {
            "X_log_norm": torch.norm(X_log).item(),
            "X_proprio_norm": torch.norm(X_proprio_enhanced).item(),
            "X_world_norm": torch.norm(X_world_enhanced).item(),
            "X_causal_norm": torch.norm(X_causal).item(),
            "proprioceptive_metadata": proprio_meta,
            "world_metadata": world_meta,
            "causal_metadata": causal_meta,
            "proprio_attention_weights": proprio_attn_weights,
            "world_attention_weights": world_attn_weights,
            "is_stable": is_stable_log,
            "hidden_states": {
                "proprioceptive": proprio_hidden,
                "world": world_hidden,
            },
        }

        return X_log, X_proprio_enhanced, X_world_enhanced, X_causal, metadata

    def encode_single(
        self,
        proprioceptive_state: ProprioceptiveState,
        world_state: WorldState,
        physical_constraints: Optional[PhysicalConstraints] = None,
        causal_chain: Optional[CausalChain] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode single physical state (no sequence).

        Args:
            proprioceptive_state: Proprioceptive state object
            world_state: World state object
            physical_constraints: Physical constraints object (optional)
            causal_chain: Causal chain object (optional)

        Returns:
            X_log: Logical-physical vector of shape (512,)
            X_proprio: Proprioceptive flow of shape (256,)
            X_world: World flow of shape (256,)
            X_causal: Causal features of shape (128,)
        """
        # Treat as sequence of length 1
        X_log, X_proprio, X_world, X_causal, _ = self.forward(
            proprioceptive_state=proprioceptive_state,
            world_state=world_state,
            physical_constraints=physical_constraints,
            causal_chain=causal_chain,
        )
        return (
            X_log.squeeze(0).squeeze(0),
            X_proprio.squeeze(0).squeeze(0),
            X_world.squeeze(0).squeeze(0),
            X_causal.squeeze(0).squeeze(0),
        )

    def encode_from_external_input(
        self,
        external_input_dict: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode logical-physical representation from ExternalInput dictionary.

        Args:
            external_input_dict: ExternalInput as dictionary (from ExternalInput.to_dict())

        Returns:
            X_log: Logical-physical vector of shape (512,)
            X_proprio: Proprioceptive flow of shape (256,)
            X_world: World flow of shape (256,)
            X_causal: Causal features of shape (128,)
        """
        # Extract proprioceptive state
        proprioceptive_state = self.proprioceptive_encoder.update_from_self_state(
            external_input_dict.get("self_state", {})
        )

        # Extract world state
        world_state = self.world_encoder.update_from_external_input(external_input_dict)

        # Extract physical constraints and causal chain
        physical_constraints, causal_chain = self.causal_encoder.extract_from_external_input(
            external_input_dict
        )

        return self.encode_single(
            proprioceptive_state=proprioceptive_state,
            world_state=world_state,
            physical_constraints=physical_constraints,
            causal_chain=causal_chain,
        )

    def get_output_dimensions(self) -> Dict[str, int]:
        """
        Get output dimensions for each component.

        Returns:
            Dictionary with dimension information
        """
        return {
            "X_log": self.output_dim,
            "X_proprio": self.proprio_dim,
            "X_world": self.world_dim,
            "X_causal": self.causal_dim,
        }

    def validate_output(
        self,
        X_log: torch.Tensor,
        X_proprio: torch.Tensor,
        X_world: torch.Tensor,
        X_causal: torch.Tensor,
    ) -> Tuple[bool, List[str]]:
        """
        Validate output dimensions and numerical stability.

        Args:
            X_log: Logical-physical vector
            X_proprio: Proprioceptive flow
            X_world: World flow
            X_causal: Causal features

        Returns:
            (is_valid, error_messages): Validation result and error messages
        """
        errors = []

        # Check dimensions
        expected_dims = self.get_output_dimensions()
        if X_log.shape[-1] != expected_dims["X_log"]:
            errors.append(
                f"X_log dimension mismatch: expected {expected_dims['X_log']}, "
                f"got {X_log.shape[-1]}"
            )
        if X_proprio.shape[-1] != expected_dims["X_proprio"]:
            errors.append(
                f"X_proprio dimension mismatch: expected {expected_dims['X_proprio']}, "
                f"got {X_proprio.shape[-1]}"
            )
        if X_world.shape[-1] != expected_dims["X_world"]:
            errors.append(
                f"X_world dimension mismatch: expected {expected_dims['X_world']}, "
                f"got {X_world.shape[-1]}"
            )
        if X_causal.shape[-1] != expected_dims["X_causal"]:
            errors.append(
                f"X_causal dimension mismatch: expected {expected_dims['X_causal']}, "
                f"got {X_causal.shape[-1]}"
            )

        # Check numerical stability
        for name, tensor in [
            ("X_log", X_log),
            ("X_proprio", X_proprio),
            ("X_world", X_world),
            ("X_causal", X_causal),
        ]:
            if torch.isnan(tensor).any():
                errors.append(f"{name} contains NaN values")
            if torch.isinf(tensor).any():
                errors.append(f"{name} contains Inf values")

        is_valid = len(errors) == 0
        return is_valid, errors


# Convenience function for creating LogicalEncoder
def create_logical_encoder(config: Optional[EncoderConfig] = None) -> LogicalEncoder:
    """
    Create LogicalEncoder with optional configuration.

    Args:
        config: Optional encoder configuration (uses default if not provided)

    Returns:
        LogicalEncoder instance
    """
    if config is None:
        from ..utils.config import get_config
        config = get_config().encoder

    return LogicalEncoder(config)