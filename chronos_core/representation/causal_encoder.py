"""
Causal and Physical Constraints Encoder
========================================

This module encodes:
- Physical constraints (energy conservation, momentum conservation, etc.)
- Causal chains (event sequences, causal relationships)
- Implements causal reasoning module

This encoder processes structured causal information and physical constraints
to maintain consistency in the physical representation.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
import logging
import math

from .ssm import StackedSSM, check_numerical_stability
from ..utils.config import EncoderConfig

logger = logging.getLogger(__name__)


class PhysicalConstraints:
    """
    Container for physical constraint information.
    """

    def __init__(
        self,
        energy_conservation: Optional[float] = None,
        momentum_conservation: Optional[torch.Tensor] = None,
        entropy_constraint: Optional[float] = None,
        gravity_strength: Optional[float] = None,
        friction_coefficient: Optional[float] = None,
        collision_constraints: Optional[List[Dict[str, Any]]] = None,
        boundary_constraints: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Initialize physical constraints.

        Args:
            energy_conservation: Energy conservation factor (0.0-1.0)
            momentum_conservation: Momentum conservation vector (3D)
            entropy_constraint: Entropy constraint value
            gravity_strength: Gravity strength (normalized)
            friction_coefficient: Friction coefficient
            collision_constraints: List of collision constraints
            boundary_constraints: List of boundary constraints
        """
        self.energy_conservation = energy_conservation if energy_conservation is not None else 1.0
        self.momentum_conservation = momentum_conservation if momentum_conservation is not None else torch.ones(3)
        self.entropy_constraint = entropy_constraint if entropy_constraint is not None else 0.0
        self.gravity_strength = gravity_strength if gravity_strength is not None else 0.5
        self.friction_coefficient = friction_coefficient if friction_coefficient is not None else 0.3
        self.collision_constraints = collision_constraints if collision_constraints is not None else []
        self.boundary_constraints = boundary_constraints if boundary_constraints is not None else []

    def to_tensor(self, max_constraints: int = 5) -> torch.Tensor:
        """
        Convert physical constraints to tensor.

        Args:
            max_constraints: Maximum number of collision/boundary constraints to encode

        Returns:
            Tensor encoding physical constraints
        """
        # Fundamental physical constraints (8 dims)
        fundamental_constraints = torch.tensor([
            self.energy_conservation,
            self.momentum_conservation[0],
            self.momentum_conservation[1],
            self.momentum_conservation[2],
            self.entropy_constraint,
            self.gravity_strength,
            self.friction_coefficient,
            1.0,  # Placeholder for additional constraint
        ], dtype=torch.float32)

        # Encode collision constraints (each: 4 dims)
        collision_features = []
        for i, constraint in enumerate(self.collision_constraints[:max_constraints]):
            collision_feat = torch.tensor([
                constraint.get('object_id', 0.0),
                constraint.get('force', 0.0),
                constraint.get('direction', 0.0),
                constraint.get('duration', 0.0),
            ], dtype=torch.float32)
            collision_features.append(collision_feat)

        # Pad collision features
        while len(collision_features) < max_constraints:
            collision_features.append(torch.zeros(4, dtype=torch.float32))

        collisions_tensor = torch.stack(collision_features).flatten()

        # Encode boundary constraints (each: 3 dims)
        boundary_features = []
        for i, constraint in enumerate(self.boundary_constraints[:max_constraints]):
            boundary_feat = torch.tensor([
                constraint.get('position', 0.0),
                constraint.get('normal', 0.0),
                constraint.get('stiffness', 0.0),
            ], dtype=torch.float32)
            boundary_features.append(boundary_feat)

        # Pad boundary features
        while len(boundary_features) < max_constraints:
            boundary_features.append(torch.zeros(3, dtype=torch.float32))

        boundaries_tensor = torch.stack(boundary_features).flatten()

        # Concatenate all constraints
        # Total dims: 8 + max_constraints*4 + max_constraints*3
        # For max_constraints=5: 8 + 20 + 15 = 43
        constraints_tensor = torch.cat([
            fundamental_constraints,
            collisions_tensor,
            boundaries_tensor,
        ])

        return constraints_tensor


class CausalChain:
    """
    Container for causal chain information (event sequences and causal relationships).
    """

    def __init__(
        self,
        events: Optional[List[Dict[str, Any]]] = None,
        causal_relations: Optional[List[Dict[str, Any]]] = None,
        causal_graph: Optional[torch.Tensor] = None,
    ):
        """
        Initialize causal chain.

        Args:
            events: List of events with timestamps, types, and effects
            causal_relations: List of causal relations (cause -> effect)
            causal_graph: Optional causal graph matrix
        """
        self.events = events if events is not None else []
        self.causal_relations = causal_relations if causal_relations is not None else []
        self.causal_graph = causal_graph

    def to_tensor(self, max_events: int = 10, max_relations: int = 15) -> torch.Tensor:
        """
        Convert causal chain to tensor.

        Args:
            max_events: Maximum number of events to encode
            max_relations: Maximum number of causal relations to encode

        Returns:
            Tensor encoding causal chain
        """
        # Encode events (each event: 6 dims)
        event_features = []
        for i, event in enumerate(self.events[:max_events]):
            event_feat = torch.tensor([
                event.get('timestamp', 0.0),
                event.get('type', 0.0),
                event.get('intensity', 0.5),
                event.get('duration', 0.0),
                event.get('location', 0.0),
                event.get('effect_strength', 0.5),
            ], dtype=torch.float32)
            event_features.append(event_feat)

        # Pad event features
        while len(event_features) < max_events:
            event_features.append(torch.zeros(6, dtype=torch.float32))

        events_tensor = torch.stack(event_features).flatten()

        # Encode causal relations (each relation: 4 dims)
        relation_features = []
        for i, relation in enumerate(self.causal_relations[:max_relations]):
            relation_feat = torch.tensor([
                relation.get('cause_id', 0.0),
                relation.get('effect_id', 0.0),
                relation.get('strength', 0.5),
                relation.get('delay', 0.0),
            ], dtype=torch.float32)
            relation_features.append(relation_feat)

        # Pad relation features
        while len(relation_features) < max_relations:
            relation_features.append(torch.zeros(4, dtype=torch.float32))

        relations_tensor = torch.stack(relation_features).flatten()

        # Concatenate events and relations
        # Total dims: max_events*6 + max_relations*4
        # For max_events=10, max_relations=15: 60 + 60 = 120
        causal_tensor = torch.cat([events_tensor, relations_tensor])

        return causal_tensor


class CausalReasoningModule(nn.Module):
    """
    Module for causal reasoning and inference.

    This module processes causal chains and performs causal reasoning
    to identify cause-effect relationships and predict outcomes.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_heads: int = 8,
    ):
        """
        Initialize causal reasoning module.

        Args:
            input_dim: Input dimension from causal chain encoder
            hidden_dim: Hidden dimension for reasoning
            output_dim: Output dimension for causal predictions
            num_heads: Number of attention heads for causal graph attention
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Causal attention mechanism
        self.causal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )

        # Causal inference network
        self.inference_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Causal prediction network
        self.prediction_net = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        logger.info(
            f"Initialized CausalReasoningModule with input_dim={input_dim}, "
            f"output_dim={output_dim}"
        )

    def forward(
        self,
        causal_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through causal reasoning module.

        Args:
            causal_input: Input tensor from causal chain encoder

        Returns:
            causal_output: Output tensor with causal predictions
            metadata: Dictionary with reasoning metadata
        """
        batch_size, seq_len, input_dim = causal_input.shape

        # Expand input to hidden dimension
        hidden = self.inference_net(causal_input)

        # Apply causal attention (self-attention for causal relationships)
        attn_output, attn_weights = self.causal_attention(
            hidden, hidden, hidden,
            need_weights=True,
        )

        # Combine attention output with original hidden
        combined = hidden + attn_output

        # Generate causal predictions
        causal_output = self.prediction_net(combined)

        # Prepare metadata
        metadata = {
            "attention_weights": attn_weights,
            "causal_strength": torch.norm(causal_output).item(),
        }

        return causal_output, metadata


class CausalEncoder(nn.Module):
    """
    Encoder for physical constraints and causal chains.

    This encoder processes physical constraints and causal relationships
    to maintain consistency in the physical representation and enable
    causal reasoning.

    Output dimension: 128 (to be integrated into X_log)
    """

    def __init__(self, config: EncoderConfig):
        """
        Initialize causal encoder.

        Args:
            config: Encoder configuration
        """
        super().__init__()

        self.config = config
        # Input dimensions from PhysicalConstraints and CausalChain
        self.max_constraints = 5
        self.max_events = 10
        self.max_relations = 15
        self.constraints_dim = 8 + self.max_constraints * 7  # 43
        self.causal_dim = self.max_events * 6 + self.max_relations * 4  # 120
        self.input_dim = self.constraints_dim + self.causal_dim  # 163
        self.output_dim = 128  # Output for causal/constraint features

        # Physical constraints embedding
        self.constraints_embedding = nn.Sequential(
            nn.Linear(self.constraints_dim, config.physical_hidden_dim // 2),
            nn.LayerNorm(config.physical_hidden_dim // 2),
            nn.ReLU(),
        )

        # Causal chain embedding
        self.causal_embedding = nn.Sequential(
            nn.Linear(self.causal_dim, config.physical_hidden_dim // 2),
            nn.LayerNorm(config.physical_hidden_dim // 2),
            nn.ReLU(),
        )

        # Combine constraints and causal features
        self.combine_layer = nn.Sequential(
            nn.Linear(config.physical_hidden_dim, config.physical_hidden_dim),
            nn.LayerNorm(config.physical_hidden_dim),
            nn.ReLU(),
        )

        # Causal reasoning module
        self.causal_reasoning = CausalReasoningModule(
            input_dim=config.physical_hidden_dim,
            hidden_dim=config.physical_hidden_dim,
            output_dim=self.output_dim,
            num_heads=8,
        )

        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )

        logger.info(
            f"Initialized CausalEncoder with input_dim={self.input_dim}, "
            f"output_dim={self.output_dim}"
        )

    def forward(
        self,
        physical_constraints: Optional[PhysicalConstraints] = None,
        causal_chain: Optional[CausalChain] = None,
        input_tensor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through causal encoder.

        Args:
            physical_constraints: Physical constraints object
            causal_chain: Causal chain object
            input_tensor: Pre-computed input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            causal_output: Output causal/constraint features of shape (batch_size, seq_len, 128)
            metadata: Dictionary containing encoding metadata
        """
        # Convert inputs to tensor if needed
        if input_tensor is None:
            # Create default objects if not provided
            if physical_constraints is None:
                physical_constraints = PhysicalConstraints()
            if causal_chain is None:
                causal_chain = CausalChain()

            constraints_tensor = physical_constraints.to_tensor(max_constraints=self.max_constraints)
            causal_tensor = causal_chain.to_tensor(
                max_events=self.max_events,
                max_relations=self.max_relations
            )

            input_tensor = torch.cat([constraints_tensor, causal_tensor])

            # Add batch and sequence dimensions if missing
            if input_tensor.dim() == 1:
                input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
            elif input_tensor.dim() == 2:
                input_tensor = input_tensor.unsqueeze(0)

        # Ensure correct shape
        assert input_tensor.shape[-1] == self.input_dim

        batch_size, seq_len, _ = input_tensor.shape

        # Numerical stability check
        is_stable, msg = check_numerical_stability(input_tensor, "causal_input")
        if not is_stable:
            logger.warning(msg)

        # Split input into constraints and causal parts
        constraints_part = input_tensor[:, :, :self.constraints_dim]
        causal_part = input_tensor[:, :, self.constraints_dim:]

        # Embed constraints and causal chains
        constraints_emb = self.constraints_embedding(constraints_part)
        causal_emb = self.causal_embedding(causal_part)

        # Combine embeddings
        combined = torch.cat([constraints_emb, causal_emb], dim=-1)
        combined = self.combine_layer(combined)

        # Apply causal reasoning
        causal_reasoning_out, reasoning_metadata = self.causal_reasoning(combined)

        # Final projection
        causal_output = self.output_proj(causal_reasoning_out)

        # Numerical stability check on output
        is_stable, msg = check_numerical_stability(causal_output, "causal_output")
        if not is_stable:
            logger.warning(msg)

        # Prepare metadata
        metadata = {
            "input_norm": torch.norm(input_tensor).item(),
            "output_norm": torch.norm(causal_output).item(),
            "is_stable": is_stable,
            "reasoning_metadata": reasoning_metadata,
        }

        return causal_output, metadata

    def encode_single(
        self,
        physical_constraints: PhysicalConstraints,
        causal_chain: CausalChain,
    ) -> torch.Tensor:
        """
        Encode single physical constraints and causal chain (no sequence).

        Args:
            physical_constraints: Physical constraints object
            causal_chain: Causal chain object

        Returns:
            causal_output: Output tensor of shape (128,)
        """
        # Treat as sequence of length 1
        constraints_tensor = physical_constraints.to_tensor(max_constraints=self.max_constraints)
        causal_tensor = causal_chain.to_tensor(
            max_events=self.max_events,
            max_relations=self.max_relations
        )
        input_tensor = torch.cat([constraints_tensor, causal_tensor]).unsqueeze(0).unsqueeze(0)
        causal_output, _ = self.forward(input_tensor=input_tensor)
        return causal_output.squeeze(0).squeeze(0)

    def extract_from_external_input(
        self,
        external_input_dict: Dict[str, Any],
    ) -> Tuple[PhysicalConstraints, CausalChain]:
        """
        Extract physical constraints and causal information from ExternalInput.

        Args:
            external_input_dict: ExternalInput as dictionary

        Returns:
            Tuple of PhysicalConstraints and CausalChain objects
        """
        metadata = external_input_dict.get("metadata", {})

        # Extract physical constraints
        energy_conservation = metadata.get("energy_conservation", 1.0)
        gravity_strength = metadata.get("gravity_strength", 0.5)
        friction_coefficient = metadata.get("friction_coefficient", 0.3)

        physical_constraints = PhysicalConstraints(
            energy_conservation=energy_conservation,
            gravity_strength=gravity_strength,
            friction_coefficient=friction_coefficient,
        )

        # Extract causal chain
        events = metadata.get("events", [])
        causal_relations = metadata.get("causal_relations", [])

        causal_chain = CausalChain(
            events=events,
            causal_relations=causal_relations,
        )

        return physical_constraints, causal_chain