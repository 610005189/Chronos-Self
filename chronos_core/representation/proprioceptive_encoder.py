"""
Proprioceptive Flow Encoder
============================

This module encodes the agent's internal state (proprioceptive information):
- Posture state (position, orientation, velocity)
- Energy level (computational resources, battery status)
- Resource utilization (CPU, memory, GPU usage)

Output dimension: 256
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import logging

from .ssm import StackedSSM, check_numerical_stability
from ..utils.config import EncoderConfig

logger = logging.getLogger(__name__)


class ProprioceptiveState:
    """
    Container for proprioceptive state information.
    """

    def __init__(
        self,
        position: Optional[torch.Tensor] = None,
        orientation: Optional[torch.Tensor] = None,
        velocity: Optional[torch.Tensor] = None,
        energy_level: Optional[float] = None,
        battery_status: Optional[float] = None,
        cpu_usage: Optional[float] = None,
        memory_usage: Optional[float] = None,
        gpu_usage: Optional[float] = None,
    ):
        """
        Initialize proprioceptive state.

        Args:
            position: Position in space (3D coordinates)
            orientation: Orientation (quaternion or euler angles, 4D or 3D)
            velocity: Velocity vector (3D)
            energy_level: Overall energy level (0.0-1.0)
            battery_status: Battery status (0.0-1.0)
            cpu_usage: CPU utilization (0.0-1.0)
            memory_usage: Memory utilization (0.0-1.0)
            gpu_usage: GPU utilization (0.0-1.0)
        """
        # Posture state (10 dimensions: 3 pos + 4 orient + 3 vel)
        self.position = position if position is not None else torch.zeros(3)
        self.orientation = orientation if orientation is not None else torch.zeros(4)
        self.velocity = velocity if velocity is not None else torch.zeros(3)

        # Energy and resource state (4 dimensions)
        self.energy_level = energy_level if energy_level is not None else 1.0
        self.battery_status = battery_status if battery_status is not None else 1.0
        self.cpu_usage = cpu_usage if cpu_usage is not None else 0.0
        self.memory_usage = memory_usage if memory_usage is not None else 0.0
        self.gpu_usage = gpu_usage if gpu_usage is not None else 0.0

    def to_tensor(self) -> torch.Tensor:
        """
        Convert proprioceptive state to a single tensor.

        Returns:
            Tensor of shape (15,) containing all proprioceptive information
        """
        # Posture state
        posture = torch.cat([
            self.position,
            self.orientation,
            self.velocity,
        ])

        # Energy and resource state
        resources = torch.tensor([
            self.energy_level,
            self.battery_status,
            self.cpu_usage,
            self.memory_usage,
            self.gpu_usage,
        ])

        return torch.cat([posture, resources])


class ProprioceptiveEncoder(nn.Module):
    """
    Encoder for proprioceptive flow (internal state).

    This encoder processes the agent's internal state information using SSM
    to capture temporal dynamics of posture, energy, and resource utilization.

    Output: X_proprio (dimension 256)
    """

    def __init__(self, config: EncoderConfig):
        """
        Initialize proprioceptive encoder.

        Args:
            config: Encoder configuration
        """
        super().__init__()

        self.config = config
        self.input_dim = 15  # Posture (10: 3 pos + 4 orient + 3 vel) + Resources (5) = 15
        self.output_dim = 256  # Fixed output dimension for proprioceptive flow

        # Input embedding to project low-dim proprioceptive state to higher dim
        self.input_embedding = nn.Sequential(
            nn.Linear(self.input_dim, config.physical_hidden_dim),
            nn.LayerNorm(config.physical_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.physical_hidden_dim, config.physical_hidden_dim),
        )

        # Stacked SSM for temporal modeling
        self.ssm = StackedSSM(
            input_dim=config.physical_hidden_dim,
            state_dim=config.physical_state_dim,
            num_layers=config.physical_num_layers // 2,  # Half for proprioceptive
            expansion_factor=2,
            dropout=0.1,
        )

        # Output projection to 256 dimensions
        self.output_proj = nn.Sequential(
            nn.Linear(config.physical_hidden_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )

        logger.info(
            f"Initialized ProprioceptiveEncoder with input_dim={self.input_dim}, "
            f"output_dim={self.output_dim}"
        )

    def forward(
        self,
        proprioceptive_state: Optional[ProprioceptiveState] = None,
        proprioceptive_tensor: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through proprioceptive encoder.

        Args:
            proprioceptive_state: Proprioceptive state object
            proprioceptive_tensor: Pre-computed proprioceptive tensor of shape (batch_size, seq_len, 14)
            hidden_state: Optional initial hidden state for SSM

        Returns:
            X_proprio: Output proprioceptive flow of shape (batch_size, seq_len, 256)
            hidden_state: Final hidden state
            metadata: Dictionary containing encoding metadata
        """
        # Convert proprioceptive state to tensor if needed
        if proprioceptive_tensor is None:
            if proprioceptive_state is None:
                # Create default proprioceptive state
                proprioceptive_state = ProprioceptiveState()

            proprioceptive_tensor = proprioceptive_state.to_tensor()

            # Add batch and sequence dimensions if missing
            if proprioceptive_tensor.dim() == 1:
                proprioceptive_tensor = proprioceptive_tensor.unsqueeze(0).unsqueeze(0)
            elif proprioceptive_tensor.dim() == 2:
                proprioceptive_tensor = proprioceptive_tensor.unsqueeze(0)

        # Ensure correct shape: (batch_size, seq_len, 14)
        assert proprioceptive_tensor.shape[-1] == self.input_dim

        batch_size, seq_len, _ = proprioceptive_tensor.shape

        # Numerical stability check
        is_stable, msg = check_numerical_stability(proprioceptive_tensor, "proprioceptive_input")
        if not is_stable:
            logger.warning(msg)

        # Input embedding
        embedded = self.input_embedding(proprioceptive_tensor)

        # SSM processing
        ssm_out, hidden_state = self.ssm(embedded, hidden_state)

        # Output projection
        X_proprio = self.output_proj(ssm_out)

        # Numerical stability check on output
        is_stable, msg = check_numerical_stability(X_proprio, "X_proprio")
        if not is_stable:
            logger.warning(msg)

        # Prepare metadata
        metadata = {
            "input_norm": torch.norm(proprioceptive_tensor).item(),
            "output_norm": torch.norm(X_proprio).item(),
            "is_stable": is_stable,
        }

        return X_proprio, hidden_state, metadata

    def encode_single(
        self,
        proprioceptive_state: ProprioceptiveState,
    ) -> torch.Tensor:
        """
        Encode a single proprioceptive state (no sequence).

        Args:
            proprioceptive_state: Proprioceptive state object

        Returns:
            X_proprio: Output tensor of shape (256,)
        """
        # Treat single state as sequence of length 1
        proprioceptive_tensor = proprioceptive_state.to_tensor().unsqueeze(0).unsqueeze(0)
        X_proprio, _, _ = self.forward(proprioceptive_tensor=proprioceptive_tensor)
        return X_proprio.squeeze(0).squeeze(0)

    def update_from_self_state(
        self,
        self_state: Dict[str, Any],
    ) -> ProprioceptiveState:
        """
        Extract proprioceptive information from SelfState dictionary.

        Args:
            self_state: SelfState as dictionary (from SelfState.to_dict())

        Returns:
            ProprioceptiveState object
        """
        # Extract posture information if available
        position = None
        orientation = None
        velocity = None

        if "metadata" in self_state:
            metadata = self_state["metadata"]
            if "position" in metadata:
                position = torch.tensor(metadata["position"], dtype=torch.float32)
            if "orientation" in metadata:
                orientation = torch.tensor(metadata["orientation"], dtype=torch.float32)
            if "velocity" in metadata:
                velocity = torch.tensor(metadata["velocity"], dtype=torch.float32)

        # Extract energy and resource information
        energy_level = None
        battery_status = None
        cpu_usage = None
        memory_usage = None
        gpu_usage = None

        if "metadata" in self_state:
            metadata = self_state["metadata"]
            energy_level = metadata.get("energy_level", 1.0)
            battery_status = metadata.get("battery_status", 1.0)
            cpu_usage = metadata.get("cpu_usage", 0.0)
            memory_usage = metadata.get("memory_usage", 0.0)
            gpu_usage = metadata.get("gpu_usage", 0.0)

        return ProprioceptiveState(
            position=position,
            orientation=orientation,
            velocity=velocity,
            energy_level=energy_level,
            battery_status=battery_status,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
        )