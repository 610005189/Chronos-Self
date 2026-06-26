"""
World Flow Encoder
==================

This module encodes external environment state:
- Object positions and attributes
- Environmental parameters (temperature, light, sound, etc.)
- Other agents' states
- Spatial relationships

Uses SSM to capture environment dynamics.

Output dimension: 256
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
import logging

from .ssm import StackedSSM, check_numerical_stability
from ..utils.config import EncoderConfig

logger = logging.getLogger(__name__)


class WorldState:
    """
    Container for external world state information.
    """

    def __init__(
        self,
        objects: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        light_level: Optional[float] = None,
        sound_level: Optional[float] = None,
        humidity: Optional[float] = None,
        other_agents: Optional[List[Dict[str, Any]]] = None,
        spatial_grid: Optional[torch.Tensor] = None,
    ):
        """
        Initialize world state.

        Args:
            objects: List of objects with positions and attributes
                    Each object dict should have 'position', 'type', 'attributes'
            temperature: Temperature value (normalized 0-1 or actual value)
            light_level: Light intensity (0.0-1.0)
            sound_level: Sound intensity (0.0-1.0)
            humidity: Humidity level (0.0-1.0)
            other_agents: List of other agents' states
                         Each agent dict should have 'position', 'state'
            spatial_grid: Optional spatial occupancy grid tensor
        """
        self.objects = objects if objects is not None else []
        self.temperature = temperature if temperature is not None else 0.5
        self.light_level = light_level if light_level is not None else 0.5
        self.sound_level = sound_level if sound_level is not None else 0.3
        self.humidity = humidity if humidity is not None else 0.5
        self.other_agents = other_agents if other_agents is not None else []
        self.spatial_grid = spatial_grid

    def to_tensor(self, max_objects: int = 10, max_agents: int = 5) -> torch.Tensor:
        """
        Convert world state to a tensor with fixed size.

        Args:
            max_objects: Maximum number of objects to encode
            max_agents: Maximum number of other agents to encode

        Returns:
            Tensor encoding the world state
        """
        # Encode objects (each object: 3 position + 1 type + 4 attributes = 8 dims)
        object_features = []
        for i, obj in enumerate(self.objects[:max_objects]):
            # Position (3D)
            pos = obj.get('position', [0.0, 0.0, 0.0])
            position_feat = torch.tensor(pos, dtype=torch.float32)

            # Type encoding (one-hot or integer)
            obj_type = obj.get('type', 0)
            type_feat = torch.tensor([obj_type], dtype=torch.float32)

            # Attributes (4 dims)
            attributes = obj.get('attributes', {})
            attr_feat = torch.tensor([
                attributes.get('size', 0.5),
                attributes.get('mass', 0.5),
                attributes.get('color', 0.5),
                attributes.get('material', 0.5),
            ], dtype=torch.float32)

            # Concatenate object features
            object_feat = torch.cat([position_feat, type_feat, attr_feat])
            object_features.append(object_feat)

        # Pad object features to max_objects
        while len(object_features) < max_objects:
            object_features.append(torch.zeros(8, dtype=torch.float32))

        objects_tensor = torch.stack(object_features)  # (max_objects, 8)

        # Encode other agents (each agent: 3 position + 5 state = 8 dims)
        agent_features = []
        for i, agent in enumerate(self.other_agents[:max_agents]):
            # Position (3D)
            pos = agent.get('position', [0.0, 0.0, 0.0])
            position_feat = torch.tensor(pos, dtype=torch.float32)

            # Agent state (5 dims)
            state = agent.get('state', {})
            state_feat = torch.tensor([
                state.get('energy', 1.0),
                state.get('activity', 0.5),
                state.get('orientation', 0.5),
                state.get('speed', 0.5),
                state.get('intent', 0.5),
            ], dtype=torch.float32)

            # Concatenate agent features
            agent_feat = torch.cat([position_feat, state_feat])
            agent_features.append(agent_feat)

        # Pad agent features to max_agents
        while len(agent_features) < max_agents:
            agent_features.append(torch.zeros(8, dtype=torch.float32))

        agents_tensor = torch.stack(agent_features)  # (max_agents, 8)

        # Environmental parameters (4 dims)
        env_params = torch.tensor([
            self.temperature,
            self.light_level,
            self.sound_level,
            self.humidity,
        ], dtype=torch.float32)

        # Flatten objects and agents
        objects_flat = objects_tensor.flatten()  # (max_objects * 8)
        agents_flat = agents_tensor.flatten()  # (max_agents * 8)

        # Concatenate all features
        world_tensor = torch.cat([objects_flat, agents_flat, env_params])

        # Total dimensions: max_objects * 8 + max_agents * 8 + 4
        # For max_objects=10, max_agents=5: 80 + 40 + 4 = 124
        return world_tensor


class WorldEncoder(nn.Module):
    """
    Encoder for world flow (external environment state).

    This encoder processes external environment information using SSM
    to capture temporal dynamics of objects, environment parameters,
    and other agents.

    Output: X_world (dimension 256)
    """

    def __init__(self, config: EncoderConfig):
        """
        Initialize world encoder.

        Args:
            config: Encoder configuration
        """
        super().__init__()

        self.config = config
        # Input dimension depends on max_objects and max_agents
        self.max_objects = 10
        self.max_agents = 5
        self.input_dim = self.max_objects * 8 + self.max_agents * 8 + 4  # 124
        self.output_dim = 256  # Fixed output dimension for world flow

        # Input embedding to project world state to higher dimension
        self.input_embedding = nn.Sequential(
            nn.Linear(self.input_dim, config.physical_hidden_dim),
            nn.LayerNorm(config.physical_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.physical_hidden_dim, config.physical_hidden_dim),
        )

        # Stacked SSM for temporal modeling of environment dynamics
        self.ssm = StackedSSM(
            input_dim=config.physical_hidden_dim,
            state_dim=config.physical_state_dim,
            num_layers=config.physical_num_layers // 2,  # Half for world
            expansion_factor=2,
            dropout=0.1,
        )

        # Output projection to 256 dimensions
        self.output_proj = nn.Sequential(
            nn.Linear(config.physical_hidden_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )

        logger.info(
            f"Initialized WorldEncoder with input_dim={self.input_dim}, "
            f"output_dim={self.output_dim}"
        )

    def forward(
        self,
        world_state: Optional[WorldState] = None,
        world_tensor: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through world encoder.

        Args:
            world_state: World state object
            world_tensor: Pre-computed world tensor of shape (batch_size, seq_len, input_dim)
            hidden_state: Optional initial hidden state for SSM

        Returns:
            X_world: Output world flow of shape (batch_size, seq_len, 256)
            hidden_state: Final hidden state
            metadata: Dictionary containing encoding metadata
        """
        # Convert world state to tensor if needed
        if world_tensor is None:
            if world_state is None:
                # Create default world state
                world_state = WorldState()

            world_tensor = world_state.to_tensor(
                max_objects=self.max_objects,
                max_agents=self.max_agents
            )

            # Add batch and sequence dimensions if missing
            if world_tensor.dim() == 1:
                world_tensor = world_tensor.unsqueeze(0).unsqueeze(0)
            elif world_tensor.dim() == 2:
                world_tensor = world_tensor.unsqueeze(0)

        # Ensure correct shape
        assert world_tensor.shape[-1] == self.input_dim

        batch_size, seq_len, _ = world_tensor.shape

        # Numerical stability check
        is_stable, msg = check_numerical_stability(world_tensor, "world_input")
        if not is_stable:
            logger.warning(msg)

        # Input embedding
        embedded = self.input_embedding(world_tensor)

        # SSM processing to capture environment dynamics
        ssm_out, hidden_state = self.ssm(embedded, hidden_state)

        # Output projection
        X_world = self.output_proj(ssm_out)

        # Numerical stability check on output
        is_stable, msg = check_numerical_stability(X_world, "X_world")
        if not is_stable:
            logger.warning(msg)

        # Prepare metadata
        metadata = {
            "input_norm": torch.norm(world_tensor).item(),
            "output_norm": torch.norm(X_world).item(),
            "is_stable": is_stable,
            "num_objects": len(world_state.objects) if world_state else 0,
            "num_agents": len(world_state.other_agents) if world_state else 0,
        }

        return X_world, hidden_state, metadata

    def encode_single(
        self,
        world_state: WorldState,
    ) -> torch.Tensor:
        """
        Encode a single world state (no sequence).

        Args:
            world_state: World state object

        Returns:
            X_world: Output tensor of shape (256,)
        """
        # Treat single state as sequence of length 1
        world_tensor = world_state.to_tensor(
            max_objects=self.max_objects,
            max_agents=self.max_agents
        ).unsqueeze(0).unsqueeze(0)
        X_world, _, _ = self.forward(world_tensor=world_tensor)
        return X_world.squeeze(0).squeeze(0)

    def update_from_external_input(
        self,
        external_input_dict: Dict[str, Any],
    ) -> WorldState:
        """
        Extract world information from ExternalInput dictionary.

        Args:
            external_input_dict: ExternalInput as dictionary (from ExternalInput.to_dict())

        Returns:
            WorldState object
        """
        # Extract objects and environmental parameters from metadata
        objects = []
        temperature = 0.5
        light_level = 0.5
        sound_level = 0.3
        humidity = 0.5
        other_agents = []

        if "metadata" in external_input_dict:
            metadata = external_input_dict["metadata"]

            # Extract objects
            if "objects" in metadata:
                objects = metadata["objects"]

            # Extract environmental parameters
            temperature = metadata.get("temperature", 0.5)
            light_level = metadata.get("light_level", 0.5)
            sound_level = metadata.get("sound_level", 0.3)
            humidity = metadata.get("humidity", 0.5)

            # Extract other agents
            if "other_agents" in metadata:
                other_agents = metadata["other_agents"]

        return WorldState(
            objects=objects,
            temperature=temperature,
            light_level=light_level,
            sound_level=sound_level,
            humidity=humidity,
            other_agents=other_agents,
        )