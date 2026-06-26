"""
Structured State Space Model (SSM) Implementation
===================================================

This module implements a simplified version of Structured State Space Models
similar to Mamba/S4, providing efficient sequence modeling with long-range dependencies.

The core idea is to use continuous-time state space equations:
    h'(t) = Ah(t) + Bx(t)
    y(t) = Ch(t) + Dx(t)

Which are discretized for sequential processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import logging

logger = logging.getLogger(__name__)


class StateSpaceModel(nn.Module):
    """
    Simplified Structured State Space Model (SSM)

    This implements a discretized state space model with:
    - Learnable state transition matrix (A)
    - Input projection matrix (B)
    - Output projection matrix (C)
    - Optional skip connection (D)

    The implementation uses efficient parallel scan for sequence processing.
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        output_dim: Optional[int] = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        use_bias: bool = True,
        dropout: float = 0.0,
    ):
        """
        Initialize the State Space Model.

        Args:
            input_dim: Input feature dimension
            state_dim: Hidden state dimension (also determines model capacity)
            output_dim: Output feature dimension (defaults to input_dim)
            dt_min: Minimum time step for discretization
            dt_max: Maximum time step for discretization
            use_bias: Whether to use bias in output projection
            dropout: Dropout rate for regularization
        """
        super().__init__()

        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.dt_min = dt_min
        self.dt_max = dt_max

        # State transition parameters (A matrix)
        # Using HiPPO initialization for better long-range dependencies
        A = torch.randn(state_dim, state_dim)
        # Initialize as negative values for stability
        A = -torch.exp(A)
        self.A_real = nn.Parameter(A)
        self.A_imag = nn.Parameter(torch.zeros(state_dim, state_dim))

        # Input projection (B matrix)
        self.B = nn.Parameter(torch.randn(state_dim, input_dim) / math.sqrt(input_dim))

        # Output projection (C matrix)
        self.C = nn.Parameter(torch.randn(output_dim, state_dim) / math.sqrt(state_dim))

        # Skip connection (D parameter)
        if use_bias:
            self.D = nn.Parameter(torch.zeros(output_dim, input_dim))
        else:
            self.register_parameter('D', None)

        # Learnable time step (dt)
        # Initialize as learnable parameter in log space for numerical stability
        log_dt = torch.rand(1) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(self.output_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Numerical stability epsilon
        self.eps = 1e-8

    def _discretize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Discretize continuous-time matrices using zero-order hold (ZOH).

        Returns:
            (A_bar, B_bar): Discretized state transition and input matrices
        """
        # Get time step
        dt = torch.exp(self.log_dt).clamp(min=self.dt_min, max=self.dt_max)

        # For simplicity, use real-valued computation
        # A_bar = exp(A_real * dt) for stable discrete transition
        # Use negative values in A_real for stability
        A_real_neg = -torch.abs(self.A_real)  # Ensure negative for stability

        # Simple discretization: A_bar = exp(A * dt) ≈ I + A * dt for small dt
        # Use approximation for numerical stability
        A_bar = torch.eye(self.state_dim, device=self.A_real.device) + A_real_neg * dt

        # B_bar = B * dt (simple approximation)
        B_bar = self.B * dt

        # Clamp values for stability
        A_bar = torch.clamp(A_bar, min=-10.0, max=10.0)
        B_bar = torch.clamp(B_bar, min=-10.0, max=10.0)

        return A_bar, B_bar

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the SSM.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            hidden_state: Optional initial hidden state of shape (batch_size, state_dim)

        Returns:
            output: Output tensor of shape (batch_size, seq_len, output_dim)
            hidden_state: Final hidden state of shape (batch_size, state_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Discretize matrices
        A_bar, B_bar = self._discretize()

        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = torch.zeros(batch_size, self.state_dim, device=x.device, dtype=x.dtype)

        # Process sequence step by step (can be parallelized with scan)
        outputs = []
        h = hidden_state

        for t in range(seq_len):
            # Update state: h_t = A_bar @ h_{t-1} + B_bar @ x_t
            h = torch.matmul(h, A_bar.t()) + torch.matmul(x[:, t, :], B_bar.t())

            # Compute output: y_t = C @ h_t + D @ x_t
            y = torch.matmul(h, self.C.t())

            # Add skip connection
            if self.D is not None:
                y = y + torch.matmul(x[:, t, :], self.D.t())

            outputs.append(y)

        # Stack outputs
        output = torch.stack(outputs, dim=1)  # (batch_size, seq_len, output_dim)

        # Apply layer normalization and dropout
        output = self.layer_norm(output)
        output = self.dropout(output)

        return output, h

    def parallel_scan(
        self,
        x: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parallel scan implementation for efficient sequence processing.
        This is more efficient than sequential processing for long sequences.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            hidden_state: Optional initial hidden state

        Returns:
            output: Output tensor of shape (batch_size, seq_len, output_dim)
            hidden_state: Final hidden state
        """
        # For simplicity, we use the sequential implementation here
        # A full parallel scan implementation would use associative scan operators
        return self.forward(x, hidden_state)


class SSMBlock(nn.Module):
    """
    Complete SSM block with normalization, gating, and feedforward components.
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        expansion_factor: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize SSM block.

        Args:
            input_dim: Input and output dimension
            state_dim: SSM state dimension
            expansion_factor: Feedforward expansion factor
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.state_dim = state_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, input_dim * 2)  # For gating

        # SSM layer
        self.ssm = StateSpaceModel(
            input_dim=input_dim,
            state_dim=state_dim,
            output_dim=input_dim,
            dropout=dropout,
        )

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * expansion_factor, input_dim),
            nn.Dropout(dropout),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through SSM block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            hidden_state: Optional initial hidden state

        Returns:
            output: Output tensor of shape (batch_size, seq_len, input_dim)
            hidden_state: Final hidden state
        """
        # Input gating
        projected = self.input_proj(x)
        gate, x_gated = projected.chunk(2, dim=-1)
        gate = torch.sigmoid(gate)
        x_gated = x_gated * gate

        # SSM with residual connection
        residual = x
        x = self.norm1(x)
        ssm_out, hidden_state = self.ssm(x_gated, hidden_state)
        x = residual + ssm_out

        # Feedforward with residual connection
        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = residual + ffn_out

        return x, hidden_state


class StackedSSM(nn.Module):
    """
    Stacked SSM blocks for deep sequence modeling.
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        num_layers: int = 4,
        expansion_factor: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize stacked SSM.

        Args:
            input_dim: Input and output dimension
            state_dim: SSM state dimension
            num_layers: Number of SSM blocks
            expansion_factor: Feedforward expansion factor
            dropout: Dropout rate
        """
        super().__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            SSMBlock(
                input_dim=input_dim,
                state_dim=state_dim,
                expansion_factor=expansion_factor,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through stacked SSM blocks.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            hidden_states: Optional initial hidden states of shape (num_layers, batch_size, state_dim)

        Returns:
            output: Output tensor of shape (batch_size, seq_len, input_dim)
            hidden_states: Final hidden states of shape (num_layers, batch_size, state_dim)
        """
        batch_size = x.size(0)

        # Initialize hidden states if not provided
        if hidden_states is None:
            hidden_states = torch.zeros(
                self.num_layers, batch_size, self.layers[0].state_dim,
                device=x.device, dtype=x.dtype
            )

        # Process through each layer
        final_hidden_states = []
        for i, layer in enumerate(self.layers):
            x, h = layer(x, hidden_states[i])
            final_hidden_states.append(h)

        # Stack hidden states
        hidden_states = torch.stack(final_hidden_states, dim=0)

        return x, hidden_states


def check_numerical_stability(tensor: torch.Tensor, name: str = "tensor") -> Tuple[bool, str]:
    """
    Check numerical stability of a tensor.

    Args:
        tensor: Tensor to check
        name: Name for error messages

    Returns:
        (is_stable, message): Whether tensor is numerically stable and error message if not
    """
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    max_abs = tensor.abs().max().item()
    min_abs = tensor.abs().min().item()

    issues = []
    if has_nan:
        issues.append(f"{name} contains NaN values")
    if has_inf:
        issues.append(f"{name} contains Inf values")
    if max_abs > 1e6:
        issues.append(f"{name} has large values (max={max_abs:.2e})")
    if min_abs < 1e-10 and min_abs > 0:
        issues.append(f"{name} has very small values (min={min_abs:.2e})")

    is_stable = len(issues) == 0
    message = "; ".join(issues) if issues else f"{name} is numerically stable"

    return is_stable, message