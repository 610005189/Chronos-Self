"""
Cross-Attention Fusion Mechanism for Chronos-Self
==================================================

This module implements the bidirectional cross-attention fusion mechanism that
enables information exchange between semantic intent and physical state channels.

Components:
- SemanticToPhysicalCrossAttention: Semantic intent queries physical state
- PhysicalToSemanticCrossAttention: Physical state filters semantic intent
- FusionModule: Combines enriched representations from both channels

Purpose:
The fusion mechanism enables:
1. Semantic intentions to query physically relevant aspects (what physical states
   are relevant to my intent?)
2. Physical constraints to filter semantically executable intents (what intents
   are physically feasible?)
3. Unified representation combining both enriched channels for downstream tasks

Technical Details:
- Multi-head cross-attention (configurable heads: 4-8)
- Scaled dot-product attention for numerical stability
- Optional attention masks for constraint enforcement
- Layer normalization for stable training
- Dropout for regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from loguru import logger

from chronos_core.utils.config import EncoderConfig


@dataclass
class FusionOutput:
    """
    Output container for fusion module results.

    Attributes:
        X_fused: Combined fusion representation (batch_size, seq_len, fusion_dim)
        X_sem_enriched: Enriched semantic vector (batch_size, seq_len, sem_dim)
        X_log_enriched: Enriched physical vector (batch_size, seq_len, log_dim)
        sem_to_phys_attention: Attention weights from semantic to physical
        phys_to_sem_attention: Attention weights from physical to semantic
        metadata: Additional fusion statistics and diagnostics
    """
    X_fused: torch.Tensor
    X_sem_enriched: torch.Tensor
    X_log_enriched: torch.Tensor
    sem_to_phys_attention: Optional[torch.Tensor] = None
    phys_to_sem_attention: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'X_fused': self.X_fused.cpu().numpy() if self.X_fused is not None else None,
            'X_sem_enriched': self.X_sem_enriched.cpu().numpy() if self.X_sem_enriched is not None else None,
            'X_log_enriched': self.X_log_enriched.cpu().numpy() if self.X_log_enriched is not None else None,
            'sem_to_phys_attention': self.sem_to_phys_attention.cpu().numpy() if self.sem_to_phys_attention is not None else None,
            'phys_to_sem_attention': self.phys_to_sem_attention.cpu().numpy() if self.phys_to_sem_attention is not None else None,
            'metadata': self.metadata
        }


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention mechanism with numerical stability.

    Computes attention weights as:
        Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V

    Features:
    - Scaling factor to prevent gradient explosion
    - Optional masking for constrained attention
    - Numerical stability checks (NaN/Inf detection)
    - Supports batch processing
    """

    def __init__(
        self,
        dropout: float = 0.1,
        scale_by_dim: bool = True
    ):
        """
        Initialize scaled dot-product attention.

        Args:
            dropout: Dropout probability for attention weights
            scale_by_dim: Whether to scale by dimension (sqrt(d_k))
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale_by_dim = scale_by_dim

        logger.debug(f"Initialized ScaledDotProductAttention: dropout={dropout}, scale_by_dim={scale_by_dim}")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute scaled dot-product attention.

        Args:
            query: Query tensor (batch_size, num_queries, query_dim)
            key: Key tensor (batch_size, num_keys, key_dim)
            value: Value tensor (batch_size, num_values, value_dim)
            attention_mask: Optional mask (batch_size, num_queries, num_keys)
                           True/1 = masked position (not attended to)
            need_weights: Whether to return attention weights

        Returns:
            output: Attended output (batch_size, num_queries, value_dim)
            attention_weights: Optional attention weights (batch_size, num_queries, num_keys)
        """
        # Check numerical stability before computation
        if torch.isnan(query).any() or torch.isinf(query).any():
            logger.warning("Query contains NaN/Inf, applying correction")
            query = torch.nan_to_num(query, nan=0.0, posinf=1.0, neginf=-1.0)

        if torch.isnan(key).any() or torch.isinf(key).any():
            logger.warning("Key contains NaN/Inf, applying correction")
            key = torch.nan_to_num(key, nan=0.0, posinf=1.0, neginf=-1.0)

        if torch.isnan(value).any() or torch.isinf(value).any():
            logger.warning("Value contains NaN/Inf, applying correction")
            value = torch.nan_to_num(value, nan=0.0, posinf=1.0, neginf=-1.0)

        # Compute attention scores
        # query: (batch_size, num_queries, query_dim)
        # key: (batch_size, num_keys, key_dim)
        # Note: query_dim and key_dim must match for dot-product

        # Transpose key for dot-product: (batch_size, key_dim, num_keys)
        key_transposed = key.transpose(-2, -1)

        # Compute raw scores: (batch_size, num_queries, num_keys)
        attention_scores = torch.matmul(query, key_transposed)

        # Scale by dimension to prevent large values
        if self.scale_by_dim:
            scale_factor = query.size(-1) ** 0.5
            attention_scores = attention_scores / scale_factor

        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert mask to proper format (masked positions get -inf)
            # Assuming mask shape: (batch_size, num_queries, num_keys)
            if attention_mask.dtype == torch.bool:
                attention_scores = attention_scores.masked_fill(
                    attention_mask,
                    float('-inf')
                )
            else:
                # Numeric mask: assume 1 = masked
                attention_scores = attention_scores.masked_fill(
                    attention_mask > 0,
                    float('-inf')
                )

        # Compute attention weights with numerical stability
        # Check for extreme values that could cause softmax overflow
        max_score = attention_scores.max()
        if max_score > 100.0:
            logger.warning(f"Large attention scores detected (max={max_score}), applying clipping")
            attention_scores = torch.clamp(attention_scores, min=-100.0, max=100.0)

        attention_weights = F.softmax(attention_scores, dim=-1)

        # Check for numerical stability in softmax output
        if torch.isnan(attention_weights).any():
            logger.warning("NaN in attention weights after softmax, replacing with uniform distribution")
            num_keys = attention_weights.size(-1)
            attention_weights = torch.ones_like(attention_weights) / num_keys

        # Apply dropout to attention weights
        attention_weights_dropped = self.dropout(attention_weights)

        # Compute output: (batch_size, num_queries, value_dim)
        output = torch.matmul(attention_weights_dropped, value)

        # Return attention weights if requested (before dropout)
        if need_weights:
            return output, attention_weights
        else:
            return output, None


class SemanticToPhysicalCrossAttention(nn.Module):
    """
    Semantic to Physical Cross-Attention Module.

    Purpose: Enable semantic intentions to query relevant physical states.

    Mechanism:
    - Query: Semantic intent vector X_sem (dimension 256)
    - Key/Value: Logical physical vector X_log (dimension 512)
    - Output: Enriched semantic vector X_sem_enriched (dimension 256)

    The semantic intention "asks" the physical state: "What physical states
    are relevant to my intent?" This enables context-aware semantic processing
    that is grounded in physical reality.

    Example:
    - Semantic intent: "I want to reach the table"
    - Physical state: "Table is 2 meters away, left side"
    - Cross-attention helps semantic encoding focus on relevant distance/location

    Implementation:
    - Multi-head attention for diverse attention patterns
    - Projection layers for dimension compatibility
    - Layer normalization for stable training
    """

    def __init__(
        self,
        sem_dim: int = 256,
        log_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        output_dim: Optional[int] = None
    ):
        """
        Initialize Semantic to Physical Cross-Attention.

        Args:
            sem_dim: Semantic intent vector dimension (query dimension)
            log_dim: Logical physical vector dimension (key/value dimension)
            num_heads: Number of attention heads (4-8 recommended)
            dropout: Dropout probability
            output_dim: Output dimension (default: same as sem_dim)
        """
        super().__init__()

        self.sem_dim = sem_dim
        self.log_dim = log_dim
        self.num_heads = num_heads
        self.output_dim = output_dim if output_dim is not None else sem_dim

        # Ensure dimensions are divisible by num_heads for multi-head attention
        assert sem_dim % num_heads == 0, f"sem_dim ({sem_dim}) must be divisible by num_heads ({num_heads})"
        self.head_dim = sem_dim // num_heads

        # Query projection (semantic vector to query)
        self.query_projection = nn.Linear(sem_dim, sem_dim)

        # Key and Value projections (physical vector to key/value)
        # Need to project log_dim to sem_dim for compatibility
        self.key_projection = nn.Linear(log_dim, sem_dim)
        self.value_projection = nn.Linear(log_dim, sem_dim)

        # Output projection
        self.output_projection = nn.Linear(sem_dim, self.output_dim)

        # Layer normalization for stability
        self.layer_norm_query = nn.LayerNorm(sem_dim)
        self.layer_norm_output = nn.LayerNorm(self.output_dim)

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout=dropout, scale_by_dim=True)

        logger.info(
            f"Initialized SemanticToPhysicalCrossAttention: "
            f"sem_dim={sem_dim}, log_dim={log_dim}, num_heads={num_heads}, "
            f"head_dim={self.head_dim}, output_dim={self.output_dim}"
        )

    def forward(
        self,
        X_sem: torch.Tensor,
        X_log: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through semantic to physical cross-attention.

        Args:
            X_sem: Semantic intent vector (batch_size, seq_len, sem_dim)
            X_log: Logical physical vector (batch_size, seq_len, log_dim)
            attention_mask: Optional mask (batch_size, seq_len, seq_len)
            need_weights: Whether to return attention weights

        Returns:
            X_sem_enriched: Enriched semantic vector (batch_size, seq_len, output_dim)
            attention_weights: Optional attention weights (batch_size, num_heads, seq_len, seq_len)
        """
        # Validate inputs
        if torch.isnan(X_sem).any() or torch.isinf(X_sem).any():
            logger.warning("X_sem contains NaN/Inf, applying correction")
            X_sem = torch.nan_to_num(X_sem, nan=0.0, posinf=1.0, neginf=-1.0)

        if torch.isnan(X_log).any() or torch.isinf(X_log).any():
            logger.warning("X_log contains NaN/Inf, applying correction")
            X_log = torch.nan_to_num(X_log, nan=0.0, posinf=1.0, neginf=-1.0)

        batch_size, seq_len = X_sem.shape[:2]

        # Layer normalization on input query
        X_sem_normed = self.layer_norm_query(X_sem)

        # Project to query, key, value
        query = self.query_projection(X_sem_normed)  # (batch_size, seq_len, sem_dim)
        key = self.key_projection(X_log)  # (batch_size, seq_len, sem_dim)
        value = self.value_projection(X_log)  # (batch_size, seq_len, sem_dim)

        # Reshape for multi-head attention
        # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute multi-head attention
        # For each head, compute scaled dot-product attention
        # Combine all heads' results

        # Compute attention for each head
        # We'll process heads in parallel using batch dimension
        attended_outputs = []
        attention_weights_list = []

        for head_idx in range(self.num_heads):
            # Extract head-specific query, key, value
            head_query = query[:, head_idx, :, :]  # (batch_size, seq_len, head_dim)
            head_key = key[:, head_idx, :, :]  # (batch_size, seq_len, head_dim)
            head_value = value[:, head_idx, :, :]  # (batch_size, seq_len, head_dim)

            # Compute attention for this head
            head_output, head_attention = self.attention(
                head_query, head_key, head_value,
                attention_mask=attention_mask,
                need_weights=need_weights
            )

            attended_outputs.append(head_output)
            if head_attention is not None:
                attention_weights_list.append(head_attention)

        # Concatenate all heads' outputs
        # (batch_size, seq_len, head_dim) * num_heads -> (batch_size, seq_len, sem_dim)
        concatenated_output = torch.cat(attended_outputs, dim=-1)

        # Project to output dimension
        output = self.output_projection(concatenated_output)

        # Apply dropout and residual connection
        output = self.dropout(output)
        X_sem_enriched = self.layer_norm_output(X_sem + output)

        # Combine attention weights from all heads
        if need_weights and len(attention_weights_list) > 0:
            # Stack attention weights: (batch_size, num_heads, seq_len, seq_len)
            attention_weights = torch.stack(attention_weights_list, dim=1)
        else:
            attention_weights = None

        # Final numerical stability check
        if torch.isnan(X_sem_enriched).any() or torch.isinf(X_sem_enriched).any():
            logger.warning("X_sem_enriched contains NaN/Inf after processing")
            X_sem_enriched = torch.nan_to_num(X_sem_enriched, nan=0.0, posinf=1.0, neginf=-1.0)

        return X_sem_enriched, attention_weights


class PhysicalToSemanticCrossAttention(nn.Module):
    """
    Physical to Semantic Cross-Attention Module.

    Purpose: Enable physical constraints to filter semantically executable intents.

    Mechanism:
    - Query: Logical physical vector X_log (dimension 512)
    - Key/Value: Semantic intent vector X_sem (dimension 256)
    - Output: Enriched physical vector X_log_enriched (dimension 512)

    The physical state "asks" the semantic intent: "Which semantic intents are
    physically executable given my current state?" This enables physical-grounded
    filtering of semantic intentions based on feasibility.

    Example:
    - Physical state: "Legs are tired, energy level low"
    - Semantic intent: "I want to run 5 kilometers"
    - Cross-attention helps physical encoding focus on feasible alternatives

    Implementation:
    - Multi-head attention for diverse attention patterns
    - Projection layers for dimension compatibility
    - Layer normalization for stable training
    """

    def __init__(
        self,
        log_dim: int = 512,
        sem_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        output_dim: Optional[int] = None
    ):
        """
        Initialize Physical to Semantic Cross-Attention.

        Args:
            log_dim: Logical physical vector dimension (query dimension)
            sem_dim: Semantic intent vector dimension (key/value dimension)
            num_heads: Number of attention heads (4-8 recommended)
            dropout: Dropout probability
            output_dim: Output dimension (default: same as log_dim)
        """
        super().__init__()

        self.log_dim = log_dim
        self.sem_dim = sem_dim
        self.num_heads = num_heads
        self.output_dim = output_dim if output_dim is not None else log_dim

        # For physical query, we need to ensure log_dim is divisible by num_heads
        # If not, we'll use a smaller effective dimension
        if log_dim % num_heads != 0:
            # Adjust head dimension to make it compatible
            self.effective_query_dim = (log_dim // num_heads) * num_heads
            logger.warning(
                f"log_dim ({log_dim}) not divisible by num_heads ({num_heads}), "
                f"using effective_query_dim={self.effective_query_dim}"
            )
        else:
            self.effective_query_dim = log_dim

        self.head_dim = self.effective_query_dim // num_heads

        # Query projection (physical vector to query)
        self.query_projection = nn.Linear(log_dim, self.effective_query_dim)

        # Key and Value projections (semantic vector to key/value)
        # Need to project sem_dim to effective_query_dim for compatibility
        self.key_projection = nn.Linear(sem_dim, self.effective_query_dim)
        self.value_projection = nn.Linear(sem_dim, self.effective_query_dim)

        # Output projection (back to log_dim)
        self.output_projection = nn.Linear(self.effective_query_dim, self.output_dim)

        # Layer normalization for stability
        self.layer_norm_query = nn.LayerNorm(log_dim)
        self.layer_norm_output = nn.LayerNorm(self.output_dim)

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout=dropout, scale_by_dim=True)

        logger.info(
            f"Initialized PhysicalToSemanticCrossAttention: "
            f"log_dim={log_dim}, sem_dim={sem_dim}, num_heads={num_heads}, "
            f"effective_query_dim={self.effective_query_dim}, head_dim={self.head_dim}, "
            f"output_dim={self.output_dim}"
        )

    def forward(
        self,
        X_log: torch.Tensor,
        X_sem: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through physical to semantic cross-attention.

        Args:
            X_log: Logical physical vector (batch_size, seq_len, log_dim)
            X_sem: Semantic intent vector (batch_size, seq_len, sem_dim)
            attention_mask: Optional mask (batch_size, seq_len, seq_len)
            need_weights: Whether to return attention weights

        Returns:
            X_log_enriched: Enriched physical vector (batch_size, seq_len, output_dim)
            attention_weights: Optional attention weights (batch_size, num_heads, seq_len, seq_len)
        """
        # Validate inputs
        if torch.isnan(X_log).any() or torch.isinf(X_log).any():
            logger.warning("X_log contains NaN/Inf, applying correction")
            X_log = torch.nan_to_num(X_log, nan=0.0, posinf=1.0, neginf=-1.0)

        if torch.isnan(X_sem).any() or torch.isinf(X_sem).any():
            logger.warning("X_sem contains NaN/Inf, applying correction")
            X_sem = torch.nan_to_num(X_sem, nan=0.0, posinf=1.0, neginf=-1.0)

        batch_size, seq_len = X_log.shape[:2]

        # Layer normalization on input query
        X_log_normed = self.layer_norm_query(X_log)

        # Project to query, key, value
        query = self.query_projection(X_log_normed)  # (batch_size, seq_len, effective_query_dim)
        key = self.key_projection(X_sem)  # (batch_size, seq_len, effective_query_dim)
        value = self.value_projection(X_sem)  # (batch_size, seq_len, effective_query_dim)

        # Reshape for multi-head attention
        # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute multi-head attention
        attended_outputs = []
        attention_weights_list = []

        for head_idx in range(self.num_heads):
            # Extract head-specific query, key, value
            head_query = query[:, head_idx, :, :]  # (batch_size, seq_len, head_dim)
            head_key = key[:, head_idx, :, :]  # (batch_size, seq_len, head_dim)
            head_value = value[:, head_idx, :, :]  # (batch_size, seq_len, head_dim)

            # Compute attention for this head
            head_output, head_attention = self.attention(
                head_query, head_key, head_value,
                attention_mask=attention_mask,
                need_weights=need_weights
            )

            attended_outputs.append(head_output)
            if head_attention is not None:
                attention_weights_list.append(head_attention)

        # Concatenate all heads' outputs
        # (batch_size, seq_len, head_dim) * num_heads -> (batch_size, seq_len, effective_query_dim)
        concatenated_output = torch.cat(attended_outputs, dim=-1)

        # Project to output dimension
        output = self.output_projection(concatenated_output)

        # Apply dropout and residual connection
        output = self.dropout(output)
        X_log_enriched = self.layer_norm_output(X_log + output)

        # Combine attention weights from all heads
        if need_weights and len(attention_weights_list) > 0:
            # Stack attention weights: (batch_size, num_heads, seq_len, seq_len)
            attention_weights = torch.stack(attention_weights_list, dim=1)
        else:
            attention_weights = None

        # Final numerical stability check
        if torch.isnan(X_log_enriched).any() or torch.isinf(X_log_enriched).any():
            logger.warning("X_log_enriched contains NaN/Inf after processing")
            X_log_enriched = torch.nan_to_num(X_log_enriched, nan=0.0, posinf=1.0, neginf=-1.0)

        return X_log_enriched, attention_weights


class FusionModule(nn.Module):
    """
    Fusion Module for Combining Enriched Semantic and Physical Representations.

    Purpose: Create a unified representation that integrates both enriched channels
    for downstream tasks (e.g., integration engine, dynamics prediction).

    Mechanism:
    - Inputs: X_sem_enriched (256) + X_log_enriched (512)
    - Outputs: X_fused (768 = 256 + 512)
    - Fusion strategies: Simple concatenation + optional attention-based refinement

    The fusion module provides a unified interface that can be used by the
    integration engine to combine semantic intent and physical state into
    a single actionable representation.

    Implementation:
    - Simple concatenation fusion (default)
    - Optional attention-based fusion for weighted combination
    - Projection layers for dimension adjustment
    - Layer normalization for stability
    """

    def __init__(
        self,
        sem_dim: int = 256,
        log_dim: int = 512,
        fusion_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_attention_fusion: bool = False,
        config: Optional[EncoderConfig] = None
    ):
        """
        Initialize Fusion Module.

        Args:
            sem_dim: Semantic intent dimension (256)
            log_dim: Logical physical dimension (512)
            fusion_dim: Fusion output dimension (768 = 256 + 512)
            num_heads: Number of attention heads for attention fusion
            dropout: Dropout probability
            use_attention_fusion: Whether to use attention-based fusion (default: False)
            config: EncoderConfig instance for parameter initialization
        """
        super().__init__()

        # Use config if provided (for cross_attention parameters only)
        # Note: sem_dim and log_dim should be provided explicitly based on actual encoder outputs
        # SemanticEncoder output_dim = 256 (default)
        # LogicalEncoder output_dim = 512 (X_log)
        if config is not None:
            # Only use cross-attention config, not dimension config
            # Dimensions should match actual encoder outputs
            num_heads = config.cross_attention_heads if hasattr(config, 'cross_attention_heads') else num_heads
            dropout = config.cross_attention_dropout if hasattr(config, 'cross_attention_dropout') else dropout

        self.sem_dim = sem_dim
        self.log_dim = log_dim
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        self.use_attention_fusion = use_attention_fusion

        # Semantic to Physical Cross-Attention
        self.sem_to_phys_attention = SemanticToPhysicalCrossAttention(
            sem_dim=sem_dim,
            log_dim=log_dim,
            num_heads=num_heads,
            dropout=dropout,
            output_dim=sem_dim
        )

        # Physical to Semantic Cross-Attention
        self.phys_to_sem_attention = PhysicalToSemanticCrossAttention(
            log_dim=log_dim,
            sem_dim=sem_dim,
            num_heads=num_heads,
            dropout=dropout,
            output_dim=log_dim
        )

        # Simple concatenation fusion (default)
        # Output dimension: sem_dim + log_dim = 768

        # Optional attention-based fusion
        if use_attention_fusion:
            # Self-attention on concatenated features for refinement
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=fusion_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.fusion_norm = nn.LayerNorm(fusion_dim)

        # Layer normalization for fused output
        self.output_norm = nn.LayerNorm(fusion_dim)

        # Optional projection for different output dimensions
        # (if fusion_dim != sem_dim + log_dim)
        if fusion_dim != sem_dim + log_dim:
            self.output_projection = nn.Linear(sem_dim + log_dim, fusion_dim)
            logger.info(f"Output projection: {sem_dim + log_dim} -> {fusion_dim}")
        else:
            self.output_projection = None

        logger.info(
            f"Initialized FusionModule: sem_dim={sem_dim}, log_dim={log_dim}, "
            f"fusion_dim={fusion_dim}, use_attention_fusion={use_attention_fusion}, "
            f"num_heads={num_heads}"
        )

    def forward(
        self,
        X_sem: torch.Tensor,
        X_log: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_enriched: bool = True,
        need_attention_weights: bool = False
    ) -> Union[torch.Tensor, FusionOutput]:
        """
        Forward pass through fusion module.

        Args:
            X_sem: Semantic intent vector (batch_size, seq_len, sem_dim)
            X_log: Logical physical vector (batch_size, seq_len, log_dim)
            attention_mask: Optional attention mask for cross-attention
            return_enriched: Whether to return enriched representations separately
            need_attention_weights: Whether to return attention weights

        Returns:
            If return_enriched is False:
                X_fused: Fused representation (batch_size, seq_len, fusion_dim)
            If return_enriched is True:
                FusionOutput containing X_fused, X_sem_enriched, X_log_enriched, and metadata
        """
        # Validate inputs
        if torch.isnan(X_sem).any() or torch.isinf(X_sem).any():
            logger.warning("X_sem input contains NaN/Inf, applying correction")
            X_sem = torch.nan_to_num(X_sem, nan=0.0, posinf=1.0, neginf=-1.0)

        if torch.isnan(X_log).any() or torch.isinf(X_log).any():
            logger.warning("X_log input contains NaN/Inf, applying correction")
            X_log = torch.nan_to_num(X_log, nan=0.0, posinf=1.0, neginf=-1.0)

        batch_size, seq_len = X_sem.shape[:2]

        # Apply semantic to physical cross-attention
        X_sem_enriched, sem_to_phys_attn = self.sem_to_phys_attention(
            X_sem, X_log,
            attention_mask=attention_mask,
            need_weights=need_attention_weights
        )

        # Apply physical to semantic cross-attention
        X_log_enriched, phys_to_sem_attn = self.phys_to_sem_attention(
            X_log, X_sem,
            attention_mask=attention_mask,
            need_weights=need_attention_weights
        )

        # Concatenate enriched representations
        # (batch_size, seq_len, sem_dim + log_dim) = (batch_size, seq_len, 768)
        X_fused_raw = torch.cat([X_sem_enriched, X_log_enriched], dim=-1)

        # Optional attention-based fusion refinement
        if self.use_attention_fusion:
            # Apply self-attention on concatenated features
            X_fused_attended, _ = self.fusion_attention(
                X_fused_raw, X_fused_raw, X_fused_raw,
                need_weights=False
            )
            X_fused_raw = self.fusion_norm(X_fused_raw + X_fused_attended)

        # Apply output projection if needed
        if self.output_projection is not None:
            X_fused = self.output_projection(X_fused_raw)
        else:
            X_fused = X_fused_raw

        # Apply final layer normalization
        X_fused = self.output_norm(X_fused)

        # Final numerical stability check
        if torch.isnan(X_fused).any() or torch.isinf(X_fused).any():
            logger.warning("X_fused contains NaN/Inf after processing")
            X_fused = torch.nan_to_num(X_fused, nan=0.0, posinf=1.0, neginf=-1.0)

        if torch.isnan(X_sem_enriched).any() or torch.isinf(X_sem_enriched).any():
            logger.warning("X_sem_enriched contains NaN/Inf after processing")
            X_sem_enriched = torch.nan_to_num(X_sem_enriched, nan=0.0, posinf=1.0, neginf=-1.0)

        if torch.isnan(X_log_enriched).any() or torch.isinf(X_log_enriched).any():
            logger.warning("X_log_enriched contains NaN/Inf after processing")
            X_log_enriched = torch.nan_to_num(X_log_enriched, nan=0.0, posinf=1.0, neginf=-1.0)

        # Return based on request
        if return_enriched:
            # Prepare metadata
            metadata = {
                'X_sem_norm': torch.norm(X_sem).item(),
                'X_log_norm': torch.norm(X_log).item(),
                'X_sem_enriched_norm': torch.norm(X_sem_enriched).item(),
                'X_log_enriched_norm': torch.norm(X_log_enriched).item(),
                'X_fused_norm': torch.norm(X_fused).item(),
                'batch_size': batch_size,
                'seq_len': seq_len,
                'fusion_strategy': 'attention' if self.use_attention_fusion else 'concatenation',
                'dimensions': {
                    'sem_dim': self.sem_dim,
                    'log_dim': self.log_dim,
                    'fusion_dim': self.fusion_dim
                }
            }

            output = FusionOutput(
                X_fused=X_fused,
                X_sem_enriched=X_sem_enriched,
                X_log_enriched=X_log_enriched,
                sem_to_phys_attention=sem_to_phys_attn if need_attention_weights else None,
                phys_to_sem_attention=phys_to_sem_attn if need_attention_weights else None,
                metadata=metadata
            )

            return output
        else:
            return X_fused

    def fuse_batch(
        self,
        semantic_vectors: torch.Tensor,
        physical_vectors: torch.Tensor,
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Batch fusion for large-scale processing.

        Args:
            semantic_vectors: Semantic vectors (total_samples, sem_dim)
            physical_vectors: Physical vectors (total_samples, log_dim)
            batch_size: Processing batch size

        Returns:
            fused_vectors: Fused vectors (total_samples, fusion_dim)
        """
        total_samples = semantic_vectors.shape[0]

        # Reshape to sequence format: (batch, seq_len, dim)
        # Treat each sample as sequence of length 1
        X_sem = semantic_vectors.unsqueeze(1)  # (total_samples, 1, sem_dim)
        X_log = physical_vectors.unsqueeze(1)  # (total_samples, 1, log_dim)

        # Process in batches
        fused_results = []
        for i in range(0, total_samples, batch_size):
            batch_sem = X_sem[i:i + batch_size]
            batch_log = X_log[i:i + batch_size]

            # Fuse batch
            batch_fused = self.forward(
                batch_sem, batch_log,
                return_enriched=False,
                need_attention_weights=False
            )

            fused_results.append(batch_fused)

            logger.debug(f"Fused batch {i // batch_size + 1}/{(total_samples + batch_size - 1) // batch_size}")

        # Concatenate all batches and remove sequence dimension
        fused_vectors = torch.cat(fused_results, dim=0).squeeze(1)  # (total_samples, fusion_dim)

        logger.info(f"Fused {total_samples} vectors in {(total_samples + batch_size - 1) // batch_size} batches")

        return fused_vectors


def create_fusion_module(
    config: Optional[EncoderConfig] = None,
    sem_dim: int = 256,
    log_dim: int = 512,
    fusion_dim: int = 768,
    use_attention_fusion: bool = False
) -> FusionModule:
    """
    Factory function to create fusion module.

    Args:
        config: EncoderConfig instance (uses default if None)
        sem_dim: Semantic dimension (256)
        log_dim: Physical dimension (512)
        fusion_dim: Fusion dimension (768)
        use_attention_fusion: Whether to use attention fusion

    Returns:
        Initialized FusionModule instance
    """
    if config is None:
        from chronos_core.utils.config import EncoderConfig
        config = EncoderConfig()

    return FusionModule(
        sem_dim=sem_dim,
        log_dim=log_dim,
        fusion_dim=fusion_dim,
        config=config,
        use_attention_fusion=use_attention_fusion
    )