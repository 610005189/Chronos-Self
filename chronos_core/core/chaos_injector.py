"""
混沌注入器
==========

实现高维混沌信号投影注入机制，将低维混沌吸引子信号注入到高维核心子空间。

核心功能：
- 固定随机正交投影矩阵 W (k×3, k=64维核心子空间)
- 混沌信号投影注入：B(t) = P_core · Chaos(z(t))
- 仅耦合到核心子空间，避免高维稀释
- 自适应增益控制
"""

import torch
import numpy as np
from typing import Optional, Dict, Tuple
import logging
from dataclasses import dataclass

try:
    from scipy.linalg import ortho_group
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class InjectionConfig:
    """混沌注入配置"""
    core_subspace_dim: int = 64
    chaos_dim: int = 3
    base_gain: float = 0.1
    target_variance: float = 1.0
    gain_smoothing: float = 0.95
    variance_window: int = 100


class ChaosInjector:
    """
    混沌信号注入器

    将三维混沌吸引子信号投影注入到高维核心子空间，
    为系统提供内源性动力学源。

    注入公式：
        B(t) = g_chaos(t) · W · z(t)

    其中：
        - W: 固定随机正交投影矩阵 (k×3)
        - z(t): 混沌状态向量 (3,)
        - g_chaos(t): 自适应增益

    Attributes:
        W: 正交投影矩阵 (core_dim, chaos_dim)
        config: 注入配置
        gain: 当前增益值
    """

    def __init__(
        self,
        core_subspace_dim: int = 64,
        chaos_dim: int = 3,
        base_gain: float = 0.1,
        min_gain: float = 0.1,
        target_variance: float = 1.0,
        device: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        初始化混沌注入器

        Args:
            core_subspace_dim: 核心子空间维度 (k)
            chaos_dim: 混沌吸引子维度 (默认为3)
            base_gain: 基础注入增益 (g0)
            min_gain: 最小注入增益（防止自适应增益过低）
            target_variance: 目标方差 (σ²_target)
            device: 计算设备
            seed: 随机种子（用于固定投影矩阵）
        """
        self.core_dim = core_subspace_dim
        self.chaos_dim = chaos_dim
        self.base_gain = base_gain
        self.min_gain = min_gain
        self.target_variance = target_variance
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 随机种子
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # 生成固定随机正交投影矩阵
        self.W = self._generate_orthogonal_matrix(seed)

        # 增益状态
        self.current_gain = base_gain
        self.gain_smoothing = 0.95

        # 方差追踪（用于自适应增益）
        self.variance_window_size = 100
        self.variance_history: list = []

        # 统计信息
        self.total_injections = 0
        self.injection_history: list = []

        logger.info(
            f"ChaosInjector initialized: core_dim={core_subspace_dim}, "
            f"chaos_dim={chaos_dim}, base_gain={base_gain}, "
            f"W_shape={self.W.shape}"
        )

    def _generate_orthogonal_matrix(self, seed: Optional[int] = None) -> torch.Tensor:
        """
        生成固定随机正交投影矩阵 W (k×3)

        W 的列向量是正交的，确保投影不产生冗余。

        Args:
            seed: 随机种子

        Returns:
            正交投影矩阵 (core_dim, chaos_dim)
        """
        if seed is not None:
            np.random.seed(seed)

        # 使用 scipy 生成正交矩阵（如果可用）
        if SCIPY_AVAILABLE and self.core_dim >= self.chaos_dim:
            # 生成随机正交矩阵组
            random_ortho = ortho_group.rvs(self.core_dim, random_state=seed)
            # 取前 chaos_dim 列作为投影矩阵
            W_np = random_ortho[:, :self.chaos_dim]
        else:
            # 手动生成并正交化
            W_np = np.random.randn(self.core_dim, self.chaos_dim)
            # QR 分解正交化
            W_np, _ = np.linalg.qr(W_np)

        # 转换为 PyTorch 张量
        W = torch.tensor(W_np, dtype=torch.float32, device=self.device)

        # 验证正交性
        self._verify_orthogonality(W)

        logger.debug(f"Generated orthogonal projection matrix W: shape={W.shape}")

        return W

    def _verify_orthogonality(self, W: torch.Tensor) -> bool:
        """
        验证投影矩阵的正交性

        Args:
            W: 投影矩阵

        Returns:
            是否正交
        """
        # W^T W 应接近单位矩阵
        WtW = torch.mm(W.t(), W)
        identity = torch.eye(self.chaos_dim, device=self.device)
        error = torch.norm(WtW - identity).item()

        is_orthogonal = error < 1e-5

        if not is_orthogonal:
            logger.warning(f"Projection matrix not perfectly orthogonal: error={error:.6f}")
        else:
            logger.debug(f"Projection matrix verified orthogonal: error={error:.6f}")

        return is_orthogonal

    def inject(
        self,
        z: torch.Tensor,
        E_fast_core: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        执行混沌信号投影注入

        计算注入信号：B = g_chaos · W · z

        Args:
            z: 混沌状态向量 (chaos_dim,)
            E_fast_core: 快变量核心子空间分量 (core_dim,)
                         用于计算自适应增益

        Returns:
            注入信号向量 (core_dim,)
        """
        # 确保输入在正确设备上
        z = z.to(self.device)

        # NaN/Inf check on input chaos signal
        if torch.isnan(z).any() or torch.isinf(z).any():
            z = torch.nan_to_num(z, nan=0.0, posinf=1e6, neginf=-1e6)
            logger.warning("Chaos signal z contains NaN/Inf, values clamped")

        # 计算自适应增益（如果提供了快变量）
        if E_fast_core is not None:
            # NaN/Inf check on E_fast_core
            if torch.isnan(E_fast_core).any() or torch.isinf(E_fast_core).any():
                E_fast_core = torch.nan_to_num(E_fast_core, nan=0.0, posinf=1e6, neginf=-1e6)
            self._update_adaptive_gain(E_fast_core)

        # 投影注入：B = g · W · z
        # W: (core_dim, chaos_dim)
        # z: (chaos_dim,)
        # B: (core_dim,)
        B = self.current_gain * torch.mv(self.W, z)

        # NaN/Inf check on output
        if torch.isnan(B).any() or torch.isinf(B).any():
            B = torch.nan_to_num(B, nan=0.0, posinf=1e6, neginf=-1e6)
            logger.warning("Injected signal B contains NaN/Inf, values clamped")

        # 记录统计
        self.total_injections += 1
        self._record_injection(z, B)

        logger.debug(
            f"Injected chaos signal: gain={self.current_gain:.4f}, "
            f"z_norm={torch.norm(z).item():.4f}, B_norm={torch.norm(B).item():.4f}"
        )

        return B

    def _update_adaptive_gain(self, E_fast_core: torch.Tensor) -> None:
        """
        更新自适应增益

        增益公式：
            g_chaos(t) = g0 · σ²_target / (σ²_target + Var(P_core^T E_fast(t)))

        当核心子空间方差增大时，减小增益以避免过度扰动。

        Args:
            E_fast_core: 快变量核心子空间分量
        """
        E_fast_core = E_fast_core.to(self.device)

        # 计算当前方差
        current_variance = torch.var(E_fast_core).item()

        # 更新方差历史
        self.variance_history.append(current_variance)
        if len(self.variance_history) > self.variance_window_size:
            self.variance_history = self.variance_history[-self.variance_window_size:]

        # 计算平均方差
        avg_variance = np.mean(self.variance_history) if self.variance_history else 0.0

        # 自适应增益公式（改进版：更积极）
        # 原公式：过于保守，g = g0 * σ²_target / (σ²_target + Var)
        # 新公式：保持合理增益水平，不随方差急剧下降
        # g = g0 * (1 + α) / (1 + β * Var/σ²_target)，其中 α=1.0, β=0.3
        if self.target_variance > 0:
            variance_ratio = avg_variance / self.target_variance
            new_gain = self.base_gain * (2.0) / (1.0 + 0.3 * variance_ratio)
        else:
            new_gain = self.base_gain

        # 增益平滑变化
        self.current_gain = self.gain_smoothing * self.current_gain + (
            1 - self.gain_smoothing
        ) * new_gain

        # 增益范围限制（使用 min_gain 防止增益过低）
        self.current_gain = max(self.min_gain, min(1.0, self.current_gain))

        logger.debug(
            f"Updated adaptive gain: variance={avg_variance:.4f}, "
            f"new_gain={new_gain:.4f}, smoothed_gain={self.current_gain:.4f}"
        )

    def _record_injection(self, z: torch.Tensor, B: torch.Tensor) -> None:
        """记录注入历史（用于分析）"""
        record = {
            "step": self.total_injections,
            "z_norm": torch.norm(z).item(),
            "B_norm": torch.norm(B).item(),
            "gain": self.current_gain
        }
        self.injection_history.append(record)

        # 限制历史长度
        if len(self.injection_history) > 1000:
            self.injection_history = self.injection_history[-1000:]

    def get_full_dimension_injection(
        self,
        z: torch.Tensor,
        full_dim: int = 2048
    ) -> torch.Tensor:
        """
        获取全维度注入信号

        注入仅耦合到核心子空间，其余维度为零。

        Args:
            z: 混沌状态向量
            full_dim: 全维度

        Returns:
            全维度注入信号 (full_dim,)
        """
        # 核心子空间注入
        B_core = self.inject(z)

        # 扩展到全维度（其余为零）
        B_full = torch.zeros(full_dim, device=self.device)
        B_full[:self.core_dim] = B_core

        return B_full

    def project_to_core(
        self,
        E_fast: torch.Tensor,
        projection_matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        将快变量投影到核心子空间

        用于计算自适应增益所需的方差。

        Args:
            E_fast: 快变量向量 (full_dim,)
            projection_matrix: 核心子空间投影矩阵 (core_dim, full_dim)
                               如果未提供，使用假设的前 core_dim 维

        Returns:
            核心子空间分量 (core_dim,)
        """
        E_fast = E_fast.to(self.device)

        if projection_matrix is not None:
            # 使用提供的投影矩阵
            return torch.mv(projection_matrix.to(self.device), E_fast)
        else:
            # 假设核心子空间在前 core_dim 维
            return E_fast[:self.core_dim]

    def get_statistics(self) -> Dict:
        """
        获取注入器统计信息

        Returns:
            统计信息字典
        """
        stats = {
            "total_injections": self.total_injections,
            "current_gain": self.current_gain,
            "base_gain": self.base_gain,
            "target_variance": self.target_variance,
            "average_variance": np.mean(self.variance_history) if self.variance_history else 0.0,
            "projection_matrix_shape": self.W.shape,
            "projection_matrix_orthogonal_error": torch.norm(
                torch.mm(self.W.t(), self.W) - torch.eye(self.chaos_dim, device=self.device)
            ).item()
        }

        return stats

    def reset(self, seed: Optional[int] = None) -> None:
        """
        重置注入器状态

        Args:
            seed: 新的随机种子（如果为None，保留原投影矩阵）
        """
        self.current_gain = self.base_gain
        self.variance_history.clear()
        self.injection_history.clear()
        self.total_injections = 0

        # 如果提供了新种子，重新生成投影矩阵
        if seed is not None:
            self.W = self._generate_orthogonal_matrix(seed)

        logger.info("ChaosInjector reset")

    def save_projection_matrix(self, filepath: str) -> None:
        """
        保存投影矩阵到文件

        Args:
            filepath: 文件路径
        """
        W_np = self.W.detach().cpu().numpy()
        np.save(filepath, W_np)
        logger.info(f"Saved projection matrix to {filepath}")

    def load_projection_matrix(self, filepath: str) -> None:
        """
        从文件加载投影矩阵

        Args:
            filepath: 文件路径
        """
        W_np = np.load(filepath)
        self.W = torch.tensor(W_np, dtype=torch.float32, device=self.device)
        self._verify_orthogonality(self.W)
        logger.info(f"Loaded projection matrix from {filepath}")

    def set_gain(self, gain: float) -> None:
        """
        手动设置增益

        Args:
            gain: 新的增益值
        """
        self.current_gain = max(0.001, min(1.0, gain))
        logger.debug(f"Manual gain set to {self.current_gain}")

    def __repr__(self) -> str:
        return (
            f"ChaosInjector(core_dim={self.core_dim}, chaos_dim={self.chaos_dim}, "
            f"gain={self.current_gain:.4f}, injections={self.total_injections})"
        )


class CoreSubspaceProjector:
    """
    核心子空间投影器

    管理核心子空间的投影和恢复操作。

    用于：
    - 将高维状态投影到核心子空间
    - 将核心子空间信号注入回高维状态
    """

    def __init__(
        self,
        full_dim: int = 2048,
        core_dim: int = 64,
        device: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        初始化核心子空间投影器

        Args:
            full_dim: 全维度
            core_dim: 核心子空间维度
            device: 计算设备
            seed: 随机种子
        """
        self.full_dim = full_dim
        self.core_dim = core_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # 生成投影矩阵 P_core (core_dim, full_dim)
        self.P = self._generate_projection_matrix(seed)

        logger.info(
            f"CoreSubspaceProjector initialized: full_dim={full_dim}, "
            f"core_dim={core_dim}"
        )

    def _generate_projection_matrix(self, seed: Optional[int] = None) -> torch.Tensor:
        """
        生成核心子空间投影矩阵

        使用随机正交投影，确保核心子空间的独立性。

        Args:
            seed: 随机种子

        Returns:
            投影矩阵 (core_dim, full_dim)
        """
        if seed is not None:
            np.random.seed(seed)

        # 生成随机矩阵并正交化行
        P_np = np.random.randn(self.core_dim, self.full_dim)

        # 对行进行 QR 正交化
        P_np = np.linalg.qr(P_np.T)[0].T[:self.core_dim]

        # 归一化行向量
        row_norms = np.linalg.norm(P_np, axis=1, keepdims=True)
        P_np = P_np / row_norms

        return torch.tensor(P_np, dtype=torch.float32, device=self.device)

    def project(self, E_full: torch.Tensor) -> torch.Tensor:
        """
        投影到核心子空间

        Args:
            E_full: 全维度向量 (full_dim,)

        Returns:
            核心子空间向量 (core_dim,)
        """
        E_full = E_full.to(self.device)
        return torch.mv(self.P, E_full)

    def inject_to_full(self, B_core: torch.Tensor) -> torch.Tensor:
        """
        将核心子空间信号注入到全维度

        使用投影矩阵的转置进行伪逆注入。

        Args:
            B_core: 核心子空间信号 (core_dim,)

        Returns:
            全维度注入信号 (full_dim,)
        """
        B_core = B_core.to(self.device)
        # 使用 P^T 进行注入
        return torch.mv(self.P.t(), B_core)

    def get_matrix(self) -> torch.Tensor:
        """获取投影矩阵"""
        return self.P.clone()

    def __repr__(self) -> str:
        return (
            f"CoreSubspaceProjector(full_dim={self.full_dim}, "
            f"core_dim={self.core_dim})"
        )