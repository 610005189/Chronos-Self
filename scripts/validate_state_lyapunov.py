"""
验证三种动力学状态（REST, WORK, EXPLORE）的 Lyapunov 指数差异
================================================================

任务：
1. 分别在三种状态下运行系统（各运行 2000 步）
2. 使用快速 Jacobian Lyapunov 计算，获取每种状态的 λ₁ 值
3. 统计分析：三种状态的 λ₁ 均值差异是否显著（t 检验 p<0.01）
4. 生成状态控制效果报告

验收标准：
- 三种状态 λ₁ 均值有显著差异（t 检验 p<0.01）
- WORK 状态 λ₁ ∈ (0, 0.2)（如可达）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from scipy import stats
import logging
from pathlib import Path
import time
from typing import Dict, List, Any
from dataclasses import dataclass

from chronos_core.core.fast_dynamics import FastDynamicsSystem, FastDynamicsConfig, FastDynamicsFunction
from chronos_core.core.state_controller import StateMode, StateController, STATE_PARAMS_CONFIG
from chronos_core.core.state import SelfState
from chronos_core.utils.config import ChronosConfig

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class StateLyapunovResult:
    """状态 Lyapunov 测试结果"""
    state_mode: str
    lyapunov_values: List[float]
    mean: float
    std: float
    min: float
    max: float
    num_samples: int


def calculate_lyapunov_jacobian(
    dynamics_fn: FastDynamicsFunction,
    state: SelfState,
    num_exponents: int = 10,
    calc_steps: int = 100,
    dt: float = 0.01,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    使用 Jacobian QR 方法快速计算 Lyapunov 指数
    
    Args:
        dynamics_fn: 动力学函数
        state: 当前状态
        num_exponents: 要计算的指数数量
        calc_steps: 计算步数
        dt: 时间步长
        device: 计算设备
        
    Returns:
        计算结果字典
    """
    try:
        fast_dim = state.E_fast.shape[-1]
        k = min(num_exponents, fast_dim)
        
        # 初始化 QR 分解所需的矩阵 (fast_dim x k)
        Q = torch.eye(fast_dim, k, device=device)  # 创建 rectangular identity matrix
        log_evolutions = torch.zeros(k, device=device)
        
        current_state = state.copy()
        E_fast = current_state.E_fast.to(device)
        E_slow = current_state.E_slow.to(device)
        
        for step in range(calc_steps):
            # 计算动力学导数
            y = E_fast.unsqueeze(0).clone()
            y.requires_grad_(True)
            
            t_tensor = torch.tensor([step * dt], device=device)
            
            # 定义动力学函数（用于 Jacobian 计算）
            def dynamics_for_jacobian(x):
                return dynamics_fn.forward(
                    t_tensor,
                    x,
                    E_slow=E_slow.unsqueeze(0),
                    X_sem=None,
                    X_log=None,
                    X_fused=None,
                    C_meta=None,
                    B_chaos=None
                ).squeeze(0)
            
            # 计算 Jacobian
            J = torch.autograd.functional.jacobian(
                dynamics_for_jacobian,
                y,
                create_graph=False
            )
            
            # 处理 Jacobian 的维度：如果输入是 (1, dim)，输出是 (dim,)
            # 则 Jacobian 是 (dim, 1, dim)，squeeze 到 (dim, dim)
            if J.dim() == 3:
                J = J.squeeze(1)  # 从 (dim, 1, dim) -> (dim, dim)
            
            # QR 分解
            JQ = J @ Q[:, :k]
            Q_new, R = torch.linalg.qr(JQ)
            
            # 累积对角元素的对数
            diag_R = torch.diag(R).abs()
            # 防止 log(0)
            diag_R = torch.clamp(diag_R, min=1e-10)
            log_evolutions += torch.log(diag_R)
            
            # 更新 Q
            Q = Q_new
            
            # 更新状态（欧拉积分）
            with torch.no_grad():
                dydt = dynamics_fn.forward(
                    t_tensor,
                    E_fast.unsqueeze(0),
                    E_slow=E_slow.unsqueeze(0),
                    X_sem=None,
                    X_log=None,
                    X_fused=None,
                    C_meta=None,
                    B_chaos=None
                ).squeeze(0)
                E_fast = E_fast + dt * dydt
                
                # 状态范数裁剪
                norm = torch.norm(E_fast).item()
                threshold = dynamics_fn.config.state_norm_threshold
                if norm > threshold:
                    E_fast = E_fast * (threshold / norm)
        
        # 计算 Lyapunov 指数谱
        total_time = calc_steps * dt
        lyapunov_spectrum = log_evolutions / total_time
        lyapunov_spectrum = torch.sort(lyapunov_spectrum, descending=True)[0]
        
        lambda_max = lyapunov_spectrum[0].item()
        lambda_sum = lyapunov_spectrum.sum().item()
        positive_count = (lyapunov_spectrum > 0).sum().item()
        
        return {
            "lambda_max": lambda_max,
            "lambda_mean": lyapunov_spectrum.mean().item(),
            "lambda_std": lyapunov_spectrum.std().item(),
            "lambda_sum": lambda_sum,
            "positive_count": positive_count,
            "spectrum": lyapunov_spectrum.cpu().numpy().tolist(),
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Jacobian Lyapunov 计算失败: {e}")
        return {
            "lambda_max": np.nan,
            "lambda_mean": np.nan,
            "lambda_std": np.nan,
            "lambda_sum": np.nan,
            "positive_count": 0,
            "spectrum": [],
            "success": False,
            "error": str(e)
        }


def run_state_experiment(
    state_mode: StateMode,
    num_steps: int = 2000,
    num_lyapunov_samples: int = 10,
    lyapunov_calc_steps: int = 100,
    dt: float = 0.01,
    device: str = 'cpu',
    seed: int = 42
) -> StateLyapunovResult:
    """
    在指定状态下运行实验并测量 Lyapunov 指数
    
    Args:
        state_mode: 状态模式
        num_steps: 总运行步数
        num_lyapunov_samples: Lyapunov 测量次数
        lyapunov_calc_steps: 每次 Lyapunov 计算的步数
        dt: 时间步长
        device: 计算设备
        seed: 随机种子
        
    Returns:
        StateLyapunovResult: 测试结果
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"开始 {state_mode.value} 状态实验")
    logger.info(f"{'='*60}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 获取状态参数配置
    state_params = STATE_PARAMS_CONFIG[state_mode]
    
    # 创建动力学配置（应用状态参数）
    config = FastDynamicsConfig(
        fast_dim=256,  # 使用较小维度以提高计算速度
        slow_dim=64,
        semantic_dim=64,
        physical_dim=64,
        fusion_dim=128,
        meta_cognitive_dim=16,
        chaos_dim=32,
        hidden_dim=128,
        num_hidden_layers=2,
        activation="tanh",
        
        # 应用状态参数
        decay_rate=state_params.decay_rate,
        gamma=state_params.gamma,
        dynamics_scale=state_params.dynamics_scale,
        noise_scale=state_params.noise_scale,
        ei_ratio=state_params.ei_ratio,
        alpha=state_params.alpha,
        state_norm_threshold=state_params.state_norm_threshold,
        max_gradient_norm=state_params.max_gradient_norm,
        
        state_mode=state_mode.value
    )
    
    logger.info(f"状态参数:")
    logger.info(f"  decay_rate={state_params.decay_rate}")
    logger.info(f"  gamma={state_params.gamma}")
    logger.info(f"  dynamics_scale={state_params.dynamics_scale}")
    logger.info(f"  noise_scale={state_params.noise_scale}")
    
    # 创建动力学系统
    system = FastDynamicsSystem(config=config, device=device)
    system.initialize()
    
    # 强制切换到目标状态
    system.switch_state(state_mode, force=True)
    logger.info(f"系统状态: {system.get_current_state_mode().value}")
    
    # 初始化状态
    E_fast = torch.randn(config.fast_dim, device=device) * 0.1
    E_slow = torch.zeros(config.slow_dim, device=device)
    
    # 预热运行（消除初始 transient）
    warmup_steps = 100
    logger.info(f"预热运行 {warmup_steps} 步...")
    for i in range(warmup_steps):
        E_fast = system.step(E_fast, E_slow, dt=dt, t=i*dt)
    
    # Lyapunov 测量间隔
    lyapunov_interval = num_steps // num_lyapunov_samples
    
    lyapunov_values = []
    norms_history = []
    
    logger.info(f"开始运行 {num_steps} 步，每 {lyapunov_interval} 步测量一次 Lyapunov...")
    
    start_time = time.time()
    
    for step in range(num_steps):
        # 运行系统
        E_fast = system.step(E_fast, E_slow, dt=dt, t=(warmup_steps + step) * dt)
        
        # 记录范数
        norms_history.append(torch.norm(E_fast).item())
        
        # 定期测量 Lyapunov 指数
        if step % lyapunov_interval == 0 and step > 0:
            current_state = SelfState(
                E_fast=E_fast.detach().cpu(),
                E_slow=E_slow.detach().cpu(),
                timestamp=(warmup_steps + step) * dt
            )
            
            result = calculate_lyapunov_jacobian(
                system.dynamics_fn,
                current_state,
                num_exponents=5,  # 只计算前 5 个指数
                calc_steps=lyapunov_calc_steps,
                dt=dt,
                device=device
            )
            
            if result["success"]:
                lyapunov_values.append(result["lambda_max"])
                logger.info(
                    f"  步 {step}: λ_max={result['lambda_max']:.6f}, "
                    f"norm={torch.norm(E_fast).item():.4f}"
                )
            else:
                logger.warning(f"  步 {step}: Lyapunov 计算失败")
    
    elapsed_time = time.time() - start_time
    
    # 计算统计值
    if len(lyapunov_values) > 0:
        mean_val = np.mean(lyapunov_values)
        std_val = np.std(lyapunov_values)
        min_val = np.min(lyapunov_values)
        max_val = np.max(lyapunov_values)
        
        logger.info(f"\n{state_mode.value} 状态结果:")
        logger.info(f"  λ₁ 均值: {mean_val:.6f}")
        logger.info(f"  λ₁ 标准差: {std_val:.6f}")
        logger.info(f"  λ₁ 范围: [{min_val:.6f}, {max_val:.6f}]")
        logger.info(f"  样本数: {len(lyapunov_values)}")
        logger.info(f"  运行时间: {elapsed_time:.2f}秒")
        
        return StateLyapunovResult(
            state_mode=state_mode.value,
            lyapunov_values=lyapunov_values,
            mean=mean_val,
            std=std_val,
            min=min_val,
            max=max_val,
            num_samples=len(lyapunov_values)
        )
    else:
        logger.error(f"{state_mode.value} 状态未能获取有效的 Lyapunov 测量")
        return StateLyapunovResult(
            state_mode=state_mode.value,
            lyapunov_values=[],
            mean=np.nan,
            std=np.nan,
            min=np.nan,
            max=np.nan,
            num_samples=0
        )


def perform_statistical_analysis(
    results: Dict[str, StateLyapunovResult]
) -> Dict[str, Any]:
    """
    对三种状态进行统计分析
    
    Args:
        results: 三种状态的测试结果
        
    Returns:
        统计分析结果
    """
    logger.info("\n" + "="*60)
    logger.info("统计分析")
    logger.info("="*60)
    
    analysis = {
        "pairwise_t_tests": {},
        "anova_result": None,
        "summary": {}
    }
    
    # 提取各状态的 Lyapunov 值
    rest_values = results["rest"].lyapunov_values
    work_values = results["work"].lyapunov_values
    explore_values = results["explore"].lyapunov_values
    
    # 检查是否有足够的数据
    if len(rest_values) < 2 or len(work_values) < 2 or len(explore_values) < 2:
        logger.error("数据不足，无法进行统计分析")
        return analysis
    
    # 配对 t 检验
    pairs = [
        ("REST vs WORK", rest_values, work_values),
        ("REST vs EXPLORE", rest_values, explore_values),
        ("WORK vs EXPLORE", work_values, explore_values)
    ]
    
    for pair_name, values1, values2 in pairs:
        t_stat, p_value = stats.ttest_ind(values1, values2)
        
        analysis["pairwise_t_tests"][pair_name] = {
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.01,
            "mean_diff": np.mean(values1) - np.mean(values2)
        }
        
        logger.info(f"{pair_name}:")
        logger.info(f"  t 统计量: {t_stat:.4f}")
        logger.info(f"  p 值: {p_value:.6f}")
        logger.info(f"  显著性 (p<0.01): {'✓' if p_value < 0.01 else '✗'}")
        logger.info(f"  均值差异: {np.mean(values1) - np.mean(values2):.6f}")
    
    # ANOVA 检验（整体差异）
    f_stat, p_value_anova = stats.f_oneway(rest_values, work_values, explore_values)
    
    analysis["anova_result"] = {
        "f_statistic": f_stat,
        "p_value": p_value_anova,
        "significant": p_value_anova < 0.01
    }
    
    logger.info(f"\nANOVA 检验:")
    logger.info(f"  F 统计量: {f_stat:.4f}")
    logger.info(f"  p 值: {p_value_anova:.6f}")
    logger.info(f"  显著性 (p<0.01): {'✓' if p_value_anova < 0.01 else '✗'}")
    
    # 验收标准检查
    logger.info("\n验收标准检查:")
    
    # 标准 1: 三种状态 λ₁ 均值有显著差异（t 检验 p<0.01）
    criterion1_passed = (
        analysis["pairwise_t_tests"]["REST vs WORK"]["significant"] and
        analysis["pairwise_t_tests"]["REST vs EXPLORE"]["significant"] and
        analysis["pairwise_t_tests"]["WORK vs EXPLORE"]["significant"]
    )
    
    logger.info(f"  标准 1（三种状态显著差异）: {'✓ 通过' if criterion1_passed else '✗ 未通过'}")
    
    # 标准 2: WORK 状态 λ₁ ∈ (0, 0.2)
    work_mean = results["work"].mean
    criterion2_passed = 0 < work_mean < 0.2
    
    logger.info(f"  标准 2（WORK λ₁ ∈ (0, 0.2)）: {'✓ 通过' if criterion2_passed else '✗ 未通过'}")
    logger.info(f"    WORK λ₁ = {work_mean:.6f}")
    
    analysis["summary"] = {
        "criterion1_passed": criterion1_passed,
        "criterion2_passed": criterion2_passed,
        "overall_passed": criterion1_passed and criterion2_passed
    }
    
    return analysis


def generate_report(
    results: Dict[str, StateLyapunovResult],
    analysis: Dict[str, Any],
    output_path: str
) -> None:
    """
    生成 Markdown 报告
    
    Args:
        results: 测试结果
        analysis: 统计分析结果
        output_path: 输出路径
    """
    logger.info(f"\n生成报告: {output_path}")
    
    report_lines = []
    report_lines.append("# 状态控制效果验证报告")
    report_lines.append("")
    report_lines.append(f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # 实验概述
    report_lines.append("## 1. 实验概述")
    report_lines.append("")
    report_lines.append("验证三种动力学状态（REST, WORK, EXPLORE）的 Lyapunov 指数是否有显著差异。")
    report_lines.append("")
    report_lines.append("### 实验参数")
    report_lines.append("")
    report_lines.append("- 每种状态运行步数: 2000")
    report_lines.append("- Lyapunov 测量次数: 10")
    report_lines.append("- Lyapunov 计算方法: Jacobian QR 分解")
    report_lines.append("- 计算步数: 100 步/测量")
    report_lines.append("- 时间步长: 0.01")
    report_lines.append("- 状态维度: fast_dim=256")
    report_lines.append("")
    
    # 状态参数配置
    report_lines.append("## 2. 状态参数配置")
    report_lines.append("")
    
    for mode in [StateMode.REST, StateMode.WORK, StateMode.EXPLORE]:
        params = STATE_PARAMS_CONFIG[mode]
        report_lines.append(f"### {mode.value.upper()} 状态")
        report_lines.append("")
        report_lines.append(f"| 参数 | 值 |")
        report_lines.append(f"|------|-----|")
        report_lines.append(f"| decay_rate | {params.decay_rate} |")
        report_lines.append(f"| gamma | {params.gamma} |")
        report_lines.append(f"| dynamics_scale | {params.dynamics_scale} |")
        report_lines.append(f"| noise_scale | {params.noise_scale} |")
        report_lines.append(f"| alpha | {params.alpha} |")
        report_lines.append(f"| state_norm_threshold | {params.state_norm_threshold} |")
        report_lines.append("")
    
    # 测试结果
    report_lines.append("## 3. Lyapunov 指数测量结果")
    report_lines.append("")
    
    report_lines.append("| 状态 | λ₁ 均值 | λ₁ 标准差 | λ₁ 最小值 | λ₁ 最大值 | 样本数 |")
    report_lines.append("|------|---------|-----------|-----------|-----------|--------|")
    
    for mode_name in ["rest", "work", "explore"]:
        r = results[mode_name]
        if r.num_samples > 0:
            report_lines.append(
                f"| {mode_name.upper()} | {r.mean:.6f} | {r.std:.6f} | "
                f"{r.min:.6f} | {r.max:.6f} | {r.num_samples} |"
            )
        else:
            report_lines.append(f"| {mode_name.upper()} | N/A | N/A | N/A | N/A | 0 |")
    
    report_lines.append("")
    
    # 详细测量值
    report_lines.append("### 各状态 Lyapunov 指数详细值")
    report_lines.append("")
    
    for mode_name in ["rest", "work", "explore"]:
        r = results[mode_name]
        report_lines.append(f"**{mode_name.upper()}**:")
        report_lines.append("")
        if r.num_samples > 0:
            report_lines.append(f"```")
            for i, val in enumerate(r.lyapunov_values):
                report_lines.append(f"  测量 {i+1}: λ₁ = {val:.6f}")
            report_lines.append(f"```")
        else:
            report_lines.append("无有效测量")
        report_lines.append("")
    
    # 统计分析
    report_lines.append("## 4. 统计分析结果")
    report_lines.append("")
    
    # 配对 t 检验
    report_lines.append("### 配对 t 检验")
    report_lines.append("")
    report_lines.append("| 比较组 | t 统计量 | p 值 | 显著性 (p<0.01) | 均值差异 |")
    report_lines.append("|--------|----------|------|----------------|----------|")
    
    for pair_name, test_result in analysis["pairwise_t_tests"].items():
        sig_mark = "✓" if test_result["significant"] else "✗"
        report_lines.append(
            f"| {pair_name} | {test_result['t_statistic']:.4f} | "
            f"{test_result['p_value']:.6f} | {sig_mark} | "
            f"{test_result['mean_diff']:.6f} |"
        )
    
    report_lines.append("")
    
    # ANOVA
    if analysis["anova_result"]:
        report_lines.append("### ANOVA 检验")
        report_lines.append("")
        anova = analysis["anova_result"]
        sig_mark = "✓" if anova["significant"] else "✗"
        report_lines.append(f"- F 统计量: {anova['f_statistic']:.4f}")
        report_lines.append(f"- p 值: {anova['p_value']:.6f}")
        report_lines.append(f"- 显著性 (p<0.01): {sig_mark}")
        report_lines.append("")
    
    # 验收标准
    report_lines.append("## 5. 验收标准检查")
    report_lines.append("")
    
    summary = analysis.get("summary", {})
    
    # 标准 1
    c1 = summary.get("criterion1_passed", False)
    c1_mark = "✓ 通过" if c1 else "✗ 未通过"
    report_lines.append(f"### 标准 1: 三种状态 λ₁ 均值有显著差异（t 检验 p<0.01）")
    report_lines.append("")
    report_lines.append(f"**结果**: {c1_mark}")
    report_lines.append("")
    
    if not c1:
        report_lines.append("**分析**: ")
        for pair_name, test_result in analysis["pairwise_t_tests"].items():
            if not test_result["significant"]:
                report_lines.append(
                    f"- {pair_name}: p={test_result['p_value']:.6f} > 0.01，差异不显著"
                )
        report_lines.append("")
    
    # 标准 2
    c2 = summary.get("criterion2_passed", False)
    c2_mark = "✓ 通过" if c2 else "✗ 未通过"
    work_mean = results["work"].mean
    
    report_lines.append(f"### 标准 2: WORK 状态 λ₁ ∈ (0, 0.2)")
    report_lines.append("")
    report_lines.append(f"**结果**: {c2_mark}")
    report_lines.append(f"- WORK λ₁ = {work_mean:.6f}")
    report_lines.append(f"- 目标区间: (0, 0.2)")
    report_lines.append("")
    
    if not c2:
        if work_mean < 0:
            report_lines.append("**分析**: λ₁ < 0，系统过于稳定，混沌不足")
        elif work_mean > 0.2:
            report_lines.append("**分析**: λ₁ > 0.2，混沌过强，可能偏离边缘混沌")
        report_lines.append("")
    
    # 整体结论
    report_lines.append("## 6. 整体结论")
    report_lines.append("")
    
    overall = summary.get("overall_passed", False)
    overall_mark = "✓ 全部通过" if overall else "✗ 部分未通过"
    report_lines.append(f"### 验收状态: {overall_mark}")
    report_lines.append("")
    
    if overall:
        report_lines.append("**结论**: 状态控制功能有效，三种动力学状态的 Lyapunov 指数存在显著差异，")
        report_lines.append("且 WORK 状态处于边缘混沌区间，符合验收标准。")
    else:
        report_lines.append("**结论**: 状态控制效果需要优化。")
        report_lines.append("")
        report_lines.append("### 改进建议")
        report_lines.append("")
        
        if not c1:
            report_lines.append("1. **增强状态参数差异**: 当前各状态参数差异不够显著，")
            report_lines.append("   建议增加 decay_rate 和 gamma 的差异幅度。")
            report_lines.append("")
        
        if not c2:
            report_lines.append("2. **调整 WORK 状态参数**: WORK 状态的 Lyapunov 指数未落入目标区间，")
            report_lines.append("   建议调整 dynamics_scale 或 gamma 参数。")
            report_lines.append("")
    
    # 写入文件
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"报告已保存: {output_path}")


def main():
    """主函数"""
    logger.info("="*80)
    logger.info("状态 Lyapunov 指数验证实验")
    logger.info("="*80)
    
    device = 'cpu'
    num_steps = 2000
    num_lyapunov_samples = 10
    lyapunov_calc_steps = 100
    
    # 运行三种状态的实验
    results = {}
    
    for mode in [StateMode.REST, StateMode.WORK, StateMode.EXPLORE]:
        results[mode.value] = run_state_experiment(
            state_mode=mode,
            num_steps=num_steps,
            num_lyapunov_samples=num_lyapunov_samples,
            lyapunov_calc_steps=lyapunov_calc_steps,
            device=device,
            seed=42
        )
    
    # 统计分析
    analysis = perform_statistical_analysis(results)
    
    # 生成报告
    output_path = "results/state_validation_report.md"
    generate_report(results, analysis, output_path)
    
    # 最终总结
    logger.info("\n" + "="*80)
    logger.info("实验完成")
    logger.info("="*80)
    
    overall_passed = analysis.get("summary", {}).get("overall_passed", False)
    
    if overall_passed:
        logger.info("✓ 验收标准全部通过")
    else:
        logger.info("✗ 部分验收标准未通过，请查看报告了解详情")
    
    logger.info(f"\n报告位置: {output_path}")
    
    return overall_passed


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)