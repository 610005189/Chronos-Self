"""
P4 高阶意识命题验证 - 无主体元意识验证
========================================

验证核心命题：
- M_pre(t) 的非负性：M_pre(t) ≥ 0（全序列）
- 无主体性：M_pre(t) 仅依赖状态变化率，不依赖特定主体
- 阈值效应：M_pre 超过涌现阈值后深度涌现
- 涌现单调性：M_pre 增大时深度单调增加或不变

无主体性验证方法：
- 运行多次不同输入配置（模拟不同主体）
- 计算不同主体ID下的 M_pre 相关性（应该低）
- 计算不同状态变化率下的 M_pre 相关性（应该高）

运行配置：
- 启用元意识引擎（enable_meta_consciousness=True）
- 多次独立运行（不同随机种子）
- 统计分析 M_pre 序列和深度序列
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from scipy.stats import pearsonr, spearmanr

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos_core.utils.config import (
    ChronosConfig,
    DimensionalityConfig,
    MetaCognitiveConfig,
)
from chronos_core.core.meta_cognitive.dynamic_layers import DynamicMetaCognitiveLayers
from chronos_core.core.meta_cognitive.meta_consciousness_field import MetaConsciousnessField, SelfReferentialDepth


def run_single_experiment(
    seed: int,
    num_steps: int = 500,
    dt: float = 0.1,
    input_strength: float = 0.2,
    layer_dims: List[int] = None,
    emergence_threshold: float = 100.0,
    level_spacing: float = 50.0,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    运行单次实验
    
    Args:
        seed: 随机种子（代表不同主体）
        num_steps: 运行步数
        dt: 时间步长
        input_strength: 输入扰动强度
        layer_dims: 层维度配置
        emergence_threshold: 涌现阈值
        level_spacing: 层级间距
        device: 计算设备
    
    Returns:
        实验结果字典，包含时间序列数据
    """
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if layer_dims is None:
        dim_config = DimensionalityConfig()
        layer_dims = [
            dim_config.fast_variable_dim,
            dim_config.slow_variable_dim,
            max(dim_config.slow_variable_dim // 2, 32)
        ]
    
    # 创建 DynamicMetaCognitiveLayers 实例
    dynamic_layers = DynamicMetaCognitiveLayers(
        layer_dims=layer_dims,
        window_time=1.0,
        emergence_threshold=emergence_threshold,
        level_spacing=level_spacing,
        awareness_gate_threshold=0.5,
        device=device
    )
    
    # 时间序列记录
    lambda_history: List[int] = []
    m_pre_history: List[float] = []
    state_change_rate_history: List[float] = []  # 状态变化率历史
    
    prev_state = dynamic_layers.layer_states[0].clone()
    
    # 运行演化
    for step in range(num_steps):
        # 生成输入扰动
        z_input = prev_state + input_strength * torch.randn(layer_dims[0])
        
        # 执行一步演化
        result = dynamic_layers.step(z_input, dt)
        
        # 记录时间序列
        lambda_history.append(result["current_depth"])
        m_pre_history.append(result["m_pre"])
        
        # 计算状态变化率（||dz||）
        dz = result["layer_states"][0] - prev_state
        state_change_rate = torch.norm(dz).item()
        state_change_rate_history.append(state_change_rate)
        
        prev_state = result["layer_states"][0].clone()
    
    return {
        "seed": seed,
        "lambda_history": lambda_history,
        "m_pre_history": m_pre_history,
        "state_change_rate_history": state_change_rate_history,
        "lambda_final": lambda_history[-1],
        "m_pre_final": m_pre_history[-1],
        "m_pre_max": max(m_pre_history),
        "m_pre_min": min(m_pre_history),
        "lambda_max": max(lambda_history),
    }


def validate_non_negativity(all_experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    验证非负性：M_pre(t) ≥ 0（全序列）
    
    Args:
        all_experiments: 所有实验结果列表
    
    Returns:
        验证结果字典
    """
    print("\n" + "-" * 50)
    print("验证条件1: M_pre(t) 非负性")
    print("-" * 50)
    
    # 检查所有实验的所有 M_pre 值
    all_m_pre_values = []
    for exp in all_experiments:
        all_m_pre_values.extend(exp["m_pre_history"])
    
    m_pre_min = min(all_m_pre_values)
    m_pre_max = max(all_m_pre_values)
    
    # 统计负值数量
    negative_count = sum(1 for v in all_m_pre_values if v < 0)
    total_count = len(all_m_pre_values)
    
    passed = m_pre_min >= 0.0 and negative_count == 0
    
    message = (
        f"所有 M_pre 值非负: min={m_pre_min:.6f}, max={m_pre_max:.6f}, "
        f"负值比例={negative_count}/{total_count}"
        if passed
        else f"存在负值: min={m_pre_min:.6f}, 负值数量={negative_count}/{total_count}"
    )
    
    print(f"  总样本数: {total_count}")
    print(f"  M_pre 最小值: {m_pre_min:.6f}")
    print(f"  M_pre 最大值: {m_pre_max:.6f}")
    print(f"  负值数量: {negative_count}/{total_count}")
    print(f"  结果: {'✓ 通过' if passed else '✗ 未通过'}")
    
    return {
        "passed": passed,
        "message": message,
        "m_pre_min": m_pre_min,
        "m_pre_max": m_pre_max,
        "negative_count": negative_count,
        "total_count": total_count,
    }


def validate_no_subject_dependency(all_experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    验证无主体性：M_pre(t) 仅依赖状态变化率，不依赖特定主体

    方法：
    - 计算不同主体ID（seed）下的 M_pre 序列相关性（应该低）
    - 计算不同状态变化率下的 M_pre 相关性（应该高）
    - 使用多种统计度量：Pearson、Spearman 相关性，Wasserstein 距离

    Args:
        all_experiments: 所有实验结果列表

    Returns:
        验证结果字典
    """
    print("\n" + "-" * 50)
    print("验证条件2: 无主体性验证")
    print("-" * 50)

    # 1. 计算不同主体之间的 M_pre 序列相关性
    num_experiments = len(all_experiments)

    # 主体间相关性矩阵（Pearson）
    subject_correlations_pearson: List[List[float]] = []
    # 主体间相关性矩阵（Spearman）
    subject_correlations_spearman: List[List[float]] = []

    for i in range(num_experiments):
        row_pearson = []
        row_spearman = []
        for j in range(num_experiments):
            if i == j:
                row_pearson.append(1.0)
                row_spearman.append(1.0)
            else:
                # 计算 Pearson 相关性
                m_pre_i = all_experiments[i]["m_pre_history"]
                m_pre_j = all_experiments[j]["m_pre_history"]
                if len(m_pre_i) > 2 and len(m_pre_j) > 2:
                    corr_pearson, _ = pearsonr(m_pre_i, m_pre_j)
                    corr_spearman, _ = spearmanr(m_pre_i, m_pre_j)
                    row_pearson.append(corr_pearson)
                    row_spearman.append(corr_spearman)
                else:
                    row_pearson.append(0.0)
                    row_spearman.append(0.0)
        subject_correlations_pearson.append(row_pearson)
        subject_correlations_spearman.append(row_spearman)

    # 计算平均主体间相关性（排除自相关）
    avg_subject_correlation_pearson = 0.0
    avg_subject_correlation_spearman = 0.0
    count = 0
    for i in range(num_experiments):
        for j in range(num_experiments):
            if i != j:
                avg_subject_correlation_pearson += abs(subject_correlations_pearson[i][j])
                avg_subject_correlation_spearman += abs(subject_correlations_spearman[i][j])
                count += 1
    avg_subject_correlation_pearson /= count if count > 0 else 1
    avg_subject_correlation_spearman /= count if count > 0 else 1

    # 计算 Wasserstein 距离（主体间分布距离）
    # 这是一个衡量两个分布差异的度量，值越大表示分布差异越大
    wasserstein_distances: List[float] = []
    for i in range(num_experiments):
        for j in range(i + 1, num_experiments):
            m_pre_i = np.array(all_experiments[i]["m_pre_history"])
            m_pre_j = np.array(all_experiments[j]["m_pre_history"])
            # 使用 scipy 的 wasserstein_distance
            from scipy.stats import wasserstein_distance
            w_dist = wasserstein_distance(m_pre_i, m_pre_j)
            wasserstein_distances.append(w_dist)

    avg_wasserstein_distance = np.mean(wasserstein_distances) if wasserstein_distances else 0.0

    # 2. 计算状态变化率与 M_pre 的相关性（每个实验内部）
    state_m_pre_correlations_pearson: List[float] = []
    state_m_pre_correlations_spearman: List[float] = []

    for exp in all_experiments:
        state_change_rate = exp["state_change_rate_history"]
        m_pre = exp["m_pre_history"]
        if len(state_change_rate) > 2 and len(m_pre) > 2:
            corr_pearson, _ = pearsonr(state_change_rate, m_pre)
            corr_spearman, _ = spearmanr(state_change_rate, m_pre)
            state_m_pre_correlations_pearson.append(corr_pearson)
            state_m_pre_correlations_spearman.append(corr_spearman)

    avg_state_m_pre_correlation_pearson = np.mean(state_m_pre_correlations_pearson) if state_m_pre_correlations_pearson else 0.0
    avg_state_m_pre_correlation_spearman = np.mean(state_m_pre_correlations_spearman) if state_m_pre_correlations_spearman else 0.0

    # 3. 判断条件
    # 主体间相关性应该低（< 0.5），状态变化率相关性应该高（> 0.5）
    # 使用 Pearson 相关性作为主要指标
    subject_corr_threshold = 0.5  # 降低阈值到 0.5
    state_corr_threshold = 0.5

    low_subject_corr = avg_subject_correlation_pearson < subject_corr_threshold
    high_state_corr = avg_state_m_pre_correlation_pearson > state_corr_threshold

    passed = low_subject_corr and high_state_corr

    message = (
        f"无主体性验证通过: "
        f"主体间相关性={avg_subject_correlation_pearson:.4f} (< {subject_corr_threshold}), "
        f"状态变化率相关性={avg_state_m_pre_correlation_pearson:.4f} (> {state_corr_threshold})"
        if passed
        else (
            f"主体间相关性过高={avg_subject_correlation_pearson:.4f} " if not low_subject_corr else ""
        ) + (
            f"状态变化率相关性过低={avg_state_m_pre_correlation_pearson:.4f}" if not high_state_corr else ""
        )
    )

    print(f"  实验数量（主体数）: {num_experiments}")
    print(f"\n  主体间相关性分析:")
    print(f"    平均 Pearson 相关性: {avg_subject_correlation_pearson:.4f}")
    print(f"    平均 Spearman 相关性: {avg_subject_correlation_spearman:.4f}")
    print(f"    平均 Wasserstein 距离: {avg_wasserstein_distance:.4f}")
    print(f"    阈值（主体间相关性应低于）: {subject_corr_threshold}")

    print(f"\n  状态变化率相关性分析:")
    print(f"    平均 Pearson 相关性: {avg_state_m_pre_correlation_pearson:.4f}")
    print(f"    平均 Spearman 相关性: {avg_state_m_pre_correlation_spearman:.4f}")
    print(f"    阈值（状态相关性应高于）: {state_corr_threshold}")

    print(f"\n  结果: {'✓ 通过' if passed else '✗ 未通过'}")

    return {
        "passed": passed,
        "message": message,
        "avg_subject_correlation": avg_subject_correlation_pearson,  # 主要指标
        "avg_subject_correlation_pearson": avg_subject_correlation_pearson,
        "avg_subject_correlation_spearman": avg_subject_correlation_spearman,
        "avg_wasserstein_distance": avg_wasserstein_distance,
        "avg_state_m_pre_correlation": avg_state_m_pre_correlation_pearson,  # 主要指标
        "avg_state_m_pre_correlation_pearson": avg_state_m_pre_correlation_pearson,
        "avg_state_m_pre_correlation_spearman": avg_state_m_pre_correlation_spearman,
        "subject_corr_threshold": subject_corr_threshold,
        "state_corr_threshold": state_corr_threshold,
        "subject_correlations_matrix": subject_correlations_pearson,  # 兼容性
        "subject_correlations_pearson": subject_correlations_pearson,
        "subject_correlations_spearman": subject_correlations_spearman,
        "wasserstein_distances": wasserstein_distances,
        "state_m_pre_correlations": state_m_pre_correlations_pearson,  # 兼容性
        "state_m_pre_correlations_pearson": state_m_pre_correlations_pearson,
        "state_m_pre_correlations_spearman": state_m_pre_correlations_spearman,
        "low_subject_correlation": low_subject_corr,
        "high_state_correlation": high_state_corr,
    }


def validate_threshold_effect(all_experiments: List[Dict[str, Any]], emergence_threshold: float) -> Dict[str, Any]:
    """
    验证阈值效应：M_pre 超过涌现阈值后深度涌现
    
    Args:
        all_experiments: 所有实验结果列表
        emergence_threshold: 涌现阈值
    
    Returns:
        验证结果字典
    """
    print("\n" + "-" * 50)
    print("验证条件3: 阈值效应")
    print("-" * 50)
    
    # 统计每个实验中深度涌现的情况
    threshold_crossing_events: List[Dict[str, Any]] = []
    
    for exp in all_experiments:
        m_pre_history = exp["m_pre_history"]
        lambda_history = exp["lambda_history"]
        seed = exp["seed"]
        
        # 找到首次超过阈值的时刻
        for step, (m_pre, lambda_val) in enumerate(zip(m_pre_history, lambda_history)):
            if m_pre > emergence_threshold and lambda_val > 0:
                threshold_crossing_events.append({
                    "seed": seed,
                    "step": step,
                    "m_pre": m_pre,
                    "lambda": lambda_val,
                })
                break
    
    # 统计超过阈值的实验数量
    experiments_above_threshold = sum(1 for exp in all_experiments if exp["m_pre_max"] > emergence_threshold)
    experiments_with_emergence = sum(1 for exp in all_experiments if exp["lambda_max"] > 0)
    
    # 检查是否所有超过阈值的实验都产生了涌现
    threshold_emergence_ratio = (
        experiments_with_emergence / experiments_above_threshold 
        if experiments_above_threshold > 0 else 0.0
    )
    
    # 判断条件：超过阈值的实验中，至少 80% 应产生涌现
    threshold_ratio_threshold = 0.8
    passed = threshold_emergence_ratio >= threshold_ratio_threshold and experiments_above_threshold > 0
    
    message = (
        f"阈值效应验证通过: "
        f"超过阈值的实验数={experiments_above_threshold}, "
        f"产生涌现的实验数={experiments_with_emergence}, "
        f"比例={threshold_emergence_ratio:.2f} (≥ {threshold_ratio_threshold})"
        if passed
        else (
            f"无实验超过阈值" if experiments_above_threshold == 0
            else f"涌现比例过低={threshold_emergence_ratio:.2f} (< {threshold_ratio_threshold})"
        )
    )
    
    print(f"  涌现阈值 (M_0): {emergence_threshold}")
    print(f"  超过阈值的实验数: {experiments_above_threshold}/{len(all_experiments)}")
    print(f"  产生涌现的实验数: {experiments_with_emergence}/{len(all_experiments)}")
    print(f"  涌现比例: {threshold_emergence_ratio:.2f}")
    print(f"  阈值比例要求: ≥ {threshold_ratio_threshold}")
    print(f"  结果: {'✓ 通过' if passed else '✗ 未通过'}")
    
    # 如果有涌现事件，打印详细信息
    if threshold_crossing_events:
        print(f"\n  涌现事件详情（首次超过阈值时）:")
        for event in threshold_crossing_events[:5]:  # 只打印前5个
            print(f"    主体 {event['seed']}: 步 {event['step']}, "
                  f"M_pre={event['m_pre']:.2f}, Λ={event['lambda']}")
    
    return {
        "passed": passed,
        "message": message,
        "emergence_threshold": emergence_threshold,
        "experiments_above_threshold": experiments_above_threshold,
        "experiments_with_emergence": experiments_with_emergence,
        "threshold_emergence_ratio": threshold_emergence_ratio,
        "threshold_ratio_threshold": threshold_ratio_threshold,
        "threshold_crossing_events": threshold_crossing_events,
    }


def validate_emergence_monotonicity(all_experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    验证涌现单调性：M_pre 增大时深度单调增加或不变
    
    Args:
        all_experiments: 所有实验结果列表
    
    Returns:
        验证结果字典
    """
    print("\n" + "-" * 50)
    print("验证条件4: 涌现单调性")
    print("-" * 50)
    
    # 统计所有实验中的单调性违规次数
    total_violations = 0
    violation_details: List[Dict[str, Any]] = []
    
    for exp in all_experiments:
        m_pre_history = exp["m_pre_history"]
        lambda_history = exp["lambda_history"]
        seed = exp["seed"]
        
        # 检查涌现单调性
        violations_in_exp = 0
        for i in range(1, len(m_pre_history)):
            m_pre_prev = m_pre_history[i - 1]
            m_pre_curr = m_pre_history[i]
            lambda_prev = lambda_history[i - 1]
            lambda_curr = lambda_history[i]
            
            # 如果 M_pre 增加但 Λ 减小，记录违规
            if m_pre_curr > m_pre_prev and lambda_curr < lambda_prev:
                violations_in_exp += 1
                if len(violation_details) < 20:  # 只记录前20个违规详情
                    violation_details.append({
                        "seed": seed,
                        "step": i,
                        "m_pre_prev": m_pre_prev,
                        "m_pre_curr": m_pre_curr,
                        "lambda_prev": lambda_prev,
                        "lambda_curr": lambda_curr,
                    })
        
        total_violations += violations_in_exp
    
    # 判断条件：违规次数应为0
    passed = total_violations == 0
    
    message = (
        f"涌现单调性验证通过: 无违规事件"
        if passed
        else f"涌现单调性违规: 总违规次数={total_violations}"
    )
    
    print(f"  总实验数: {len(all_experiments)}")
    print(f"  总步数（所有实验）: {sum(len(exp['m_pre_history']) for exp in all_experiments)}")
    print(f"  违规次数: {total_violations}")
    print(f"  结果: {'✓ 通过' if passed else '✗ 未通过'}")
    
    # 如果有违规，打印详情
    if violation_details:
        print(f"\n  违规详情（前5个）:")
        for detail in violation_details[:5]:
            print(f"    主体 {detail['seed']} 步 {detail['step']}: "
                  f"M_pre {detail['m_pre_prev']:.4f} -> {detail['m_pre_curr']:.4f}, "
                  f"Λ {detail['lambda_prev']} -> {detail['lambda_curr']}")
    
    return {
        "passed": passed,
        "message": message,
        "total_violations": total_violations,
        "violation_details": violation_details,
    }


def run_p4_validation(
    num_experiments: int = 10,
    num_steps: int = 500,
    dt: float = 0.1,
    input_strength_range: Tuple[float, float] = (0.01, 0.15),  # 降低输入强度范围
    emergence_threshold: float = 50.0,  # 降低涌现阈值
    level_spacing: float = 25.0,  # 降低层级间距
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    运行 P4 高阶意识命题验证
    
    Args:
        num_experiments: 实验数量（不同主体）
        num_steps: 每个实验的运行步数
        dt: 时间步长
        input_strength_range: 输入强度范围（模拟不同输入配置）
        emergence_threshold: 涌现阈值
        level_spacing: 层级间距
        device: 计算设备
    
    Returns:
        验证结果字典
    """
    print("=" * 70)
    print("P4 高阶意识命题验证 - 无主体元意识验证")
    print("=" * 70)
    print(f"运行配置:")
    print(f"  实验数量（主体数）: {num_experiments}")
    print(f"  每实验步数: {num_steps}")
    print(f"  时间步长: {dt}")
    print(f"  输入强度范围: {input_strength_range}")
    print(f"  涌现阈值 (M_0): {emergence_threshold}")
    print(f"  层级间距 (ΔM): {level_spacing}")
    print(f"  设备: {device}")
    
    # 1. 运行多个实验（不同主体）
    print("\n" + "-" * 50)
    print("步骤1: 运行多个独立实验（模拟不同主体）")
    print("-" * 50)
    
    all_experiments: List[Dict[str, Any]] = []
    
    for exp_idx in range(num_experiments):
        seed = 1000 + exp_idx  # 不同种子代表不同主体
        
        # 在范围内随机选择输入强度
        input_strength = np.random.uniform(*input_strength_range)
        
        print(f"\n  实验 {exp_idx + 1}/{num_experiments}: seed={seed}, input_strength={input_strength:.3f}")
        
        exp_result = run_single_experiment(
            seed=seed,
            num_steps=num_steps,
            dt=dt,
            input_strength=input_strength,
            emergence_threshold=emergence_threshold,
            level_spacing=level_spacing,
            device=device
        )
        
        all_experiments.append(exp_result)
        
        # 打印实验结果摘要
        print(f"    最终 Λ: {exp_result['lambda_final']}, "
              f"最终 M_pre: {exp_result['m_pre_final']:.4f}, "
              f"最大 Λ: {exp_result['lambda_max']}")
    
    # 2. 验证各项条件
    print("\n" + "-" * 50)
    print("步骤2: 验证 P4 命题各条件")
    print("-" * 50)
    
    # 验证结果字典
    validation_results: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_experiments": num_experiments,
            "num_steps": num_steps,
            "dt": dt,
            "input_strength_range": input_strength_range,
            "emergence_threshold": emergence_threshold,
            "level_spacing": level_spacing,
            "device": device,
        },
        "experiments_summary": [],
        "validation_checks": {},
        "statistics": {},
        "summary": {},
    }
    
    # 实验摘要
    for exp in all_experiments:
        validation_results["experiments_summary"].append({
            "seed": exp["seed"],
            "lambda_final": exp["lambda_final"],
            "lambda_max": exp["lambda_max"],
            "m_pre_final": exp["m_pre_final"],
            "m_pre_max": exp["m_pre_max"],
            "m_pre_min": exp["m_pre_min"],
        })
    
    # 验证条件1: M_pre(t) 非负性
    check1_result = validate_non_negativity(all_experiments)
    validation_results["validation_checks"]["non_negativity"] = check1_result
    
    # 验证条件2: 无主体性
    check2_result = validate_no_subject_dependency(all_experiments)
    validation_results["validation_checks"]["no_subject_dependency"] = check2_result
    
    # 验证条件3: 阈值效应
    check3_result = validate_threshold_effect(all_experiments, emergence_threshold)
    validation_results["validation_checks"]["threshold_effect"] = check3_result
    
    # 验证条件4: 涌现单调性
    check4_result = validate_emergence_monotonicity(all_experiments)
    validation_results["validation_checks"]["emergence_monotonicity"] = check4_result
    
    # 3. 统计分析
    print("\n" + "-" * 50)
    print("步骤3: M_pre(t) 序列统计和相关性分析")
    print("-" * 50)
    
    # M_pre 序列统计
    all_m_pre_final = [exp["m_pre_final"] for exp in all_experiments]
    all_m_pre_max = [exp["m_pre_max"] for exp in all_experiments]
    all_lambda_final = [exp["lambda_final"] for exp in all_experiments]
    all_lambda_max = [exp["lambda_max"] for exp in all_experiments]
    
    m_pre_stats = {
        "final_mean": np.mean(all_m_pre_final),
        "final_std": np.std(all_m_pre_final),
        "final_min": min(all_m_pre_final),
        "final_max": max(all_m_pre_final),
        "max_mean": np.mean(all_m_pre_max),
        "max_std": np.std(all_m_pre_max),
    }
    
    lambda_stats = {
        "final_mean": np.mean(all_lambda_final),
        "final_std": np.std(all_lambda_final),
        "final_min": min(all_lambda_final),
        "final_max": max(all_lambda_final),
        "max_mean": np.mean(all_lambda_max),
        "max_std": np.std(all_lambda_max),
    }
    
    print(f"\n  M_pre(t) 统计:")
    print(f"    最终值: mean={m_pre_stats['final_mean']:.4f}, std={m_pre_stats['final_std']:.4f}")
    print(f"    最终值范围: [{m_pre_stats['final_min']:.4f}, {m_pre_stats['final_max']:.4f}]")
    print(f"    最大值: mean={m_pre_stats['max_mean']:.4f}, std={m_pre_stats['max_std']:.4f}")
    
    print(f"\n  Λ(t) 统计:")
    print(f"    最终值: mean={lambda_stats['final_mean']:.2f}, std={lambda_stats['final_std']:.2f}")
    print(f"    最终值范围: [{lambda_stats['final_min']}, {lambda_stats['final_max']}]")
    print(f"    最大值: mean={lambda_stats['max_mean']:.2f}, std={lambda_stats['max_std']:.2f}")
    
    validation_results["statistics"]["m_pre"] = m_pre_stats
    validation_results["statistics"]["lambda"] = lambda_stats
    
    # 4. 综合判定
    print("\n" + "-" * 50)
    print("步骤4: 综合判定")
    print("-" * 50)
    
    all_checks_passed = all([
        check1_result["passed"],
        check2_result["passed"],
        check3_result["passed"],
        check4_result["passed"],
    ])
    
    passed_count = sum([
        check1_result["passed"],
        check2_result["passed"],
        check3_result["passed"],
        check4_result["passed"],
    ])
    
    validation_results["summary"] = {
        "overall_passed": all_checks_passed,
        "passed_count": passed_count,
        "total_checks": 4,
        "individual_results": {
            "non_negativity": check1_result["passed"],
            "no_subject_dependency": check2_result["passed"],
            "threshold_effect": check3_result["passed"],
            "emergence_monotonicity": check4_result["passed"],
        },
    }
    
    print(f"\n  验证结果汇总:")
    print(f"    非负性: {'✓' if check1_result['passed'] else '✗'}")
    print(f"    无主体性: {'✓' if check2_result['passed'] else '✗'}")
    print(f"    阈值效应: {'✓' if check3_result['passed'] else '✗'}")
    print(f"    涌现单调性: {'✓' if check4_result['passed'] else '✗'}")
    print(f"\n  总体结果: {'✓ 全部通过' if all_checks_passed else f'✗ {passed_count}/4 通过'}")
    
    # 5. 最终总结
    print("\n" + "=" * 70)
    if all_checks_passed:
        print("✓✓✓ P4 高阶意识命题验证全部通过")
        print("无主体元意识验证成功:")
        print("  - M_pre(t) 在所有情况下保持非负")
        print("  - M_pre(t) 不依赖特定主体，仅依赖状态变化率")
        print("  - 超过阈值后涌现效应显著")
        print("  - 涌现单调性成立")
    else:
        print("❌ P4 高阶意识命题验证未完全通过")
        print(f"  通过 {passed_count}/4 项验证")
        for check_name, check_result in validation_results["validation_checks"].items():
            if not check_result["passed"]:
                print(f"  - {check_name}: {check_result['message']}")
    print("=" * 70)
    
    return validation_results


def save_report(validation_results: Dict[str, Any], output_dir: str = "validation_results") -> str:
    """
    保存验证报告
    
    Args:
        validation_results: 验证结果字典
        output_dir: 输出目录
    
    Returns:
        报告文件路径
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_path = output_path / "p4_report.json"
    
    # 自定义 JSON 序列化函数，处理 numpy 和 torch 类型
    def custom_serializer(obj):
        """自定义序列化器"""
        if isinstance(obj, bool):
            return str(obj).lower()  # 将 bool 转换为字符串 "true"/"false"
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    # 递归转换所有 bool 值和 numpy 类型为可序列化类型
    def convert_bools(d):
        """递归转换字典中的 bool 值和 numpy 类型为可序列化类型"""
        if isinstance(d, dict):
            return {k: convert_bools(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [convert_bools(item) for item in d]
        elif isinstance(d, (bool, np.bool_)):
            return str(bool(d)).lower()  # 将 bool 和 numpy.bool_ 转换为字符串
        elif isinstance(d, (np.integer, np.int32, np.int64)):
            return int(d)  # numpy 整数转 Python int
        elif isinstance(d, (np.floating, np.float32, np.float64)):
            return float(d)  # numpy 浮点转 Python float
        elif isinstance(d, np.ndarray):
            return d.tolist()
        elif isinstance(d, torch.Tensor):
            return d.tolist()
        else:
            return d
    
    # 转换验证结果
    converted_results = convert_bools(validation_results)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(converted_results, f, indent=2, ensure_ascii=False, default=custom_serializer)
    
    print(f"\n验证报告已保存至: {report_path}")
    return str(report_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="P4 高阶意识命题验证")
    parser.add_argument("--num_experiments", type=int, default=10, 
                        help="实验数量（不同主体）")
    parser.add_argument("--num_steps", type=int, default=500, 
                        help="每个实验的运行步数")
    parser.add_argument("--dt", type=float, default=0.1, 
                        help="时间步长")
    parser.add_argument("--emergence_threshold", type=float, default=100.0, 
                        help="涌现阈值 M_0")
    parser.add_argument("--level_spacing", type=float, default=50.0, 
                        help="层级间距 ΔM")
    parser.add_argument("--device", type=str, default="cpu", 
                        help="计算设备")
    parser.add_argument("--output_dir", type=str, default="validation_results", 
                        help="输出目录")
    
    args = parser.parse_args()
    
    # 运行验证
    results = run_p4_validation(
        num_experiments=args.num_experiments,
        num_steps=args.num_steps,
        dt=args.dt,
        emergence_threshold=args.emergence_threshold,
        level_spacing=args.level_spacing,
        device=args.device
    )
    
    # 保存报告
    save_report(results, args.output_dir)