"""
P3 元意识命题验证 - 有限自指终止验证
========================================

验证核心命题：
- Λ(t) 在有限时间内收敛到某个值 ≤ L_max = 3
- 涌现单调性：M_pre增大时深度不减小
- M_pre(t) 非负

运行配置：
- 启用元意识引擎（enable_meta_consciousness=True）
- 运行足够长时间（500-1000步）
- 观察 Λ(t) 和 M_pre(t) 演化时间序列
"""

import sys
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos_core.utils.config import (
    ChronosConfig,
    DimensionalityConfig,
    MetaCognitiveConfig,
)
from chronos_core.core.meta_cognitive.dynamic_layers import DynamicMetaCognitiveLayers


def run_p3_validation(
    num_steps: int = 800,
    convergence_threshold: int = 10,
    dt: float = 0.1,
    input_strength: float = 0.2,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    运行 P3 元意识命题验证
    
    Args:
        num_steps: 总运行步数
        convergence_threshold: 收敛判定阈值（连续N步Λ不变视为收敛）
        dt: 时间步长
        input_strength: 输入扰动强度
        device: 计算设备
    
    Returns:
        验证结果字典
    """
    print("=" * 70)
    print("P3 元意识命题验证 - 有限自指终止")
    print("=" * 70)
    print(f"运行配置: num_steps={num_steps}, dt={dt}, device={device}")
    
    # 1. 创建配置并启用元意识引擎
    print("\n" + "-" * 50)
    print("步骤1: 创建配置并启用元意识引擎")
    print("-" * 50)
    
    dim_config = DimensionalityConfig()
    
    # 元意识配置
    meta_config = MetaCognitiveConfig(
        enable_meta_consciousness=True,  # 关键：启用元意识引擎
        meta_consciousness_window_time=1.0,
        meta_consciousness_emergence_threshold=100.0,
        meta_consciousness_level_spacing=50.0,
        meta_consciousness_awareness_gate_threshold=0.5,
    )
    
    print(f"  enable_meta_consciousness: {meta_config.enable_meta_consciousness}")
    print(f"  emergence_threshold (M_0): {meta_config.meta_consciousness_emergence_threshold}")
    print(f"  level_spacing (ΔM): {meta_config.meta_consciousness_level_spacing}")
    print(f"  awareness_gate_threshold (G_th): {meta_config.meta_consciousness_awareness_gate_threshold}")
    
    # 2. 创建 DynamicMetaCognitiveLayers 实例
    print("\n" + "-" * 50)
    print("步骤2: 创建 DynamicMetaCognitiveLayers 实例")
    print("-" * 50)
    
    # 层维度配置：[L0_dim, L1_dim, L2_dim]
    layer_dims = [
        dim_config.fast_variable_dim,
        dim_config.slow_variable_dim,
        max(dim_config.slow_variable_dim // 2, 32)
    ]
    
    dynamic_layers = DynamicMetaCognitiveLayers(
        layer_dims=layer_dims,
        window_time=meta_config.meta_consciousness_window_time,
        emergence_threshold=meta_config.meta_consciousness_emergence_threshold,
        level_spacing=meta_config.meta_consciousness_level_spacing,
        awareness_gate_threshold=meta_config.meta_consciousness_awareness_gate_threshold,
        device=device
    )
    
    print(f"  DynamicMetaCognitiveLayers 创建成功")
    print(f"  层数: {dynamic_layers.num_layers} (L_max = {dynamic_layers.num_layers - 1})")
    print(f"  层维度: {layer_dims}")
    print(f"  初始深度 Λ: {dynamic_layers.current_depth}")
    print(f"  初始 M_pre: {dynamic_layers.meta_field.get_value():.6f}")
    
    L_max = dynamic_layers.num_layers - 1  # L_max = 2（索引从0开始）
    
    # 3. 运行演化并记录时间序列
    print("\n" + "-" * 50)
    print("步骤3: 运行演化并记录 Λ(t), M_pre(t) 时间序列")
    print("-" * 50)
    
    # 时间序列记录
    lambda_history: List[int] = []
    m_pre_history: List[float] = []
    layer_active_history: List[List[bool]] = []
    convergence_step: int = -1  # 收敛发生的步数
    
    prev_state = dynamic_layers.layer_states[0].clone()
    
    # 运行演化
    for step in range(num_steps):
        # 生成输入扰动
        z_input = prev_state + input_strength * torch.randn(layer_dims[0])
        
        # 执行一步演化
        result = dynamic_layers.step(z_input, dt)
        prev_state = result["layer_states"][0].clone()
        
        # 记录时间序列
        lambda_history.append(result["current_depth"])
        m_pre_history.append(result["m_pre"])
        layer_active_history.append(result["layer_active"].copy())
        
        # 每100步打印进度
        if (step + 1) % 100 == 0 or step < 10:
            print(
                f"  Step {step + 1}: "
                f"Λ={result['current_depth']}, "
                f"M_pre={result['m_pre']:.4f}, "
                f"Active={result['layer_active']}"
            )
        
        # 检测收敛（连续 convergence_threshold 步 Λ 不变）
        if convergence_step < 0 and len(lambda_history) >= convergence_threshold:
            recent = lambda_history[-convergence_threshold:]
            if all(d == recent[0] for d in recent):
                convergence_step = step + 1
                print(f"  ★ 收敛检测: Λ 收敛于步 {convergence_step}, 值 = {recent[0]}")
    
    # 4. 验证各项条件
    print("\n" + "-" * 50)
    print("步骤4: 验证 P3 命题各条件")
    print("-" * 50)
    
    # 验证结果字典
    validation_results: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_steps": num_steps,
            "convergence_threshold": convergence_threshold,
            "dt": dt,
            "input_strength": input_strength,
            "device": device,
            "L_max": L_max,
            "layer_dims": layer_dims,
            "emergence_threshold": meta_config.meta_consciousness_emergence_threshold,
            "level_spacing": meta_config.meta_consciousness_level_spacing,
        },
        "time_series": {
            "lambda_history": lambda_history,
            "m_pre_history": m_pre_history,
            "convergence_step": convergence_step,
        },
        "validation_checks": {},
        "summary": {},
    }
    
    # 验证条件1: Λ(t) 在有限时间内收敛
    check1_passed = convergence_step > 0
    check1_message = (
        f"Λ(t) 在步 {convergence_step} 收敛" if check1_passed 
        else f"Λ(t) 未在 {num_steps} 步内收敛"
    )
    validation_results["validation_checks"]["finite_time_convergence"] = {
        "passed": check1_passed,
        "message": check1_message,
        "convergence_step": convergence_step,
        "convergence_value": lambda_history[-1] if convergence_step > 0 else None,
    }
    print(f"  [条件1] Λ(t) 有限时间收敛: {'✓ 通过' if check1_passed else '✗ 未通过'}")
    print(f"          {check1_message}")
    
    # 验证条件2: Λ(t) ≤ L_max = 3
    lambda_max = max(lambda_history)
    lambda_final = lambda_history[-1]
    check2_passed = lambda_max <= L_max and lambda_final <= L_max
    check2_message = (
        f"max(Λ)={lambda_max}, final(Λ)={lambda_final} ≤ L_max={L_max}" 
        if check2_passed 
        else f"Λ 超出 L_max={L_max}: max={lambda_max}, final={lambda_final}"
    )
    validation_results["validation_checks"]["lambda_upper_bound"] = {
        "passed": check2_passed,
        "message": check2_message,
        "lambda_max": lambda_max,
        "lambda_final": lambda_final,
        "L_max": L_max,
    }
    print(f"  [条件2] Λ(t) ≤ L_max: {'✓ 通过' if check2_passed else '✗ 未通过'}")
    print(f"          {check2_message}")
    
    # 验证条件3: 涌现单调性（M_pre增大时深度不减小）
    # 检测是否存在 M_pre 增加但 Λ 减小的情况
    monotonicity_violations: List[Tuple[int, int, float, float]] = []
    for i in range(1, len(lambda_history)):
        m_pre_prev = m_pre_history[i - 1]
        m_pre_curr = m_pre_history[i]
        lambda_prev = lambda_history[i - 1]
        lambda_curr = lambda_history[i]
        
        # 如果 M_pre 增加但 Λ 减小，记录违规
        if m_pre_curr > m_pre_prev and lambda_curr < lambda_prev:
            monotonicity_violations.append((i, lambda_curr - lambda_prev, m_pre_curr - m_pre_prev))
    
    check3_passed = len(monotonicity_violations) == 0
    check3_message = (
        "涌现单调性成立：M_pre增加时Λ不减小" 
        if check3_passed 
        else f"涌现单调性违反 {len(monotonicity_violations)} 次"
    )
    validation_results["validation_checks"]["emergence_monotonicity"] = {
        "passed": check3_passed,
        "message": check3_message,
        "violations": monotonicity_violations,
        "violation_count": len(monotonicity_violations),
    }
    print(f"  [条件3] 涌现单调性: {'✓ 通过' if check3_passed else '✗ 未通过'}")
    print(f"          {check3_message}")
    
    # 验证条件4: M_pre(t) 非负
    m_pre_min = min(m_pre_history)
    m_pre_final = m_pre_history[-1]
    check4_passed = m_pre_min >= 0.0
    check4_message = (
        f"min(M_pre)={m_pre_min:.6f} ≥ 0, final(M_pre)={m_pre_final:.4f}"
        if check4_passed
        else f"M_pre 存在负值: min={m_pre_min:.6f}"
    )
    validation_results["validation_checks"]["m_pre_nonnegative"] = {
        "passed": check4_passed,
        "message": check4_message,
        "m_pre_min": m_pre_min,
        "m_pre_max": max(m_pre_history),
        "m_pre_final": m_pre_final,
    }
    print(f"  [条件4] M_pre(t) 非负: {'✓ 通过' if check4_passed else '✗ 未通过'}")
    print(f"          {check4_message}")
    
    # 5. 计算时间序列统计
    print("\n" + "-" * 50)
    print("步骤5: 时间序列统计")
    print("-" * 50)
    
    # Λ(t) 统计
    lambda_stats = {
        "min": min(lambda_history),
        "max": max(lambda_history),
        "final": lambda_history[-1],
        "unique_values": sorted(set(lambda_history)),
        "value_counts": {
            str(v): lambda_history.count(v) for v in set(lambda_history)
        },
    }
    
    # M_pre(t) 统计
    m_pre_stats = {
        "min": min(m_pre_history),
        "max": max(m_pre_history),
        "final": m_pre_history[-1],
        "mean": sum(m_pre_history) / len(m_pre_history),
        "trend": "increasing" if m_pre_history[-1] > m_pre_history[0] else "stable/decreasing",
    }
    
    # 层激活统计
    activation_stats = {
        "final_layer_active": layer_active_history[-1],
        "max_layers_active": max(
            sum(1 for a in layer_active_history[i]) 
            for i in range(len(layer_active_history))
        ),
    }
    
    validation_results["time_series_stats"] = {
        "lambda_stats": lambda_stats,
        "m_pre_stats": m_pre_stats,
        "activation_stats": activation_stats,
    }
    
    print(f"  Λ(t) 统计:")
    print(f"    - 最小值: {lambda_stats['min']}")
    print(f"    - 最大值: {lambda_stats['max']}")
    print(f"    - 最终值: {lambda_stats['final']}")
    print(f"    - 不同值: {lambda_stats['unique_values']}")
    print(f"  M_pre(t) 统计:")
    print(f"    - 最小值: {m_pre_stats['min']:.4f}")
    print(f"    - 最大值: {m_pre_stats['max']:.4f}")
    print(f"    - 最终值: {m_pre_stats['final']:.4f}")
    print(f"    - 平均值: {m_pre_stats['mean']:.4f}")
    print(f"    - 趋势: {m_pre_stats['trend']}")
    print(f"  层激活统计:")
    print(f"    - 最终激活状态: {activation_stats['final_layer_active']}")
    print(f"    - 最大激活层数: {activation_stats['max_layers_active']}")
    
    # 6. 总结
    all_passed = check1_passed and check2_passed and check3_passed and check4_passed
    
    validation_results["summary"] = {
        "all_checks_passed": all_passed,
        "passed_count": sum([
            check1_passed, check2_passed, check3_passed, check4_passed
        ]),
        "total_checks": 4,
        "proposition_status": "验证通过" if all_passed else "验证未通过",
        "final_lambda": lambda_final,
        "final_m_pre": m_pre_final,
    }
    
    print("\n" + "=" * 70)
    print("验证总结")
    print("=" * 70)
    print(f"  总体结果: {'✓ 所有条件通过' if all_passed else '✗ 存在未通过条件'}")
    print(f"  通过条件数: {validation_results['summary']['passed_count']}/4")
    print(f"  P3 命题状态: {validation_results['summary']['proposition_status']}")
    print(f"  最终 Λ: {lambda_final}")
    print(f"  最终 M_pre: {m_pre_final:.4f}")
    print("=" * 70)
    
    return validation_results


def main():
    """主函数：运行验证并保存报告"""
    # 运行验证
    results = run_p3_validation(
        num_steps=800,
        convergence_threshold=10,
        dt=0.1,
        input_strength=0.2,
        device="cpu"
    )
    
    # 保存报告到 JSON 文件
    output_path = Path(__file__).parent.parent / "validation_results" / "p3_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n报告已保存至: {output_path}")
    
    return results


if __name__ == "__main__":
    main()