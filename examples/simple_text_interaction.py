"""
Chronos-Self 简单文本交互示例
=============================

演示如何使用 Chronos-Self 系统进行文本交互。

功能演示：
1. 系统初始化
2. 文本输入处理
3. 多轮对话交互
4. 状态监控
5. 系统生命周期管理

运行方式：
    python examples/simple_text_interaction.py

环境要求：
- Python 3.8+
- PyTorch 1.12+
- GPU 推荐（CPU 也可运行）

配置选项：
- 默认使用最小配置（D_f=256, D_s=128）
- 可通过命令行参数调整

作者: Chronos-Self Team
"""

import sys
import time
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from chronos_core.integration.system_integration import (
    ChronosSystem,
    ChronosSystemConfig,
    ChronosSystemController,
    SystemStatus,
    SystemResponse,
)
from chronos_core.utils.config import ChronosConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SimpleInteraction")


class SimpleTextInteraction:
    """
    简单文本交互演示
    
    提供命令行交互界面，演示 Chronos-Self 系统的基本功能。
    """
    
    def __init__(
        self,
        fast_dim: int = 256,
        slow_dim: int = 128,
        device: str = "auto"
    ):
        """
        初始化交互系统
        
        Args:
            fast_dim: 快变量维度（使用较小值便于演示）
            slow_dim: 慢变量维度
            device: 计算设备 ('auto', 'cuda', 'cpu')
        """
        # 设备选择
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # 创建配置
        self.config = self._create_config(fast_dim, slow_dim)
        
        # 系统组件
        self.system: Optional[ChronosSystem] = None
        self.controller: Optional[ChronosSystemController] = None
        
        # 交互历史
        self.history: list[Dict[str, Any]] = []
        
        # 运行状态
        self._running = False
        
        logger.info(
            f"SimpleTextInteraction initialized: "
            f"D_f={fast_dim}, D_s={slow_dim}, device={self.device}"
        )
    
    def _create_config(self, fast_dim: int, slow_dim: int) -> ChronosConfig:
        """创建配置"""
        config = ChronosConfig()
        
        # 设置维度
        config.dim.fast_variable_dim = fast_dim
        config.dim.slow_variable_dim = slow_dim
        
        # 设置相关维度
        config.dim.semantic_dim = min(fast_dim // 2, 128)
        config.dim.physical_dim = min(fast_dim // 2, 128)
        
        # 设备设置
        config.device = self.device
        
        return config
    
    def start(self) -> None:
        """启动系统"""
        if self._running:
            logger.warning("系统已在运行")
            return
        
        logger.info("=" * 60)
        logger.info("启动 Chronos-Self 系统...")
        logger.info("=" * 60)
        
        # 创建系统配置
        system_config = ChronosSystemConfig(
            enable_semantic_encoder=False,  # 简化演示
            enable_logical_encoder=False,
            enable_fusion_module=False,
            enable_meta_cognitive=True,
            enable_reflection=True,
            enable_working_memory=True,
            enable_training_system=False,
            enable_validation_system=False,
        )
        
        # 创建系统
        self.system = ChronosSystem(
            config=system_config,
            global_config=self.config,
            device=self.device
        )
        
        # 初始化
        logger.info("初始化系统组件...")
        self.system.initialize()
        
        # 创建控制器
        self.controller = ChronosSystemController(self.system)
        self.controller.start()
        
        self._running = True
        
        logger.info("系统启动完成！")
        logger.info("=" * 60)
    
    def stop(self) -> None:
        """停止系统"""
        if not self._running:
            return
        
        logger.info("停止系统...")
        
        if self.controller:
            self.controller.stop()
        
        self._running = False
        
        logger.info("系统已停止")
    
    def process_input(self, text: str) -> SystemResponse:
        """
        处理输入
        
        Args:
            text: 输入文本
        
        Returns:
            系统响应
        """
        if not self._running:
            raise ValueError("系统未启动")
        
        # 处理输入
        start_time = time.time()
        response = self.controller.process_input(text=text)
        processing_time = time.time() - start_time
        
        # 记录历史
        self.history.append({
            "text": text,
            "response": response.to_dict(),
            "processing_time_ms": processing_time * 1000,
            "timestamp": time.time(),
        })
        
        return response
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        if not self._running:
            return {"status": "stopped"}
        
        system_state = self.controller.get_system_state()
        statistics = self.system.get_statistics()
        
        return {
            "status": system_state.status.value,
            "total_steps": system_state.total_steps,
            "simulated_time": system_state.simulated_time,
            "avg_step_time_ms": system_state.avg_step_time_ms,
            "is_stable": system_state.is_stable,
            "current_self_state": {
                "E_fast_norm": statistics["current_self_state"]["E_fast_norm"],
                "E_slow_norm": statistics["current_self_state"]["E_slow_norm"],
                "timestamp": statistics["current_self_state"]["timestamp"],
            },
        }
    
    def run_interactive_session(self) -> None:
        """
        运行交互式会话
        
        提供命令行交互界面。
        """
        self.start()
        
        print("\n" + "=" * 60)
        print("Chronos-Self 交互式演示")
        print("=" * 60)
        print("输入文本与系统交互，输入 'exit' 或 'quit' 退出")
        print("输入 'status' 查看系统状态")
        print("输入 'help' 查看帮助")
        print("=" * 60 + "\n")
        
        while self._running:
            try:
                # 获取输入
                user_input = input("用户: ").strip()
                
                if not user_input:
                    continue
                
                # 处理特殊命令
                if user_input.lower() in ["exit", "quit"]:
                    print("\n退出交互会话...")
                    self.stop()
                    break
                
                if user_input.lower() == "status":
                    status = self.get_status()
                    print("\n系统状态:")
                    print(f"  - 状态: {status['status']}")
                    print(f"  - 总步数: {status['total_steps']}")
                    print(f"  - 模拟时间: {status['simulated_time']:.2f}s")
                    print(f"  - 平均步时: {status['avg_step_time_ms']:.2f}ms")
                    print(f"  - 快变量范数: {status['current_self_state']['E_fast_norm']:.4f}")
                    print(f"  - 慢变量范数: {status['current_self_state']['E_slow_norm']:.4f}")
                    print()
                    continue
                
                if user_input.lower() == "help":
                    print("\n帮助:")
                    print("  - 输入任意文本与系统交互")
                    print("  - 'status' - 查看系统状态")
                    print("  - 'history' - 查看交互历史")
                    print("  - 'exit' 或 'quit' - 退出会话")
                    print()
                    continue
                
                if user_input.lower() == "history":
                    print("\n交互历史:")
                    for i, entry in enumerate(self.history[-5:]):
                        print(f"  [{i+1}] 用户: {entry['text']}")
                        print(f"      处理时间: {entry['processing_time_ms']:.2f}ms")
                    print()
                    continue
                
                # 处理输入
                response = self.process_input(user_input)
                
                # 显示响应
                print(f"\n系统: {response.content}")
                print(f"  [处理时间: {response.processing_time_ms:.2f}ms]")
                print(f"  [意图类型: {response.intent_type}]")
                print(f"  [置信度: {response.confidence:.4f}]")
                print()
                
            except KeyboardInterrupt:
                print("\n\n中断信号，退出...")
                self.stop()
                break
            
            except Exception as e:
                logger.error(f"处理错误: {e}")
                print(f"\n错误: {e}\n")
        
        print("\n" + "=" * 60)
        print("会话结束")
        print(f"总交互数: {len(self.history)}")
        print("=" * 60 + "\n")
    
    def run_demo_session(self, num_interactions: int = 5) -> None:
        """
        运行演示会话
        
        自动处理预设输入，演示系统功能。
        
        Args:
            num_interactions: 交互次数
        """
        self.start()
        
        # 预设输入
        demo_inputs = [
            "你好，请介绍一下你自己",
            "今天天气怎么样？",
            "我有一个问题想问你",
            "你能帮我做些什么？",
            "再见，谢谢你的帮助",
        ]
        
        print("\n" + "=" * 60)
        print("Chronos-Self 演示会话")
        print("=" * 60)
        
        for i, text in enumerate(demo_inputs[:num_interactions]):
            print(f"\n[交互 {i+1}/{num_interactions}]")
            print(f"用户: {text}")
            
            response = self.process_input(text)
            
            print(f"系统: {response.content}")
            print(f"  - 处理时间: {response.processing_time_ms:.2f}ms")
            print(f"  - 意图类型: {response.intent_type}")
            
            # 获取状态
            status = self.get_status()
            print(f"  - 状态范数: {status['current_self_state']['E_fast_norm']:.4f}")
        
        print("\n" + "=" * 60)
        print("演示完成")
        
        # 显示统计
        total_time = sum(h["processing_time_ms"] for h in self.history)
        avg_time = total_time / len(self.history) if self.history else 0
        
        print(f"总交互数: {len(self.history)}")
        print(f"总处理时间: {total_time:.2f}ms")
        print(f"平均处理时间: {avg_time:.2f}ms")
        print("=" * 60 + "\n")
        
        self.stop()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Chronos-Self 简单文本交互")
    parser.add_argument("--demo", action="store_true", help="运行演示模式")
    parser.add_argument("--device", type=str, default="auto", help="计算设备")
    parser.add_argument("--fast_dim", type=int, default=256, help="快变量维度")
    parser.add_argument("--slow_dim", type=int, default=128, help="慢变量维度")
    parser.add_argument("--num_interactions", type=int, default=5, help="演示交互次数")
    
    args = parser.parse_args()
    
    # 创建交互实例
    interaction = SimpleTextInteraction(
        fast_dim=args.fast_dim,
        slow_dim=args.slow_dim,
        device=args.device
    )
    
    # 运行
    if args.demo:
        interaction.run_demo_session(args.num_interactions)
    else:
        interaction.run_interactive_session()


if __name__ == "__main__":
    main()