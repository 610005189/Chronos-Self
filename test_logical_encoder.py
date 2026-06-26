"""
Test script for LogicalEncoder and its components.

This script validates the implementation of:
- Structured State Space Model (SSM)
- Proprioceptive Flow Encoder
- World Flow Encoder
- Causal Encoder (Physical Constraints & Causal Chains)
- LogicalEncoder (Main integration class)
"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chronos_core.representation import (
    LogicalEncoder,
    create_logical_encoder,
    StateSpaceModel,
    SSMBlock,
    StackedSSM,
    ProprioceptiveEncoder,
    ProprioceptiveState,
    WorldEncoder,
    WorldState,
    CausalEncoder,
    PhysicalConstraints,
    CausalChain,
    check_numerical_stability,
)
from chronos_core.utils.config import EncoderConfig, get_config


def test_ssm_components():
    """Test SSM core components."""
    print("\n" + "=" * 50)
    print("Testing SSM Components")
    print("=" * 50)

    # Test StateSpaceModel
    print("\n1. Testing StateSpaceModel:")
    ssm = StateSpaceModel(
        input_dim=64,
        state_dim=128,
        output_dim=64,
    )

    # Create test input
    batch_size = 2
    seq_len = 10
    input_dim = 64
    x = torch.randn(batch_size, seq_len, input_dim)

    # Forward pass
    output, hidden_state = ssm(x)

    # Validate output
    assert output.shape == (batch_size, seq_len, 64), f"Output shape mismatch: {output.shape}"
    assert hidden_state.shape == (batch_size, 128), f"Hidden state shape mismatch: {hidden_state.shape}"

    is_stable, msg = check_numerical_stability(output, "SSM_output")
    print(f"   ✓ Output shape: {output.shape}")
    print(f"   ✓ Hidden state shape: {hidden_state.shape}")
    print(f"   ✓ Numerical stability: {is_stable}")

    # Test SSMBlock
    print("\n2. Testing SSMBlock:")
    ssm_block = SSMBlock(
        input_dim=64,
        state_dim=128,
        expansion_factor=2,
    )

    output, hidden_state = ssm_block(x)
    assert output.shape == (batch_size, seq_len, 64), f"Output shape mismatch: {output.shape}"
    print(f"   ✓ Output shape: {output.shape}")

    # Test StackedSSM
    print("\n3. Testing StackedSSM:")
    stacked_ssm = StackedSSM(
        input_dim=64,
        state_dim=128,
        num_layers=4,
    )

    output, hidden_states = stacked_ssm(x)
    assert output.shape == (batch_size, seq_len, 64), f"Output shape mismatch: {output.shape}"
    assert hidden_states.shape[0] == 4, f"Hidden states layers mismatch: {hidden_states.shape}"
    print(f"   ✓ Output shape: {output.shape}")
    print(f"   ✓ Hidden states shape: {hidden_states.shape}")

    print("\n✓ All SSM components tests passed!")


def test_proprioceptive_encoder():
    """Test Proprioceptive Encoder."""
    print("\n" + "=" * 50)
    print("Testing Proprioceptive Encoder")
    print("=" * 50)

    config = EncoderConfig()
    encoder = ProprioceptiveEncoder(config)

    # Test with ProprioceptiveState object
    print("\n1. Testing with ProprioceptiveState:")
    proprio_state = ProprioceptiveState(
        position=torch.tensor([1.0, 2.0, 3.0]),
        orientation=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        velocity=torch.tensor([0.1, 0.2, 0.3]),
        energy_level=0.8,
        battery_status=0.9,
        cpu_usage=0.4,
        memory_usage=0.6,
        gpu_usage=0.2,
    )

    # Encode single state
    X_proprio = encoder.encode_single(proprio_state)

    assert X_proprio.shape == (256,), f"Output shape mismatch: {X_proprio.shape}"
    print(f"   ✓ Output shape: {X_proprio.shape}")
    print(f"   ✓ Output norm: {torch.norm(X_proprio).item():.4f}")

    # Test with tensor input
    print("\n2. Testing with tensor input:")
    proprio_tensor = proprio_state.to_tensor().unsqueeze(0).unsqueeze(0)
    X_proprio_seq, hidden_state, metadata = encoder(proprioceptive_tensor=proprio_tensor)

    assert X_proprio_seq.shape == (1, 1, 256), f"Output shape mismatch: {X_proprio_seq.shape}"
    print(f"   ✓ Output shape: {X_proprio_seq.shape}")
    print(f"   ✓ Metadata keys: {list(metadata.keys())}")

    print("\n✓ Proprioceptive encoder tests passed!")


def test_world_encoder():
    """Test World Encoder."""
    print("\n" + "=" * 50)
    print("Testing World Encoder")
    print("=" * 50)

    config = EncoderConfig()
    encoder = WorldEncoder(config)

    # Test with WorldState object
    print("\n1. Testing with WorldState:")
    world_state = WorldState(
        objects=[
            {
                'position': [1.0, 2.0, 3.0],
                'type': 1,
                'attributes': {'size': 0.8, 'mass': 1.5, 'color': 0.6, 'material': 0.3}
            },
            {
                'position': [4.0, 5.0, 6.0],
                'type': 2,
                'attributes': {'size': 1.2, 'mass': 2.0, 'color': 0.4, 'material': 0.7}
            }
        ],
        temperature=0.6,
        light_level=0.8,
        sound_level=0.3,
        humidity=0.5,
        other_agents=[
            {
                'position': [7.0, 8.0, 9.0],
                'state': {'energy': 0.7, 'activity': 0.5, 'orientation': 0.3, 'speed': 0.4, 'intent': 0.6}
            }
        ]
    )

    # Encode single state
    X_world = encoder.encode_single(world_state)

    assert X_world.shape == (256,), f"Output shape mismatch: {X_world.shape}"
    print(f"   ✓ Output shape: {X_world.shape}")
    print(f"   ✓ Output norm: {torch.norm(X_world).item():.4f}")

    # Test with tensor input
    print("\n2. Testing with tensor input:")
    world_tensor = world_state.to_tensor(encoder.max_objects, encoder.max_agents).unsqueeze(0).unsqueeze(0)
    X_world_seq, hidden_state, metadata = encoder(world_tensor=world_tensor)

    assert X_world_seq.shape == (1, 1, 256), f"Output shape mismatch: {X_world_seq.shape}"
    print(f"   ✓ Output shape: {X_world_seq.shape}")
    print(f"   ✓ Metadata: num_objects={metadata['num_objects']}, num_agents={metadata['num_agents']}")

    print("\n✓ World encoder tests passed!")


def test_causal_encoder():
    """Test Causal Encoder."""
    print("\n" + "=" * 50)
    print("Testing Causal Encoder")
    print("=" * 50)

    config = EncoderConfig()
    encoder = CausalEncoder(config)

    # Test with PhysicalConstraints and CausalChain
    print("\n1. Testing with PhysicalConstraints and CausalChain:")
    physical_constraints = PhysicalConstraints(
        energy_conservation=1.0,
        gravity_strength=0.5,
        friction_coefficient=0.3,
    )

    causal_chain = CausalChain(
        events=[
            {'timestamp': 1.0, 'type': 1, 'intensity': 0.5, 'duration': 0.2, 'location': 0.8, 'effect_strength': 0.6},
            {'timestamp': 2.0, 'type': 2, 'intensity': 0.7, 'duration': 0.3, 'location': 0.4, 'effect_strength': 0.8},
        ],
        causal_relations=[
            {'cause_id': 1, 'effect_id': 2, 'strength': 0.8, 'delay': 0.5},
        ]
    )

    # Encode single
    X_causal = encoder.encode_single(physical_constraints, causal_chain)

    assert X_causal.shape == (128,), f"Output shape mismatch: {X_causal.shape}"
    print(f"   ✓ Output shape: {X_causal.shape}")
    print(f"   ✓ Output norm: {torch.norm(X_causal).item():.4f}")

    # Test with tensor input
    print("\n2. Testing with tensor input:")
    constraints_tensor = physical_constraints.to_tensor(encoder.max_constraints)
    causal_tensor = causal_chain.to_tensor(encoder.max_events, encoder.max_relations)
    input_tensor = torch.cat([constraints_tensor, causal_tensor]).unsqueeze(0).unsqueeze(0)

    X_causal_seq, metadata = encoder(input_tensor=input_tensor)

    assert X_causal_seq.shape == (1, 1, 128), f"Output shape mismatch: {X_causal_seq.shape}"
    print(f"   ✓ Output shape: {X_causal_seq.shape}")
    print(f"   ✓ Metadata keys: {list(metadata.keys())}")

    print("\n✓ Causal encoder tests passed!")


def test_logical_encoder():
    """Test LogicalEncoder (main integration class)."""
    print("\n" + "=" * 50)
    print("Testing LogicalEncoder")
    print("=" * 50)

    config = EncoderConfig()
    encoder = LogicalEncoder(config)

    print(f"\nEncoder dimensions:")
    dims = encoder.get_output_dimensions()
    for name, dim in dims.items():
        print(f"   {name}: {dim}")

    # Test single encoding
    print("\n1. Testing single encoding:")
    proprio_state = ProprioceptiveState(
        position=torch.tensor([1.0, 2.0, 3.0]),
        orientation=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        velocity=torch.tensor([0.1, 0.2, 0.3]),
        energy_level=0.8,
        battery_status=0.9,
    )

    world_state = WorldState(
        objects=[
            {'position': [1.0, 2.0, 3.0], 'type': 1, 'attributes': {'size': 0.8, 'mass': 1.5}}
        ],
        temperature=0.6,
    )

    physical_constraints = PhysicalConstraints(energy_conservation=1.0)
    causal_chain = CausalChain(events=[{'timestamp': 1.0, 'type': 1, 'intensity': 0.5}])

    X_log, X_proprio, X_world, X_causal = encoder.encode_single(
        proprio_state, world_state, physical_constraints, causal_chain
    )

    # Validate dimensions
    assert X_log.shape == (512,), f"X_log shape mismatch: {X_log.shape}"
    assert X_proprio.shape == (256,), f"X_proprio shape mismatch: {X_proprio.shape}"
    assert X_world.shape == (256,), f"X_world shape mismatch: {X_world.shape}"
    assert X_causal.shape == (128,), f"X_causal shape mismatch: {X_causal.shape}"

    print(f"   ✓ X_log shape: {X_log.shape}, norm: {torch.norm(X_log).item():.4f}")
    print(f"   ✓ X_proprio shape: {X_proprio.shape}, norm: {torch.norm(X_proprio).item():.4f}")
    print(f"   ✓ X_world shape: {X_world.shape}, norm: {torch.norm(X_world).item():.4f}")
    print(f"   ✓ X_causal shape: {X_causal.shape}, norm: {torch.norm(X_causal).item():.4f}")

    # Validate output
    is_valid, errors = encoder.validate_output(X_log, X_proprio, X_world, X_causal)
    assert is_valid, f"Validation failed: {errors}"
    print(f"   ✓ Output validation: passed")

    # Test sequence encoding
    print("\n2. Testing sequence encoding:")
    batch_size = 2
    seq_len = 5

    proprio_tensor = proprio_state.to_tensor().unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1)
    world_tensor = world_state.to_tensor(encoder.world_encoder.max_objects, encoder.world_encoder.max_agents).unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1)

    # Create causal tensor with matching dimensions
    constraints_tensor = physical_constraints.to_tensor(encoder.causal_encoder.max_constraints)
    causal_tensor = causal_chain.to_tensor(encoder.causal_encoder.max_events, encoder.causal_encoder.max_relations)
    causal_input_tensor = torch.cat([constraints_tensor, causal_tensor]).unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1)

    X_log_seq, X_proprio_seq, X_world_seq, X_causal_seq, metadata = encoder(
        proprioceptive_tensor=proprio_tensor,
        world_tensor=world_tensor,
        causal_tensor=causal_input_tensor,
    )

    assert X_log_seq.shape == (batch_size, seq_len, 512), f"X_log shape mismatch: {X_log_seq.shape}"
    print(f"   ✓ X_log_seq shape: {X_log_seq.shape}")
    print(f"   ✓ Metadata keys: {list(metadata.keys())}")

    print("\n✓ LogicalEncoder tests passed!")


def test_integration():
    """Test integration with existing code."""
    print("\n" + "=" * 50)
    print("Testing Integration")
    print("=" * 50)

    # Test with ChronosConfig
    print("\n1. Testing with ChronosConfig:")
    config = get_config()
    encoder = create_logical_encoder(config.encoder)

    print(f"   ✓ Created encoder with config")
    print(f"   ✓ Encoder dimensions: {encoder.get_output_dimensions()}")

    # Test import
    print("\n2. Testing module imports:")
    from chronos_core.representation import (
        LogicalEncoder,
        ProprioceptiveState,
        WorldState,
        PhysicalConstraints,
        CausalChain,
    )
    print("   ✓ All imports successful")

    print("\n✓ Integration tests passed!")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("LogicalEncoder Implementation Tests")
    print("=" * 80)

    try:
        test_ssm_components()
        test_proprioceptive_encoder()
        test_world_encoder()
        test_causal_encoder()
        test_logical_encoder()
        test_integration()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nImplementation Summary:")
        print("   ✓ SubTask 4.1: SSM架构实现完成")
        print("   ✓ SubTask 4.2: 本体感觉流编码完成")
        print("   ✓ SubTask 4.3: 外部世界流编码完成")
        print("   ✓ SubTask 4.4: 物理约束和因果链条编码完成")
        print("   ✓ LogicalEncoder主类整合完成")
        print("   ✓ 完整类型注解已添加")
        print("   ✓ 数值稳定性检查已实现")
        print("\nFiles created:")
        print("   - chronos_core/representation/ssm.py")
        print("   - chronos_core/representation/proprioceptive_encoder.py")
        print("   - chronos_core/representation/world_encoder.py")
        print("   - chronos_core/representation/causal_encoder.py")
        print("   - chronos_core/representation/logical_encoder.py")
        print("   - Updated: chronos_core/representation/__init__.py")
        print("=" * 80 + "\n")

        return True

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)