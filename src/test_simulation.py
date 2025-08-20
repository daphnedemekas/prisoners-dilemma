#!/usr/bin/env python3
"""
Test script to verify the refactored simulation code works correctly.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from models.agent import construct_agents, DualAgentSimulation
from models.network import NetworkSimulation
from visualization.plots import SimulationVisualizer


def test_basic_simulation():
    """Test basic dual-agent simulation."""
    print("Testing basic simulation...")
    
    # Create agents
    agent_1, agent_2, D = construct_agents(
        lr_pB_1=0.1, 
        lr_pB_2=0.1, 
        factors_to_learn="all",
        action_selection="stochastic",
        alpha_1=1.0,
        alpha_2=1.0
    )
    
    # Create simulation
    sim = DualAgentSimulation(agent_1, agent_2)
    
    # Run short simulation
    results = sim.run_simulation(T=50, initial_obs_1=[0], initial_obs_2=[0])
    
    # Check results structure
    assert 'actions' in results
    assert 'transitions' in results
    assert 'policies' in results
    assert 'states' in results
    
    assert results['actions'].shape == (50, 2)
    assert results['transitions'].shape == (50, 4, 4, 2, 2)
    assert results['policies'].shape == (50, 2, 2)
    assert results['states'].shape == (50, 4, 2)
    
    print("✓ Basic simulation test passed")


def test_network_simulation():
    """Test network simulation."""
    print("Testing network simulation...")
    
    # Create network simulation
    network_sim = NetworkSimulation(network_type="ER", n_agents=10, p=0.6, lr_pB=0.1)
    
    # Run short simulation
    rounds = network_sim.run_simulation(T=10)
    
    # Check results
    assert len(rounds) == 10
    assert all(len(round_data) > 0 for round_data in rounds.values())
    
    # Test actions matrix extraction
    actions_matrix = network_sim.get_actions_matrix(rounds, 10)
    assert actions_matrix.shape == (10, 10)
    
    print("✓ Network simulation test passed")


def test_visualization():
    """Test visualization functions."""
    print("Testing visualization...")
    
    # Create test data
    import numpy as np
    actions = np.random.randint(0, 2, (100, 2))
    
    # Create visualizer
    visualizer = SimulationVisualizer("test_figures")
    
    # Test plotting functions
    visualizer.plot_action_evolution(actions, "Test Action Evolution", "test_actions.png")
    
    # Check if file was created
    assert os.path.exists("test_figures/test_actions.png")
    
    print("✓ Visualization test passed")


def main():
    """Run all tests."""
    print("Running tests...")
    print("=" * 40)
    
    try:
        test_basic_simulation()
        test_network_simulation()
        test_visualization()
        
        print("=" * 40)
        print("All tests passed! ✓")
        
        # Clean up test files
        import shutil
        if os.path.exists("test_figures"):
            shutil.rmtree("test_figures")
        
        return 0
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
