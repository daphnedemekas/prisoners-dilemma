#!/usr/bin/env python3
"""
Main script for Prisoner's Dilemma simulations.

This script runs all simulations and generates all figures for the project.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from models.agent import construct_agents, DualAgentSimulation
from models.network import NetworkSimulation
from simulation.sweep import ParameterSweep
from visualization.plots import SimulationVisualizer


def run_basic_simulation():
    """Run a basic dual-agent simulation."""
    print("Running basic dual-agent simulation...")
    
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
    
    # Run simulation
    results = sim.run_simulation(T=1000, initial_obs_1=[0], initial_obs_2=[0])
    
    # Create visualizer and generate plots
    visualizer = SimulationVisualizer("figures/basic_simulation")
    
    # Generate all plots
    visualizer.plot_action_evolution(results['actions'], "Basic Simulation - Action Evolution")
    visualizer.plot_policy_evolution(results['policies'])
    visualizer.plot_state_beliefs(results['states'])
    visualizer.plot_transition_matrices(results['transitions'])
    visualizer.plot_reward_analysis(results['actions'])
    visualizer.create_summary_figure(results, "basic_simulation_summary.png")
    
    print("Basic simulation completed. Figures saved to figures/basic_simulation/")
    return results


def run_network_simulation():
    """Run a network-based simulation."""
    print("Running network simulation...")
    
    # Create network simulation
    network_sim = NetworkSimulation(network_type="ER", n_agents=20, p=0.8, lr_pB=0.1)
    
    # Run simulation
    rounds = network_sim.run_simulation(T=50)
    
    # Extract actions matrix
    actions_matrix = network_sim.get_actions_matrix(rounds, 50)
    
    # Create visualizer
    visualizer = SimulationVisualizer("figures/network_simulation")
    
    # Generate network animation
    network = network_sim.get_network()
    pos = network_sim.get_positions()
    visualizer.plot_network_evolution(network, actions_matrix, pos)
    
    print("Network simulation completed. Animation saved to figures/network_simulation/")
    return rounds


def run_parameter_sweep():
    """Run parameter sweep simulations."""
    print("Running parameter sweep...")
    
    # Create parameter sweep
    sweep = ParameterSweep("results/parameter_sweep")
    
    # Run deterministic sweep with smaller range for demonstration
    lrs1 = np.linspace(0.01, 0.3, 20)
    lrs2 = np.linspace(0.01, 0.3, 20)
    
    results = sweep.run_deterministic_sweep(
        T=500, 
        lrs1=lrs1, 
        lrs2=lrs2, 
        only_actions=True
    )
    
    # Create visualizer
    visualizer = SimulationVisualizer("figures/parameter_sweep")
    
    # Generate heatmaps
    visualizer.plot_parameter_sweep_heatmap(
        results['actions'], 
        results['lrs1'], 
        results['lrs2'], 
        metric='cooperation_rate',
        save_name="cooperation_rate_heatmap.png"
    )
    
    visualizer.plot_parameter_sweep_heatmap(
        results['actions'], 
        results['lrs1'], 
        results['lrs2'], 
        metric='convergence_time',
        save_name="convergence_time_heatmap.png"
    )
    
    print("Parameter sweep completed. Results saved to figures/parameter_sweep/")
    return results


def run_stochastic_sweep():
    """Run stochastic parameter sweep."""
    print("Running stochastic parameter sweep...")
    
    # Create parameter sweep
    sweep = ParameterSweep("results/stochastic_sweep")
    
    # Run stochastic sweep for a few parameter combinations
    lrs1 = np.linspace(0.01, 0.3, 5)
    lrs2 = np.linspace(0.01, 0.3, 5)
    
    for task_id in range(5):
        sweep.run_stochastic_sweep(
            T=1000, 
            num_trials=10, 
            task_id=task_id, 
            alpha=1.0,
            lrs1=lrs1,
            lrs2=lrs2
        )
    
    print("Stochastic parameter sweep completed. Results saved to results/stochastic_sweep/")


def run_comparison_simulations():
    """Run comparison simulations with different parameters."""
    print("Running comparison simulations...")
    
    # Different learning rate combinations
    scenarios = [
        {"lr1": 0.05, "lr2": 0.05, "name": "low_low"},
        {"lr1": 0.3, "lr2": 0.3, "name": "high_high"},
        {"lr1": 0.05, "lr2": 0.3, "name": "low_high"},
        {"lr1": 0.3, "lr2": 0.05, "name": "high_low"},
    ]
    
    all_results = {}
    
    for scenario in scenarios:
        print(f"Running scenario: {scenario['name']}")
        
        # Create agents
        agent_1, agent_2, D = construct_agents(
            lr_pB_1=scenario["lr1"], 
            lr_pB_2=scenario["lr2"], 
            factors_to_learn="all",
            action_selection="stochastic",
            alpha_1=1.0,
            alpha_2=1.0
        )
        
        # Create simulation
        sim = DualAgentSimulation(agent_1, agent_2)
        
        # Run simulation
        results = sim.run_simulation(T=1000, initial_obs_1=[0], initial_obs_2=[0])
        all_results[scenario['name']] = results
        
        # Create visualizer
        visualizer = SimulationVisualizer(f"figures/comparison/{scenario['name']}")
        
        # Generate plots
        visualizer.plot_action_evolution(
            results['actions'], 
            f"Scenario {scenario['name']} - Action Evolution",
            "action_evolution.png"
        )
        visualizer.plot_reward_analysis(results['actions'])
        visualizer.create_summary_figure(results, "summary.png")
    
    # Create comparison plot
    visualizer = SimulationVisualizer("figures/comparison")
    
    # Plot cooperation rates for all scenarios
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for scenario_name, results in all_results.items():
        actions = results['actions']
        cooperation_rate = 1 - np.mean(actions, axis=1)
        time_steps = np.arange(len(cooperation_rate))
        ax.plot(time_steps, cooperation_rate, label=scenario_name, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Cooperation Rate')
    ax.set_title('Cooperation Rate Comparison Across Scenarios')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("figures/comparison/cooperation_comparison.png")
    plt.close()
    
    print("Comparison simulations completed. Results saved to figures/comparison/")
    return all_results


def create_readme_figures():
    """Create figures specifically for the README."""
    print("Creating README figures...")
    
    # Create a simple demonstration simulation
    agent_1, agent_2, D = construct_agents(
        lr_pB_1=0.1, 
        lr_pB_2=0.1, 
        factors_to_learn="all",
        action_selection="stochastic",
        alpha_1=1.0,
        alpha_2=1.0
    )
    
    sim = DualAgentSimulation(agent_1, agent_2)
    results = sim.run_simulation(T=200, initial_obs_1=[0], initial_obs_2=[0])
    
    # Create visualizer for README
    visualizer = SimulationVisualizer("figures/readme")
    
    # Create clean, publication-ready figures
    visualizer.plot_action_evolution(
        results['actions'], 
        "Prisoner's Dilemma: Agent Actions Over Time",
        "agent_actions.png"
    )
    
    visualizer.plot_reward_analysis(results['actions'], "reward_analysis.png")
    
    # Create a simple network visualization
    network_sim = NetworkSimulation(network_type="ER", n_agents=10, p=0.6, lr_pB=0.1)
    network = network_sim.get_network()
    pos = network_sim.get_positions()
    
    import matplotlib.pyplot as plt
    import networkx as nx
    
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw(network, pos, node_color='lightblue', node_size=500, 
           with_labels=True, font_size=12, font_weight='bold',
           edge_color='gray', alpha=0.7, ax=ax)
    ax.set_title("Network Topology for Multi-Agent Simulation")
    plt.tight_layout()
    plt.savefig("figures/readme/network_topology.png")
    plt.close()
    
    print("README figures created in figures/readme/")


def main():
    """Main function to run all simulations and generate all figures."""
    print("Starting Prisoner's Dilemma Simulation Suite")
    print("=" * 50)
    
    # Create output directories
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Run all simulations
    try:
        # Basic simulation
        basic_results = run_basic_simulation()
        
        # Network simulation
        network_results = run_network_simulation()
        
        # Parameter sweep
        sweep_results = run_parameter_sweep()
        
        # Stochastic sweep (commented out for speed, uncomment if needed)
        # run_stochastic_sweep()
        
        # Comparison simulations
        comparison_results = run_comparison_simulations()
        
        # Create README figures
        create_readme_figures()
        
        print("\n" + "=" * 50)
        print("All simulations completed successfully!")
        print("Figures saved to the 'figures' directory")
        print("Results saved to the 'results' directory")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
