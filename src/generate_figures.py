#!/usr/bin/env python3
"""
Enhanced figure generation script for Prisoner's Dilemma simulations.

This script creates high-quality, publication-ready figures with improved styling.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from models.agent import construct_agents, DualAgentSimulation
from models.network import NetworkSimulation
from visualization.plots import SimulationVisualizer


def setup_plotting_style():
    """Set up publication-ready plotting style."""
    plt.style.use('seaborn-v0_8')
    
    # Set font sizes
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 18
    
    # Set figure DPI
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    
    # Set color palette
    sns.set_palette("husl")
    
    # Set grid style
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'


def create_enhanced_action_plot(actions, save_path="figures/enhanced_actions.png"):
    """Create an enhanced action evolution plot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])
    
    T = actions.shape[0]
    time_steps = np.arange(T)
    
    # Main action plot
    colors = ['#2E8B57', '#DC143C']  # Sea green and crimson
    labels = ['Agent 1 (Cooperate)', 'Agent 2 (Cooperate)']
    
    for i in range(2):
        # Convert to cooperation (1 - action)
        cooperation = 1 - actions[:, i]
        ax1.plot(time_steps, cooperation, color=colors[i], linewidth=2.5, 
                label=labels[i], alpha=0.8)
    
    # Add horizontal lines for reference
    ax1.axhline(y=1, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='Full Cooperation')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Full Defection')
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1, label='Mixed Strategy')
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Cooperation Level')
    ax1.set_title('Agent Cooperation Evolution Over Time', fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # Cooperation rate over time
    cooperation_rate = 1 - np.mean(actions, axis=1)
    ax2.fill_between(time_steps, cooperation_rate, alpha=0.6, color='#4CAF50', label='Population Cooperation Rate')
    ax2.plot(time_steps, cooperation_rate, color='#2E7D32', linewidth=2.5)
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Population Cooperation Rate')
    ax2.set_title('Overall Cooperation Rate', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()


def create_strategy_heatmap(actions, save_path="figures/strategy_heatmap.png"):
    """Create a heatmap showing strategy patterns."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Convert actions to cooperation (1 - action)
    cooperation = 1 - actions
    
    # Create sliding window analysis
    window_size = 20
    if len(actions) > window_size:
        cooperation_windows = []
        for i in range(len(actions) - window_size + 1):
            window = cooperation[i:i+window_size]
            cooperation_windows.append(np.mean(window, axis=0))
        
        cooperation_windows = np.array(cooperation_windows)
        
        # Heatmap of cooperation over time
        im1 = ax1.imshow(cooperation_windows.T, cmap='RdYlGn', aspect='auto', 
                        extent=[0, len(cooperation_windows), 0, 2], vmin=0, vmax=1)
        ax1.set_xlabel('Time Window')
        ax1.set_ylabel('Agent')
        ax1.set_title('Cooperation Patterns Over Time', fontweight='bold')
        ax1.set_yticks([0.5, 1.5])
        ax1.set_yticklabels(['Agent 1', 'Agent 2'])
        plt.colorbar(im1, ax=ax1, label='Cooperation Rate')
    
    # Strategy correlation matrix
    correlation_matrix = np.corrcoef(cooperation.T)
    im2 = ax2.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax2.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                           ha="center", va="center", color="white", fontweight='bold')
    
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Agent 1', 'Agent 2'])
    ax2.set_yticklabels(['Agent 1', 'Agent 2'])
    ax2.set_title('Strategy Correlation Matrix', fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Correlation')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()


def create_reward_analysis_enhanced(actions, save_path="figures/enhanced_rewards.png"):
    """Create enhanced reward analysis."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    T = actions.shape[0]
    time_steps = np.arange(T)
    
    # Calculate rewards
    rewards = np.zeros((T, 2))
    for t in range(T):
        action_1, action_2 = actions[t]
        if action_1 == 0 and action_2 == 0:  # Both cooperate
            rewards[t] = [3, 3]
        elif action_1 == 0 and action_2 == 1:  # Agent 1 cooperates, Agent 2 defects
            rewards[t] = [1, 4]
        elif action_1 == 1 and action_2 == 0:  # Agent 1 defects, Agent 2 cooperates
            rewards[t] = [4, 1]
        else:  # Both defect
            rewards[t] = [2, 2]
    
    # Individual rewards over time
    colors = ['#FF6B6B', '#4ECDC4']
    for i in range(2):
        ax1.plot(time_steps, rewards[:, i], color=colors[i], linewidth=2.5, 
                label=f'Agent {i+1}', alpha=0.8)
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Reward')
    ax1.set_title('Individual Rewards Over Time', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cumulative rewards
    cumulative_rewards = np.cumsum(rewards, axis=0)
    for i in range(2):
        ax2.plot(time_steps, cumulative_rewards[:, i], color=colors[i], linewidth=2.5, 
                label=f'Agent {i+1}', alpha=0.8)
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Cumulative Reward')
    ax2.set_title('Cumulative Rewards Over Time', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Cooperation rate with reward overlay
    cooperation_rate = 1 - np.mean(actions, axis=1)
    ax3.plot(time_steps, cooperation_rate, color='#2E8B57', linewidth=2.5, label='Cooperation Rate')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(time_steps, np.mean(rewards, axis=1), color='#FF8C00', linewidth=2.5, 
                 alpha=0.7, label='Average Reward')
    
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Cooperation Rate', color='#2E8B57')
    ax3_twin.set_ylabel('Average Reward', color='#FF8C00')
    ax3.set_title('Cooperation Rate vs Average Reward', fontweight='bold')
    
    # Combine legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Reward distribution
    reward_values = rewards.flatten()
    ax4.hist(reward_values, bins=[1, 2, 3, 4, 5], alpha=0.7, color='#9B59B6', 
            edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('Reward Value')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Reward Distribution', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()


def create_network_visualization_enhanced(save_path="figures/enhanced_network.png"):
    """Create enhanced network visualization."""
    # Create network simulation
    network_sim = NetworkSimulation(network_type="ER", n_agents=15, p=0.6, lr_pB=0.1)
    network = network_sim.get_network()
    pos = network_sim.get_positions()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Network topology
    node_colors = ['#3498DB'] * len(network.nodes())
    node_sizes = [500] * len(network.nodes())
    
    nx.draw(network, pos, node_color=node_colors, node_size=node_sizes,
           with_labels=True, font_size=10, font_weight='bold',
           edge_color='gray', alpha=0.7, ax=ax1)
    ax1.set_title('Network Topology', fontweight='bold', fontsize=16)
    
    # Network statistics
    degree_sequence = sorted([d for n, d in network.degree()], reverse=True)
    ax2.hist(degree_sequence, bins=range(min(degree_sequence), max(degree_sequence) + 2, 1),
            alpha=0.7, color='#E74C3C', edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Node Degree')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Degree Distribution', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add network statistics as text
    stats_text = f"""
    Network Statistics:
    • Nodes: {network.number_of_nodes()}
    • Edges: {network.number_of_edges()}
    • Average Degree: {np.mean(degree_sequence):.2f}
    • Clustering Coefficient: {nx.average_clustering(network):.3f}
    • Average Path Length: {nx.average_shortest_path_length(network):.3f}
    """
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()


def create_comparison_analysis(save_path="figures/comparison_analysis.png"):
    """Create comparison analysis of different scenarios."""
    scenarios = [
        {"lr1": 0.05, "lr2": 0.05, "name": "Low Learning Rates", "color": "#3498DB"},
        {"lr1": 0.3, "lr2": 0.3, "name": "High Learning Rates", "color": "#E74C3C"},
        {"lr1": 0.05, "lr2": 0.3, "name": "Asymmetric Learning", "color": "#2ECC71"},
    ]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    all_results = {}
    
    for scenario in scenarios:
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
        results = sim.run_simulation(T=500, initial_obs_1=[0], initial_obs_2=[0])
        all_results[scenario['name']] = results
        
        # Plot cooperation rate
        actions = results['actions']
        cooperation_rate = 1 - np.mean(actions, axis=1)
        time_steps = np.arange(len(cooperation_rate))
        ax1.plot(time_steps, cooperation_rate, color=scenario['color'], linewidth=2.5, 
                label=scenario['name'], alpha=0.8)
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Cooperation Rate')
    ax1.set_title('Cooperation Rate Comparison', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Final cooperation rates
    final_rates = []
    scenario_names = []
    for scenario in scenarios:
        actions = all_results[scenario['name']]['actions']
        final_rate = 1 - np.mean(actions[-50:], axis=1).mean()  # Last 50 steps
        final_rates.append(final_rate)
        scenario_names.append(scenario['name'])
    
    bars = ax2.bar(scenario_names, final_rates, color=[s['color'] for s in scenarios], alpha=0.7)
    ax2.set_ylabel('Final Cooperation Rate')
    ax2.set_title('Final Cooperation Rates by Scenario', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, rate in zip(bars, final_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Convergence analysis
    convergence_times = []
    for scenario in scenarios:
        actions = all_results[scenario['name']]['actions']
        # Find time to convergence (when cooperation rate stabilizes)
        cooperation_rate = 1 - np.mean(actions, axis=1)
        # Simple convergence metric: when rate changes less than 0.1 for 20 steps
        for t in range(20, len(cooperation_rate)):
            if np.std(cooperation_rate[t-20:t]) < 0.1:
                convergence_times.append(t)
                break
        else:
            convergence_times.append(len(cooperation_rate))
    
    bars = ax3.bar(scenario_names, convergence_times, color=[s['color'] for s in scenarios], alpha=0.7)
    ax3.set_ylabel('Convergence Time (Steps)')
    ax3.set_title('Time to Strategy Convergence', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, time in zip(bars, convergence_times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{time}', ha='center', va='bottom', fontweight='bold')
    
    # Reward comparison
    for scenario in scenarios:
        actions = all_results[scenario['name']]['actions']
        rewards = np.zeros((len(actions), 2))
        for t in range(len(actions)):
            action_1, action_2 = actions[t]
            if action_1 == 0 and action_2 == 0:
                rewards[t] = [3, 3]
            elif action_1 == 0 and action_2 == 1:
                rewards[t] = [1, 4]
            elif action_1 == 1 and action_2 == 0:
                rewards[t] = [4, 1]
            else:
                rewards[t] = [2, 2]
        
        cumulative_rewards = np.cumsum(rewards, axis=0)
        avg_cumulative = np.mean(cumulative_rewards, axis=1)
        time_steps = np.arange(len(avg_cumulative))
        ax4.plot(time_steps, avg_cumulative, color=scenario['color'], linewidth=2.5, 
                label=scenario['name'], alpha=0.8)
    
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Average Cumulative Reward')
    ax4.set_title('Average Cumulative Rewards', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    """Generate all enhanced figures."""
    print("Generating enhanced visualizations...")
    
    # Set up plotting style
    setup_plotting_style()
    
    # Create output directory
    os.makedirs("figures", exist_ok=True)
    
    # Run a basic simulation for analysis
    print("Running simulation for analysis...")
    agent_1, agent_2, D = construct_agents(
        lr_pB_1=0.1, 
        lr_pB_2=0.1, 
        factors_to_learn="all",
        action_selection="stochastic",
        alpha_1=1.0,
        alpha_2=1.0
    )
    
    sim = DualAgentSimulation(agent_1, agent_2)
    results = sim.run_simulation(T=1000, initial_obs_1=[0], initial_obs_2=[0])
    
    # Generate enhanced figures
    print("Creating enhanced action plot...")
    create_enhanced_action_plot(results['actions'])
    
    print("Creating strategy heatmap...")
    create_strategy_heatmap(results['actions'])
    
    print("Creating enhanced reward analysis...")
    create_reward_analysis_enhanced(results['actions'])
    
    print("Creating enhanced network visualization...")
    create_network_visualization_enhanced()
    
    print("Creating comparison analysis...")
    create_comparison_analysis()
    
    print("All enhanced figures generated successfully!")
    print("Figures saved to the 'figures' directory")


if __name__ == "__main__":
    main()
