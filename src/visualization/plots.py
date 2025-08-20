"""
Visualization functions for Prisoner's Dilemma simulation results.

This module contains functions to create various plots and figures
from simulation data.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import networkx as nx
import imageio
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


class SimulationVisualizer:
    """
    Handles visualization of simulation results.
    """
    
    def __init__(self, output_dir="figures"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
    
    def plot_action_evolution(self, actions, title="Action Evolution Over Time", 
                            save_name="action_evolution.png"):
        """
        Plot the evolution of actions over time.
        
        Args:
            actions: Array of shape (T, 2) with actions for both agents
            title: Plot title
            save_name: Filename to save the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        T = actions.shape[0]
        time_steps = np.arange(T)
        
        # Plot actions for both agents
        ax.plot(time_steps, actions[:, 0], label='Agent 1', linewidth=2, alpha=0.8)
        ax.plot(time_steps, actions[:, 1], label='Agent 2', linewidth=2, alpha=0.8)
        
        # Add horizontal lines for action values
        ax.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Cooperate')
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Defect')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Action (0=Cooperate, 1=Defect)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{save_name}")
        plt.close()
    
    def plot_parameter_sweep_heatmap(self, actions_data, lrs1, lrs2, 
                                   metric='cooperation_rate', save_name="parameter_sweep.png"):
        """
        Create a heatmap of parameter sweep results.
        
        Args:
            actions_data: Array of shape (T, 2, len(lrs1), len(lrs2))
            lrs1: Learning rates for agent 1
            lrs2: Learning rates for agent 2
            metric: Metric to plot ('cooperation_rate', 'convergence_time', etc.)
            save_name: Filename to save the plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if metric == 'cooperation_rate':
            # Calculate cooperation rate (average of last 100 time steps)
            last_steps = min(100, actions_data.shape[0])
            cooperation_rate = 1 - np.mean(actions_data[-last_steps:, :, :, :], axis=(0, 1))
            title = "Cooperation Rate"
            cmap = 'viridis'
        elif metric == 'convergence_time':
            # Calculate time to convergence (first time both agents take same action for 10 steps)
            convergence_time = np.zeros((len(lrs1), len(lrs2)))
            for i in range(len(lrs1)):
                for j in range(len(lrs2)):
                    actions = actions_data[:, :, i, j]
                    # Find first time both agents have same action for 10 consecutive steps
                    for t in range(10, len(actions)):
                        if np.all(actions[t-10:t, 0] == actions[t-10:t, 1]):
                            convergence_time[i, j] = t
                            break
                    else:
                        convergence_time[i, j] = len(actions)
            cooperation_rate = convergence_time
            title = "Convergence Time"
            cmap = 'plasma'
        
        # Create heatmap
        im = ax.imshow(cooperation_rate, cmap=cmap, aspect='auto', 
                      extent=[lrs2[0], lrs2[-1], lrs1[0], lrs1[-1]], origin='lower')
        
        ax.set_xlabel('Agent 2 Learning Rate')
        ax.set_ylabel('Agent 1 Learning Rate')
        ax.set_title(title)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(title)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{save_name}")
        plt.close()
    
    def plot_network_evolution(self, network, actions_matrix, pos, save_name="network_evolution.gif"):
        """
        Create an animated GIF showing network evolution.
        
        Args:
            network: NetworkX graph
            actions_matrix: Array of shape (T, n_agents) with actions
            pos: Node positions
            save_name: Filename to save the animation
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def animate(frame):
            ax.clear()
            
            # Get actions for this frame
            actions = actions_matrix[frame]
            
            # Create color map
            colors = []
            for action in actions:
                if int(action) == 0:
                    colors.append('green')  # Cooperate
                else:
                    colors.append('red')    # Defect
            
            # Draw network
            nx.draw(network, pos, node_color=colors, node_size=500, 
                   with_labels=True, font_size=10, font_weight='bold',
                   edge_color='gray', alpha=0.7)
            
            ax.set_title(f'Network State at Time Step {frame}')
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=min(100, len(actions_matrix)), 
                                     interval=200, repeat=True)
        
        # Save as GIF
        anim.save(f"{self.output_dir}/{save_name}", writer=PillowWriter(fps=5))
        plt.close()
    
    def plot_transition_matrices(self, B_over_time, agent_idx=0, save_name="transition_matrices.png"):
        """
        Plot evolution of transition matrices.
        
        Args:
            B_over_time: Array of shape (T, 4, 4, 2, 2) with transition matrices
            agent_idx: Which agent to plot (0 or 1)
            save_name: Filename to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # Plot transition matrices for each action
        for action in range(2):
            action_name = "Cooperate" if action == 0 else "Defect"
            
            # Get transition matrix for this action
            B_action = B_over_time[-1, :, :, action, agent_idx]
            
            im = axes[action].imshow(B_action, cmap='viridis', vmin=0, vmax=1)
            axes[action].set_title(f'Transition Matrix - {action_name}')
            axes[action].set_xlabel('Next State')
            axes[action].set_ylabel('Current State')
            
            # Add text annotations
            for i in range(4):
                for j in range(4):
                    text = axes[action].text(j, i, f'{B_action[i, j]:.2f}',
                                           ha="center", va="center", color="white")
            
            plt.colorbar(im, ax=axes[action])
        
        # Plot evolution over time for key transitions
        time_steps = np.arange(B_over_time.shape[0])
        
        # Plot some key transition probabilities over time
        key_transitions = [(0, 0), (1, 1), (2, 2), (3, 3)]  # Diagonal elements
        
        for i, (from_state, to_state) in enumerate(key_transitions):
            for action in range(2):
                action_name = "C" if action == 0 else "D"
                prob = B_over_time[:, from_state, to_state, action, agent_idx]
                axes[2].plot(time_steps, prob, label=f'{action_name}: {from_state}→{to_state}', alpha=0.8)
        
        axes[2].set_xlabel('Time Step')
        axes[2].set_ylabel('Transition Probability')
        axes[2].set_title('Key Transition Probabilities Over Time')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Plot cooperation rate over time
        axes[3].plot(time_steps, 1 - np.mean(B_over_time[:, :, :, 1, agent_idx], axis=(1, 2)), 
                    label='Cooperation Rate', linewidth=2)
        axes[3].set_xlabel('Time Step')
        axes[3].set_ylabel('Cooperation Rate')
        axes[3].set_title('Agent Cooperation Rate Over Time')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{save_name}")
        plt.close()
    
    def plot_policy_evolution(self, q_pi_over_time, save_name="policy_evolution.png"):
        """
        Plot evolution of policy beliefs.
        
        Args:
            q_pi_over_time: Array of shape (T, 2, 2) with policy beliefs
            save_name: Filename to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        time_steps = np.arange(q_pi_over_time.shape[0])
        
        for agent in range(2):
            for policy in range(2):
                policy_name = "Cooperate" if policy == 0 else "Defect"
                prob = q_pi_over_time[:, policy, agent]
                axes[agent].plot(time_steps, prob, label=f'Policy: {policy_name}', linewidth=2)
            
            axes[agent].set_xlabel('Time Step')
            axes[agent].set_ylabel('Policy Probability')
            axes[agent].set_title(f'Agent {agent + 1} Policy Evolution')
            axes[agent].legend()
            axes[agent].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{save_name}")
        plt.close()
    
    def plot_state_beliefs(self, q_s_over_time, save_name="state_beliefs.png"):
        """
        Plot evolution of state beliefs.
        
        Args:
            q_s_over_time: Array of shape (T, 4, 2) with state beliefs
            save_name: Filename to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        time_steps = np.arange(q_s_over_time.shape[0])
        state_names = ['CC', 'CD', 'DC', 'DD']
        
        for agent in range(2):
            for state in range(4):
                prob = q_s_over_time[:, state, agent]
                axes[agent].plot(time_steps, prob, label=f'State: {state_names[state]}', linewidth=2)
            
            axes[agent].set_xlabel('Time Step')
            axes[agent].set_ylabel('State Probability')
            axes[agent].set_title(f'Agent {agent + 1} State Beliefs')
            axes[agent].legend()
            axes[agent].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{save_name}")
        plt.close()
    
    def plot_reward_analysis(self, actions, save_name="reward_analysis.png"):
        """
        Plot reward analysis over time.
        
        Args:
            actions: Array of shape (T, 2) with actions
            save_name: Filename to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
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
        
        # Plot individual rewards
        axes[0, 0].plot(time_steps, rewards[:, 0], label='Agent 1', linewidth=2)
        axes[0, 0].plot(time_steps, rewards[:, 1], label='Agent 2', linewidth=2)
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Individual Rewards Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot cumulative rewards
        cumulative_rewards = np.cumsum(rewards, axis=0)
        axes[0, 1].plot(time_steps, cumulative_rewards[:, 0], label='Agent 1', linewidth=2)
        axes[0, 1].plot(time_steps, cumulative_rewards[:, 1], label='Agent 2', linewidth=2)
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Cumulative Reward')
        axes[0, 1].set_title('Cumulative Rewards Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot cooperation rate
        cooperation_rate = 1 - np.mean(actions, axis=1)
        axes[1, 0].plot(time_steps, cooperation_rate, linewidth=2, color='green')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Cooperation Rate')
        axes[1, 0].set_title('Overall Cooperation Rate')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot reward distribution
        axes[1, 1].hist(rewards.flatten(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].set_xlabel('Reward Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Reward Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{save_name}")
        plt.close()
    
    def create_summary_figure(self, results, save_name="summary_figure.png"):
        """
        Create a comprehensive summary figure with multiple subplots.
        
        Args:
            results: Dictionary containing simulation results
            save_name: Filename to save the plot
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Create subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Action evolution
        ax1 = fig.add_subplot(gs[0, :2])
        actions = results['actions']
        T = actions.shape[0]
        time_steps = np.arange(T)
        ax1.plot(time_steps, actions[:, 0], label='Agent 1', linewidth=2)
        ax1.plot(time_steps, actions[:, 1], label='Agent 2', linewidth=2)
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Action')
        ax1.set_title('Action Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Cooperation rate
        ax2 = fig.add_subplot(gs[0, 2])
        cooperation_rate = 1 - np.mean(actions, axis=1)
        ax2.plot(time_steps, cooperation_rate, linewidth=2, color='green')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Cooperation Rate')
        ax2.set_title('Cooperation Rate')
        ax2.grid(True, alpha=0.3)
        
        # Policy evolution
        ax3 = fig.add_subplot(gs[1, 0])
        if 'policies' in results:
            policies = results['policies']
            for agent in range(2):
                for policy in range(2):
                    policy_name = "C" if policy == 0 else "D"
                    prob = policies[:, policy, agent]
                    ax3.plot(time_steps, prob, label=f'Agent{agent+1}-{policy_name}', alpha=0.8)
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Policy Probability')
        ax3.set_title('Policy Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # State beliefs
        ax4 = fig.add_subplot(gs[1, 1])
        if 'states' in results:
            states = results['states']
            state_names = ['CC', 'CD', 'DC', 'DD']
            for state in range(4):
                prob = states[:, state, 0]  # Agent 1
                ax4.plot(time_steps, prob, label=f'State: {state_names[state]}', alpha=0.8)
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('State Probability')
        ax4.set_title('State Beliefs (Agent 1)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Transition matrix heatmap
        ax5 = fig.add_subplot(gs[1, 2])
        if 'transitions' in results:
            transitions = results['transitions']
            final_B = transitions[-1, :, :, 0, 0]  # Agent 1, cooperate action
            im = ax5.imshow(final_B, cmap='viridis', vmin=0, vmax=1)
            ax5.set_title('Final Transition Matrix\n(Agent 1, Cooperate)')
            plt.colorbar(im, ax=ax5)
        
        # Reward analysis
        ax6 = fig.add_subplot(gs[2, :])
        rewards = np.zeros((T, 2))
        for t in range(T):
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
        ax6.plot(time_steps, cumulative_rewards[:, 0], label='Agent 1', linewidth=2)
        ax6.plot(time_steps, cumulative_rewards[:, 1], label='Agent 2', linewidth=2)
        ax6.set_xlabel('Time Step')
        ax6.set_ylabel('Cumulative Reward')
        ax6.set_title('Cumulative Rewards')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{save_name}")
        plt.close()


def create_network_animation(network, actions_matrix, pos, output_dir="figures"):
    """
    Legacy function for creating network animation.
    """
    visualizer = SimulationVisualizer(output_dir)
    return visualizer.plot_network_evolution(network, actions_matrix, pos)
