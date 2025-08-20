"""
Parameter sweep simulations for the Prisoner's Dilemma.

This module implements systematic parameter sweeps to explore how different
learning rates and other parameters affect agent behavior.
"""

import numpy as np
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from models.agent import construct_agents, DualAgentSimulation


class ParameterSweep:
    """
    Manages parameter sweep simulations across different learning rates.
    """
    
    def __init__(self, output_dir="results"):
        """
        Initialize parameter sweep.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def run_deterministic_sweep(self, T=1000, lrs1=None, lrs2=None, only_actions=False):
        """
        Run deterministic parameter sweep.
        
        Args:
            T: Number of time steps
            lrs1: Learning rates for agent 1 (default: linspace(0.01, 0.6, 100))
            lrs2: Learning rates for agent 2 (default: linspace(0.01, 0.6, 100))
            only_actions: Whether to only collect actions (faster)
        """
        if lrs1 is None:
            lrs1 = np.linspace(0.01, 0.6, 100)
        if lrs2 is None:
            lrs2 = np.linspace(0.01, 0.6, 100)
        
        actions_over_time_all = np.zeros((T, 2, len(lrs1), len(lrs2)))
        
        if not only_actions:
            B1_over_time_all = np.zeros((T, 4, 4, 2, 2, len(lrs1), len(lrs2)))
            q_pi_over_time_all = np.zeros((T, 2, 2, len(lrs1), len(lrs2)))
        
        for k, lr_pB_1 in enumerate(lrs1):
            print(f"lr1 = {lr_pB_1:.3f}")
            for j, lr_pB_2 in enumerate(lrs2):
                
                # Construct agents
                agent_1, agent_2, D = construct_agents(
                    lr_pB_1=lr_pB_1, 
                    lr_pB_2=lr_pB_2, 
                    factors_to_learn="all"
                )
                
                # Create simulation
                sim = DualAgentSimulation(agent_1, agent_2)
                
                if only_actions:
                    # Run simulation and collect only actions
                    results = sim.run_simulation(T, initial_obs_1=[0], initial_obs_2=[0])
                    actions_over_time_all[:, :, k, j] = results['actions']
                else:
                    # Run full simulation
                    results = sim.run_simulation(T, initial_obs_1=[0], initial_obs_2=[0])
                    actions_over_time_all[:, :, k, j] = results['actions']
                    B1_over_time_all[:, :, :, :, :, k, j] = results['transitions']
                    q_pi_over_time_all[:, :, :, k, j] = results['policies']
        
        # Save results
        np.save(f"{self.output_dir}/actions_over_time_all", actions_over_time_all, allow_pickle=True)
        
        if not only_actions:
            np.save(f"{self.output_dir}/B1_over_time_all", B1_over_time_all, allow_pickle=True)
            np.save(f"{self.output_dir}/q_pi_over_time_all", q_pi_over_time_all, allow_pickle=True)
        
        return {
            'actions': actions_over_time_all,
            'lrs1': lrs1,
            'lrs2': lrs2,
            'T': T
        }
    
    def run_stochastic_sweep(self, T=2000, num_trials=100, task_id=0, alpha=1.0, 
                            lrs1=None, lrs2=None):
        """
        Run stochastic parameter sweep with multiple trials.
        
        Args:
            T: Number of time steps
            num_trials: Number of trials per parameter combination
            task_id: Task ID for parallel processing
            alpha: Base precision parameter
            lrs1: Learning rates for agent 1
            lrs2: Learning rates for agent 2
        """
        if lrs1 is None:
            lrs1 = np.linspace(0.01, 0.6, 100)
        if lrs2 is None:
            lrs2 = np.linspace(0.01, 0.6, 100)
        
        lr_pB_1 = lrs1[task_id]
        
        # Create directory for this learning rate
        lr_dir = f"{self.output_dir}/stochastic/{lr_pB_1}"
        os.makedirs(lr_dir, exist_ok=True)
        
        print(f"lr1 = {lr_pB_1:.3f}")
        
        for j, lr_pB_2 in enumerate(lrs2):
            # Check if already computed
            result_file = f"{lr_dir}/{lr_pB_2}/actions_over_time_all.npy"
            if os.path.exists(result_file):
                continue
            
            # Create directory for this parameter combination
            param_dir = f"{lr_dir}/{lr_pB_2}"
            os.makedirs(param_dir, exist_ok=True)
            
            print(f"  lr2 = {lr_pB_2:.3f}")
            
            # Collect results across trials
            collect = []
            for t in range(num_trials):
                # Add noise to precision parameters
                alpha_1 = np.random.normal(alpha, 0.15)
                alpha_2 = np.random.normal(alpha, 0.15)
                
                # Construct agents
                agent_1, agent_2, D = construct_agents(
                    lr_pB_1=lr_pB_1, 
                    lr_pB_2=lr_pB_2, 
                    factors_to_learn="all",
                    action_selection="stochastic",
                    alpha_1=alpha_1,
                    alpha_2=alpha_2
                )
                
                # Create simulation
                sim = DualAgentSimulation(agent_1, agent_2)
                
                # Run simulation
                results = sim.run_simulation(T, initial_obs_1=[0], initial_obs_2=[0])
                collect.append(results['actions'])
            
            # Average across trials
            actions_over_time_all = np.mean(np.array(collect), axis=0)
            
            # Save results
            np.save(f"{param_dir}/actions_over_time_all", actions_over_time_all, allow_pickle=True)
    
    def run_parallel_stochastic_sweep(self, num_tasks=100, **kwargs):
        """
        Run stochastic sweep across multiple tasks for parallel processing.
        
        Args:
            num_tasks: Number of parallel tasks
            **kwargs: Additional arguments for run_stochastic_sweep
        """
        for task_id in range(num_tasks):
            self.run_stochastic_sweep(task_id=task_id, **kwargs)


def run_dual_sweep_stochastic(T, num_trials, task_id, alpha=1, 
                             lrs1=None, lrs2=None, output_dir="results"):
    """
    Legacy function for backward compatibility.
    """
    sweep = ParameterSweep(output_dir)
    return sweep.run_stochastic_sweep(T, num_trials, task_id, alpha, lrs1, lrs2)


def run_dual_sweep_deterministic(T, lrs1=None, lrs2=None, only_actions=False, output_dir="results"):
    """
    Legacy function for backward compatibility.
    """
    sweep = ParameterSweep(output_dir)
    return sweep.run_deterministic_sweep(T, lrs1, lrs2, only_actions)
