"""
Network-based Prisoner's Dilemma simulations.

This module implements network simulations where agents play the prisoner's dilemma
with their neighbors in a network structure.
"""

import networkx as nx
import numpy as np
import random
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from models.agent import PrisonersDilemmaAgent


class NetworkSimulation:
    """
    Manages a network-based simulation of the Prisoner's Dilemma.
    
    Agents are placed on nodes of a network and play with their neighbors
    according to the network topology.
    """
    
    def __init__(self, network_type="ER", n_agents=20, p=0.8, lr_pB=0.1):
        """
        Initialize a network simulation.
        
        Args:
            network_type: Type of network ("ER" for Erdos-Renyi)
            n_agents: Number of agents in the network
            p: Connection probability for ER network
            lr_pB: Learning rate for all agents
        """
        self.n_agents = n_agents
        self.lr_pB = lr_pB
        
        # Create network
        if network_type == "ER":
            self.network = nx.fast_gnp_random_graph(n=n_agents, p=p)
        else:
            raise ValueError(f"Network type {network_type} not supported")
        
        # Set up layout for visualization
        self.pos = nx.spring_layout(self.network)
        self.network.add_nodes_from(self.pos.keys())
        for n, p in self.pos.items():
            self.network.nodes[n]["pos"] = p
        
        # Initialize agents
        self._setup_agents()
    
    def _setup_agents(self):
        """Set up agents on network nodes."""
        agents_dict = {}
        
        for i in self.network.nodes():
            agent_i = PrisonersDilemmaAgent(lr_pB=self.lr_pB)
            agent_i.observation = None
            agent_i.qs = agent_i.agent.D
            agents_dict[i] = agent_i
        
        nx.set_node_attributes(self.network, agents_dict, "agent")
    
    def run_simulation(self, T):
        """
        Run a network simulation where agents play iterative prisoner's dilemma
        with random neighbors at each trial.
        
        Args:
            T: Number of time steps
            
        Returns:
            Dictionary containing simulation results
        """
        rounds = {}
        
        for t in range(T):
            rounds[t] = {}
            
            for i in self.network.nodes():
                if rounds[t].get(i) is not None:
                    # This agent has already played as an opponent in this round
                    continue
                
                # Choose random neighbor
                i_neighbours = list(nx.neighbors(self.network, i))
                if not i_neighbours:
                    continue
                    
                o = random.choice(i_neighbours)
                
                # Get agents
                agent = self.network.nodes[i]["agent"]
                opponent = self.network.nodes[o]["agent"]
                
                # Set initial observations
                observation_1 = [0] if agent.observation is None else agent.observation
                observation_2 = [0] if opponent.observation is None else opponent.observation
                
                # Run one round
                action_1, action_2, B1, B2, q_pi_1, q_pi_2, qs_1, qs_2 = self._run_single_round(
                    agent, opponent, observation_1, observation_2, t
                )
                
                # Store results
                rounds[t][i] = {
                    "opponent": o,
                    "action": action_1,
                    "B": B1,
                    "q_pi": q_pi_1,
                    "q_s": qs_1,
                }
                rounds[t][o] = {
                    "opponent": i,
                    "action": action_2,
                    "B": B2,
                    "q_pi": q_pi_2,
                    "q_s": qs_2,
                }
        
        return rounds
    
    def _run_single_round(self, agent_1, agent_2, observation_1, observation_2, t):
        """Run a single round between two agents."""
        # Initialize previous states
        qs_prev_1 = agent_1.agent.D
        qs_prev_2 = agent_2.agent.D
        
        # Infer states
        qs_1 = agent_1.infer_states(observation_1)
        qs_2 = agent_2.infer_states(observation_2)
        
        # Update transition matrices if not first round
        if t > 0:
            agent_1.update_B(qs_prev_1)
            agent_2.update_B(qs_prev_2)
        
        # Infer policies
        q_pi_1, efe_1 = agent_1.infer_policies()
        q_pi_2, efe_2 = agent_2.infer_policies()
        
        # Sample actions
        action_1 = agent_1.sample_action()
        action_2 = agent_2.sample_action()
        
        # Set actions
        agent_1.agent.action = action_1
        agent_2.agent.action = action_2
        
        # Get observations based on actions
        new_obs_1 = self._get_observation(action_1[0], action_2[0])
        new_obs_2 = self._get_observation(action_2[0], action_1[0])
        
        # Update observations
        agent_1.observation = new_obs_1
        agent_2.observation = new_obs_2
        
        # Re-infer states with new observations
        qs_1 = agent_1.infer_states(new_obs_1)
        qs_2 = agent_2.infer_states(new_obs_2)
        
        # Update transition matrices
        agent_1.update_B(qs_1)
        agent_2.update_B(qs_2)
        
        # Re-infer policies
        q_pi_1, efe_1 = agent_1.infer_policies()
        q_pi_2, efe_2 = agent_2.infer_policies()
        
        # Sample final actions
        action_1 = agent_1.sample_action()
        action_2 = agent_2.sample_action()
        
        return (action_1[0], action_2[0], 
                agent_1.get_transition_matrix(), 
                agent_2.get_transition_matrix(),
                q_pi_1, q_pi_2, qs_1, qs_2)
    
    def _get_observation(self, action_1, action_2):
        """Convert action pair to observation."""
        action_1 = int(action_1)
        action_2 = int(action_2)
        
        if action_1 == 0 and action_2 == 0:
            return [0]  # Both cooperate
        elif action_1 == 0 and action_2 == 1:
            return [1]  # Agent 1 cooperates, Agent 2 defects
        elif action_1 == 1 and action_2 == 0:
            return [2]  # Agent 1 defects, Agent 2 cooperates
        elif action_1 == 1 and action_2 == 1:
            return [3]  # Both defect
    
    def get_actions_matrix(self, rounds, T):
        """
        Extract actions matrix from simulation results.
        
        Args:
            rounds: Simulation results from run_simulation
            T: Number of time steps
            
        Returns:
            Actions matrix of shape (T, n_agents)
        """
        actions = np.zeros((T, self.n_agents))
        
        for t in range(T):
            for i, round_data in rounds[t].items():
                actions[t, i] = round_data["action"]
        
        return actions
    
    def get_network(self):
        """Get the network object."""
        return self.network
    
    def get_positions(self):
        """Get node positions for visualization."""
        return self.pos
