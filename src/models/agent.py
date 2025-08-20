"""
Agent models for the Prisoner's Dilemma simulation.

This module contains the core agent classes and functions for implementing
active inference agents in a prisoner's dilemma setting.
"""

import numpy as np
from pymdp.agent import Agent
from pymdp import utils
from pymdp.maths import softmax, spm_log_single


class PrisonersDilemmaAgent:
    """
    A specialized agent for playing the Prisoner's Dilemma game using active inference.
    
    This agent implements the core active inference loop including:
    - State inference
    - Policy selection
    - Action sampling
    - Learning of transition dynamics
    """
    
    def __init__(self, lr_pB=0.1, factors_to_learn="all", action_selection="stochastic", alpha=1.0):
        """
        Initialize a Prisoner's Dilemma agent.
        
        Args:
            lr_pB: Learning rate for transition matrix updates
            factors_to_learn: Which factors to learn (default: "all")
            action_selection: Action selection method ("stochastic" or "deterministic")
            alpha: Precision parameter for action selection
        """
        self.lr_pB = lr_pB
        self.factors_to_learn = factors_to_learn
        self.action_selection = action_selection
        self.alpha = alpha
        
        # Initialize agent parameters
        A, B, C, D, pB_1, pB_2 = self._get_agent_params()
        
        # Create the underlying pymdp agent
        self.agent = Agent(
            A=A, B=B, C=C, D=D, pB=pB_1, lr_pB=lr_pB, factors_to_learn=factors_to_learn
        )
        
        # Set action selection method
        self.agent.action_selection = action_selection
        self.agent.alpha = alpha
        
        # Initialize observation and state
        self.observation = None
        self.qs = D
        
    def _get_agent_params(self):
        """Returns standard parameters for a prisoners dilemma agent."""
        A = self._construct_A()
        B = self._construct_B()
        
        C = utils.obj_array(1)
        C[0] = np.array([3, 1, 4, 2])  # Reward preferences
        
        D = utils.obj_array(1)
        D[0] = np.array([0.25, 0.25, 0.25, 0.25])  # Initial state beliefs
        
        pB_1 = utils.dirichlet_like(B)
        pB_2 = utils.dirichlet_like(B)
        
        return A, B, C, D, pB_1, pB_2
    
    def _construct_A(self):
        """Construct observation likelihood matrix."""
        A = utils.obj_array(1)
        A1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        A[0] = A1
        return A
    
    def _construct_B(self):
        """Construct transition matrix."""
        B = utils.obj_array(1)
        B_1 = np.ones((4, 4, 2)) * 0.5
        B_1[2:, :, 0] = 0.0  # Defect action can't lead to cooperation states
        B_1[:2, :, 1] = 0.0  # Cooperate action can't lead to defection states
        B[0] = B_1
        return B
    
    def infer_states(self, observation):
        """Infer states given observation."""
        return self.agent.infer_states(observation)
    
    def infer_policies(self):
        """Infer policies."""
        return self.agent.infer_policies()
    
    def sample_action(self):
        """Sample an action based on current beliefs."""
        return self.agent.sample_action()
    
    def update_B(self, qs_prev):
        """Update transition matrix based on experience."""
        return self.agent.update_B(qs_prev)
    
    def get_transition_matrix(self):
        """Get current transition matrix."""
        return self.agent.B[0]
    
    def get_state_beliefs(self):
        """Get current state beliefs."""
        return self.agent.qs[0]


class DualAgentSimulation:
    """
    Manages a simulation between two Prisoner's Dilemma agents.
    """
    
    def __init__(self, agent_1, agent_2):
        """
        Initialize a dual agent simulation.
        
        Args:
            agent_1: First PrisonersDilemmaAgent
            agent_2: Second PrisonersDilemmaAgent
        """
        self.agent_1 = agent_1
        self.agent_2 = agent_2
        self.D = agent_1.agent.D
    
    def run_single_round(self, observation_1, observation_2, t):
        """
        Run a single round of the prisoner's dilemma.
        
        Args:
            observation_1: Initial observation for agent 1
            observation_2: Initial observation for agent 2
            t: Current time step
            
        Returns:
            Tuple of (action_1, action_2, B1, B2, q_pi_1, q_pi_2, qs_1, qs_2)
        """
        # Initialize previous states
        qs_prev_1 = self.D
        qs_prev_2 = self.D
        
        # Infer states
        qs_1 = self.agent_1.infer_states(observation_1)
        qs_2 = self.agent_2.infer_states(observation_2)
        
        # Update transition matrices if not first round
        if t > 0:
            self.agent_1.update_B(qs_prev_1)
            self.agent_2.update_B(qs_prev_2)
        
        # Infer policies
        q_pi_1, efe_1 = self.agent_1.infer_policies()
        q_pi_2, efe_2 = self.agent_2.infer_policies()
        
        # Sample actions
        action_1 = self.agent_1.sample_action()
        action_2 = self.agent_2.sample_action()
        
        # Set actions
        self.agent_1.agent.action = action_1
        self.agent_2.agent.action = action_2
        
        # Get observations based on actions
        new_obs_1 = self._get_observation(action_1[0], action_2[0])
        new_obs_2 = self._get_observation(action_2[0], action_1[0])
        
        # Update observations
        self.agent_1.observation = new_obs_1
        self.agent_2.observation = new_obs_2
        
        # Re-infer states with new observations
        qs_1 = self.agent_1.infer_states(new_obs_1)
        qs_2 = self.agent_2.infer_states(new_obs_2)
        
        # Update transition matrices
        self.agent_1.update_B(qs_1)
        self.agent_2.update_B(qs_2)
        
        # Re-infer policies
        q_pi_1, efe_1 = self.agent_1.infer_policies()
        q_pi_2, efe_2 = self.agent_2.infer_policies()
        
        # Sample final actions
        action_1 = self.agent_1.sample_action()
        action_2 = self.agent_2.sample_action()
        
        return (action_1[0], action_2[0], 
                self.agent_1.get_transition_matrix(), 
                self.agent_2.get_transition_matrix(),
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
    
    def run_simulation(self, T, initial_obs_1=[0], initial_obs_2=[0]):
        """
        Run a full simulation for T time steps.
        
        Args:
            T: Number of time steps
            initial_obs_1: Initial observation for agent 1
            initial_obs_2: Initial observation for agent 2
            
        Returns:
            Dictionary containing simulation results
        """
        actions_over_time = np.zeros((T, 2))
        B_over_time = np.zeros((T, 4, 4, 2, 2))
        q_pi_over_time = np.zeros((T, 2, 2))
        q_s_over_time = np.zeros((T, 4, 2))
        
        for t in range(T):
            action_1, action_2, B1, B2, q_pi_1, q_pi_2, qs_1, qs_2 = self.run_single_round(
                initial_obs_1, initial_obs_2, t
            )
            
            # Store results
            actions_over_time[t] = [action_1, action_2]
            B_over_time[t, :, :, :, 0] = B1
            B_over_time[t, :, :, :, 1] = B2
            q_pi_over_time[t, :, 0] = q_pi_1
            q_pi_over_time[t, :, 1] = q_pi_2
            q_s_over_time[t, :, 0] = qs_1[0]
            q_s_over_time[t, :, 1] = qs_2[0]
        
        return {
            'actions': actions_over_time,
            'transitions': B_over_time,
            'policies': q_pi_over_time,
            'states': q_s_over_time
        }


def construct_agents(lr_pB_1=0.1, lr_pB_2=0.1, factors_to_learn="all", 
                    action_selection="stochastic", alpha_1=1.0, alpha_2=1.0):
    """
    Construct two agents for simulation.
    
    Args:
        lr_pB_1: Learning rate for agent 1
        lr_pB_2: Learning rate for agent 2
        factors_to_learn: Which factors to learn
        action_selection: Action selection method
        alpha_1: Precision for agent 1
        alpha_2: Precision for agent 2
        
    Returns:
        Tuple of (agent_1, agent_2, D)
    """
    agent_1 = PrisonersDilemmaAgent(
        lr_pB=lr_pB_1, 
        factors_to_learn=factors_to_learn,
        action_selection=action_selection,
        alpha=alpha_1
    )
    
    agent_2 = PrisonersDilemmaAgent(
        lr_pB=lr_pB_2, 
        factors_to_learn=factors_to_learn,
        action_selection=action_selection,
        alpha=alpha_2
    )
    
    return agent_1, agent_2, agent_1.agent.D
