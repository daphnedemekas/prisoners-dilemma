# Prisoner's Dilemma Simulation with Active Inference

A simulation framework for studying the Prisoner's Dilemma using active inference agents. The project explores how agents with different learning rates and parameters evolve their strategies over time, both in pairwise interactions and in network settings.

## Overview

The Prisoner's Dilemma is a fundamental game theory problem that explores cooperation vs. defection strategies. This implementation uses active inference agents that learn and adapt their strategies based on their interactions with other agents.

### Key Features

- **Active Inference Agents**: Agents that use Bayesian inference to learn about their environment and other agents
- **Network Simulations**: Multi-agent simulations on network topologies
- **Parameter Sweeps**: Systematic exploration of learning rates and other parameters
- **Comprehensive Visualization**: Multiple plot types for analyzing agent behavior
- **Modular Design**: Clean, well-organized code structure

## Project Structure

```
src/
├── models/                 # Core agent and network models
│   ├── agent.py           # Prisoner's Dilemma agent implementation
│   └── network.py         # Network-based simulations
├── simulation/            # Simulation and parameter sweep modules
│   └── sweep.py          # Parameter sweep functionality
├── visualization/         # Plotting and visualization
│   └── plots.py          # Comprehensive plotting functions
├── main.py               # Main script to run all simulations
└── dual_agent_results/   # Legacy results and analysis
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd prisoners-dilemma
```

2. Install dependencies:
```bash
pip install numpy matplotlib networkx pymdp imageio
```

## Usage

### Running All Simulations

To run all simulations and generate all figures:

```bash
cd src
python main.py
```

This will create:
- `figures/` directory with all generated plots
- `results/` directory with simulation data

## Citation

If you use this code in your research, please cite:

```bibtex
@software{prisoners_dilemma_simulation,
  title={Prisoner's Dilemma Simulation with Active Inference},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/prisoners-dilemma}
}
```
