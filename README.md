# NOC Optimization Using Reinforcement Learning

## Introduction
This project applies the Twin Delayed Deep Deterministic Policy Gradient (TD3) reinforcement learning algorithm to optimize network operations. The goal is to minimize latency and maximize bandwidth based on simulator output.

## Project Structure
```bash
/noc/
│
├── src/                    # Source files for the TD3 algorithm and environment interaction
│   ├── environment.py      # Environment definition for the network operations
│   ├── td3.py              # TD3 algorithm implementation
│   └── main.py             # Main script to run the training process
│
├── pseudocode/             # Pseudocode for calculating metrics
│   ├── avg_latency.txt     # Pseudocode to measure average latency
│   └── avg_bandwidth.txt   # Pseudocode to measure average bandwidth
│
├── requirements.txt        # Python dependencies required
└── README.md               # Documentation and instructions
```

## Environment Setup

1. Create a Python Virtual Environment: It is recommended to use a virtual environment to avoid conflicts with system-wide packages. Use the following commands to create and activate a virtual environment:

```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

2. Install Dependencies: Once the virtual environment is activated, install the required Python packages:

```bash
pip install -r requirements.txt
```

## Running the Code
To start the simulation and RL training, use the following command:

```bash
cd src
python main.py
```

This will initiate the NOC simulation and start the RL agent's training process based on the TD3 algorithm. Outputs will include latency and bandwidth statistics and model training progress.

# Algorithm Design

## States/Behaviors

The state of the environment includes:

- Average Latency: Computed as the mean latency of packets.
- Bandwidth Usage: Current bandwidth usage.
- Buffer Occupancy: Percentage of buffer capacity utilized.
- Throttling State: Boolean indicating whether throttling is currently applied.

## Actions

Actions include adjustments to the network settings:

- Adjust CPU Buffer Size
- Adjust IO Buffer Size
- Adjust CPU Weight
- Adjust Operating Frequency

## Rewards
The reward function is designed to:

- Penalize high latency above a threshold.
- Reward bandwidth usage close to capacity.
- Penalize excessive or insufficient buffer occupancy.
- Penalize frequent throttling adjustments.

## Algorithm Choice
TD3 (Twin Delayed Deep Deterministic Policy Gradient) is chosen for its effectiveness in handling high-dimensional, continuous action spaces and its ability to mitigate problems like overestimation bias found in other algorithms like DQN.

## Pseudocode for Metrics Calculation
Pseudocode for calculating average latency and bandwidth is provided in the pseudocode/ directory. These documents outline efficient and robust methods for deriving these metrics from the simulated network operations.