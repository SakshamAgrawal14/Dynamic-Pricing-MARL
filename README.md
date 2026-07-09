# Multi-Agent Reinforcement Learning for Dynamic Pricing

## Overview

This project investigates strategic pricing behaviour in a repeated two-firm market using Multi-Agent Reinforcement Learning (MARL). Two firms repeatedly compete by choosing prices, while reinforcement learning agents learn pricing strategies through repeated interaction with the market.

The project begins with a tabular Q-Learning implementation and extends it to Deep Q Networks (DQN) and Proximal Policy Optimization (PPO). Various market settings, including customer loyalty, heterogeneous demand and self-play, are explored to study equilibrium formation and strategic pricing behaviour.

---

## Problem Statement

We model a repeated pricing game involving two competing firms.

- Each firm selects a price from a discrete action space.
- The lower-priced firm captures the market.
- Equal prices split demand equally.
- Firms seek to maximise cumulative long-term profit.

The objective is to understand how reinforcement learning agents learn pricing strategies and how different market conditions influence the resulting equilibria.

---

## Features

- Two-firm repeated pricing environment
- Tabular Q-Learning implementation
- Deep Q Network (DQN)
- Proximal Policy Optimization (PPO)
- Baseline opponent strategies
  - Always50
  - Always70
  - Random
  - Tit-for-Tat
- Customer loyalty experiments
- Heterogeneous demand experiments
- Tabular self-play
- Alternating DQN self-play
- Alternating PPO self-play

---

## Project Structure

```
MARL PRO1/
│
├── Agent.py                    # Tabular Q-learning agent
├── Env.py                      # Pricing environment
├── baseline.py                 # Baseline strategies
├── train.py                    # Tabular Q-learning training
│
├── pricing_env.py              # Gymnasium wrapper for deep RL
│
├── train_dqn.py
├── evaluate_dqn.py
├── pricing_dqn.zip             # Trained DQN against fixed-strategy opponents
│
├── train_ppo.py
├── evaluate_ppo.py
├── pricing_ppo.zip             # Trained PPO against fixed-strategy opponents
│
├── frozen_rl_agent.py          # Frozen opponent used for alternating self-play
├── train_selfplay.py           # Alternating DQN/PPO self-play training
├── evaluate_selfplay.py        # Evaluation of alternating self-play
│
├── agent_A.zip                 # DQN Agent A after alternating self-play
├── agent_B.zip                 # DQN Agent B after alternating self-play
├── ppo_agent_A.zip             # PPO Agent A after alternating self-play
├── ppo_agent_B.zip             # PPO Agent B after alternating self-play
│
└── README.md
```

---

## Algorithms Implemented

### Tabular Q-Learning

Learns state-action values using a Q-table. Suitable for the compact state space used in this project.

### Deep Q Network (DQN)

Uses a neural network to approximate the Q-function, demonstrating how value-function approximation can replace tabular representations.

### Proximal Policy Optimization (PPO)

A policy-gradient reinforcement learning algorithm used to compare policy-based learning with value-based methods.

---

## Experiments

### Fixed Opponents

The RL agents were trained and evaluated against:

- Always50
- Always70
- Random
- Tit-for-Tat

### Market Variations

Experiments were conducted under different market settings:

- Winner-takes-all demand
- Customer loyalty
- Heterogeneous demand
- Alternating self-play

---

## Results

### Fixed Opponent Comparison

| Algorithm | Always50 | Always70 | Random | Tit-for-Tat |
|-----------|----------|----------|--------|-------------|
| Tabular Q-Learning | Learns arbitrary action (70 observed), Avg Reward = 0 | Learns 65, Avg Reward = 15 | Learns 60, Avg Reward ≈ 5.07 | Learns 70, Avg Reward = 10 |
| DQN | Learns arbitrary action (70 observed), Avg Reward = 0 | Learns 65, Avg Reward = 15 | Learns 60, Avg Reward ≈ 5.07 | Learns 70, Avg Reward = 10 |
| PPO | Learns arbitrary action (70 observed), Avg Reward = 0 | Learns 65, Avg Reward ≈ 14.9 | Learns 60, Avg Reward ≈ 5 | Learns 70, Avg Reward = 10 |

### Self-Play

Tabular Q-Learning:
- Frequently converged to (70,70)
- Occasionally converged to (65,65)
- Oscillatory behaviour was also observed

Alternating DQN Self-Play:
- Initial training against Always70 converged to (55,55)
- Initial training against Random converged to (60,60)

Alternating PPO Self-Play:
- Initial training against Random converged to (55,55)

---

## Key Observations

- Against fixed-strategy opponents, Tabular Q-Learning, DQN and PPO learned nearly identical pricing strategies.
- The compact state space made tabular Q-Learning sufficient for this environment, with DQN and PPO providing similar performance.
- Customer loyalty encouraged higher equilibrium prices.
- Heterogeneous demand produced asymmetric pricing outcomes.
- Self-play experiments showed that different training procedures and initial opponents could lead to different stable pricing equilibria.

---

## Technologies Used

- Python
- NumPy
- Gymnasium
- Stable-Baselines3
- PyTorch

---
