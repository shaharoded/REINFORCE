# Deep Reinforcement Learning - Assignment 2

Implementation and comparison of Policy Gradient algorithms (REINFORCE) on the CartPole-v1 task.

## Overview

This project implements the **REINFORCE** algorithm (Monte Carlo Policy Gradient) and its variant **REINFORCE with Baseline**. The goal is to solve the `CartPole-v1` environment, where the agent must balance a pole on a cart. The environment is considered solved when the average reward over 100 consecutive episodes is at least 475.0.

## Project Structure
```
Root/
├── src/
│   ├── init.py                         # Package initialization
│   ├── agent.py                        # Agent implementations
│   │   ├── Agent                       # Base class (Sampling logic)
│   │   ├── ReinforceAgent              # Vanilla REINFORCE
│   │   └── ReinforceBaselineAgent      # REINFORCE with Value Baseline
│   ├── ffnn.py                         # Neural network architectures
│   │   ├── PolicyNetwork               # Actor (Policy)
│   │   └── ValueNetwork                # Critic (Baseline)
│   └── utils.py                        # Utilities (Return calculation)
├── mainColab.ipynb                     # Main notebook for training and evaluation
├── results/                            # Training outputs (plots, summaries)
├── models/                             # Saved model checkpoints
└── requirements.txt                    # Python dependencies
```

## Algorithms

### 1. Vanilla REINFORCE
The standard Monte Carlo Policy Gradient algorithm.
- **Update Rule**: $\nabla J(\theta) \approx \sum \nabla \log \pi(a_t|s_t) \cdot G_t$
- Uses the actual discounted return $G_t$ as an unbiased but high-variance estimate of the gradient.

### 2. REINFORCE with Baseline
Improves upon vanilla REINFORCE by subtracting a state-dependent baseline $V(s)$ from the return.
- **Update Rule**: $\nabla J(\theta) \approx \sum \nabla \log \pi(a_t|s_t) \cdot (G_t - V(s_t))$
- **Advantage**: $(G_t - V(s_t))$ reduces variance without introducing bias.
- **Value Network**: A separate neural network learns to approximate $V(s)$ by minimizing MSE against $G_t$.

## Usage

Open `mainColab.ipynb` to run the full training and evaluation pipeline.
The notebook covers:
1.  Environment setup.
2.  Training Vanilla REINFORCE.
3.  Training REINFORCE with Baseline.
4.  Comparative analysis of convergence speed and stability.
