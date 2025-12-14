import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class PolicyNetwork(nn.Module):
    """
    Parametrized Policy Network (Actor).
    
    Architecture:
        Input: State vector (dimension: state_dim)
        Hidden: Fully connected layers with ReLU activation
        Output: Logits for each action (dimension: action_dim)
    
    Usage:
        - In REINFORCE: Represents the stochastic policy pi(a|s).
        - In Actor-Critic: Represents the Actor.
        
    Alias: ActorNetwork
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128]):
        super(PolicyNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Returns logits. The Softmax is usually applied implicitly 
        # by the Categorical distribution in the agent.
        return self.network(state)

class ValueNetwork(nn.Module):
    """
    State-Value Function Approximation (Baseline/Critic).
    
    Architecture:
        Input: State vector (dimension: state_dim)
        Hidden: Fully connected layers with ReLU activation
        Output: Scalar value V(s) (dimension: 1)
        
    Usage:
        - In REINFORCE with Baseline: Approximates V(s) to compute Advantage.
        - In Actor-Critic: Represents the Critic.
        
    Alias: CriticNetwork
    """
    def __init__(self, state_dim: int, hidden_dims: List[int] = [128, 128]):
        super(ValueNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, 1)) # Output is a single scalar
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)