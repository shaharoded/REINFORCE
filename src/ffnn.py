import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class PolicyNetwork(nn.Module):
    """
    Parametrized Policy Network (Actor).
    Maps state -> Probability distribution over actions (Logits).
    
    Uses a probability distribution over actions parameterized by theta.
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
    State-Value Function Approximation (Baseline).
    Maps state -> Scalar Value V(s).
    
    Used to reduce variance.
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