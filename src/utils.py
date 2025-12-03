import torch
from typing import List

def calculate_returns(rewards: List[float], gamma: float, normalize: bool = True) -> torch.Tensor:
    """
    Calculates the discounted return (G_t) for every time step in the episode.
    
    Args:
        rewards: List of rewards received during the episode
        gamma: Discount factor
        normalize: Whether to normalize returns (standard practice for stability)
        
    Returns:
        Tensor of discounted returns
    """
    returns = []
    R = 0
    
    # Iterate backwards: G_t = r_t + gamma * G_{t+1}
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
        
    returns = torch.tensor(returns, dtype=torch.float32)
    
    if normalize and len(returns) > 1:
        # Normalize to mean 0 and std 1 to stabilize gradient descent
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
    return returns