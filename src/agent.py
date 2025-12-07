from random import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, Any, Optional
import gymnasium as gym

# Local imports
from src.ffnn import PolicyNetwork, ValueNetwork, ActorNetwork, CriticNetwork
from src.utils import calculate_returns

class Agent:
    """
    Base class for Policy Gradient agents implementing the REINFORCE algorithm structure.
    
    The REINFORCE algorithm (Monte Carlo Policy Gradient) follows these general steps:
    1.  **Sampling**: Generate an episode trajectory following the current policy pi_theta.
    2.  **Evaluation**: Calculate returns G_t for each step in the episode.
    3.  **Improvement**: Update policy parameters theta using gradient ascent on J(theta).
    
    This base class handles Step 1 (in `train` and `select_action`) and the loop structure.
    Subclasses must implement Step 3 (in `update`).
    """
    def __init__(self, state_dim: int, action_dim: int, config: Optional[Dict[str, Any]] = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or {}
        
        # Hyperparameters
        self.gamma = self.config.get('gamma', 0.99)
        self.lr = self.config.get('learning_rate', 0.001)
        self.hidden_dims = self.config.get('hidden_dims', [128, 128])
        
        # Policy Network (The Actor) - Common to all PG agents
        self.policy_net = PolicyNetwork(state_dim, action_dim, self.hidden_dims)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        # Episode memory
        self.log_probs = []  # Stores log pi(a|s)
        self.rewards = []    # Stores rewards
        self.values = []     # Stores V(s) - optional, used by Baseline agent
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Selects an action by sampling from the stochastic policy pi(a|s).
        
        Algorithm Step: **Sampling**
        - Computes logits from the policy network.
        - Creates a categorical distribution.
        - Samples an action.
        - If training, stores log pi(a_t|s_t) for the gradient update later.
        """
        state_t = torch.FloatTensor(state)
        logits = self.policy_net(state_t)
        
        # Create probability distribution (Softmax handled internally)
        dist = Categorical(logits=logits)
        action = dist.sample()
        
        if training:
            # Store log probability for gradient calculation: log pi(a_t|s_t)
            self.log_probs.append(dist.log_prob(action))
            
            # Hook for Baseline agent to store Value estimates
            self._on_action_selected(state_t)
            
        return action.item()

    def _on_action_selected(self, state_tensor: torch.Tensor):
        """
        Hook called after an action is selected during training.
        
        Purpose:
        Allows subclasses (like ReinforceBaselineAgent) to perform additional operations 
        at each timestep, such as storing Value function estimates V(s_t), without 
        modifying the main `select_action` logic.
        """
        pass

    def update(self) -> float:
        """
        Performs the policy gradient update at the end of the episode.
        
        Algorithm Step: **Improvement**
        This method is 'unimplemented' in the base class because the specific update rule 
        differs between Vanilla REINFORCE and REINFORCE with Baseline.
        
        Raises:
            NotImplementedError: This is an abstract method that must be defined by subclasses.
        """
        raise NotImplementedError

    def train(self, env: gym.Env, max_episodes: int = 1000, target_reward: float = 475.0, window: int = 100) -> Dict[str, Any]:
        """
        Executes the main training loop for Episodic Policy Gradient.
        
        Algorithm Flow:
        1.  Reset environment.
        2.  **Sampling**: Run one full episode using `select_action`, storing rewards and log probs.
        3.  **Evaluation & Improvement**: Call `self.update()` to calculate returns and update weights.
        4.  Repeat until max_episodes or convergence criteria met.
        """
        stats = {'rewards': [], 'loss': [], 'episodes_trained': 0, 'converged': False}
        
        # Define a Learning Rate Scheduler
        # Will avoid the learning rate being too high for too long, which might cause later steps instability
        scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.9)
        
        for episode in range(1, max_episodes + 1):
            state, _ = env.reset()
            episode_reward = 0
            
            # 1. Generate Episode
            while True:
                action = self.select_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                self.rewards.append(reward)
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # 2. Update Policy (at the end of episode)
            loss = self.update()

            # 3. Step the scheduler
            scheduler.step()
            
            # 4. Logging and Cleanup
            stats['rewards'].append(episode_reward)
            if loss is not None:
                stats['loss'].append(loss)
            
            # Reset memory buffers
            self.log_probs = []
            self.rewards = []
            self.values = []
            
            # Check convergence
            if len(stats['rewards']) >= window:
                avg_reward = np.mean(stats['rewards'][-window:])
                if avg_reward >= target_reward:
                    print(f"\nConverged at episode {episode} with average reward {avg_reward:.2f}!")
                    stats['converged'] = True
                    stats['episodes_trained'] = episode
                    break
            
            stats['episodes_trained'] = episode
            
            if episode % 50 == 0:
                avg_r = np.mean(stats['rewards'][-min(episode, window):])
                print(f"Episode {episode} | MA Reward ({window} episodes): {avg_r:.2f}")
                
        return stats


class ReinforceAgent(Agent):
    """
    Vanilla REINFORCE Agent.
    
    Update Rule:
    grad(J) approx sum( grad(log pi(a_t|s_t)) * G_t )
    
    Where G_t is the discounted return from time t.
    """
    def update(self) -> float:
        """
        Updates the policy network using the Monte-Carlo return G_t as the weight.
        """
        # Calculate discounted returns G_t
        returns = calculate_returns(self.rewards, self.gamma)
        
        policy_loss = []
        
        # Calculate loss: - log_prob * G_t
        # (We use negative because we want Gradient Ascent, but optimizers do Descent)
        for log_prob, G_t in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G_t)
            
        # Sum losses and backpropagate
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


class ReinforceBaselineAgent(Agent):
    """
    REINFORCE with Baseline Agent.
    
    Update Rule:
    grad(J) approx sum( grad(log pi(a_t|s_t)) * (G_t - V(s_t)) )
    
    Where (G_t - V(s_t)) is the Advantage estimate.
    Also updates the Value network V to minimize (V(s_t) - G_t)^2.
    """
    def __init__(self, state_dim: int, action_dim: int, config: Optional[Dict[str, Any]] = None):
        super().__init__(state_dim, action_dim, config)
        
        # Initialize Value Network (V_v) to approximate V_pi 
        self.value_net = ValueNetwork(state_dim, self.hidden_dims)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)

    def _on_action_selected(self, state_tensor: torch.Tensor):
        """Store V(s) for the current state."""
        value = self.value_net(state_tensor)
        self.values.append(value)

    def update(self) -> float:
        # Calculate returns G_t (Actual return)
        returns = calculate_returns(self.rewards, self.gamma)
        
        policy_loss = []
        value_loss = []
        
        for log_prob, value, G_t in zip(self.log_probs, self.values, returns):
            # Advantage = G_t - V(s)
            # .detach(): We don't backpropagate policy gradients into the Value Net
            # We treat V(s) as a constant baseline for the policy update.
            advantage = G_t - value.detach()
            
            # Policy Loss: - log_prob * Advantage
            policy_loss.append(-log_prob * advantage)
            
            # Value Loss: MSE between V(s) and G_t (Make V predict G better)
            # Note: We use 'value' (with gradients) here because we DO want to update the Value Net
            value_loss.append(F.mse_loss(value, torch.tensor([G_t])))

        # Optimize Policy
        self.optimizer.zero_grad()
        p_loss = torch.stack(policy_loss).sum()
        p_loss.backward()
        self.optimizer.step()
        
        # Optimize Value Network (Baseline)
        self.value_optimizer.zero_grad()
        v_loss = torch.stack(value_loss).sum()
        v_loss.backward()
        self.value_optimizer.step()
        
        return p_loss.item() + v_loss.item()
    
class ActorCriticAgent(Agent):
    def __init__(self, state_dim: int, action_dim: int, config: Optional[Dict[str, Any]] = None):
        super().__init__(state_dim, action_dim, config)
        
        # Hyperparameters
        self.lr_actor = self.config.get('learning_rate_actor', 5e-4)
        self.lr_critic = self.config.get('learning_rate_critic', 1e-3)
        self.entropy_beta = self.config.get('entropy_beta', 0.01)
        
        # --- CHANGE 1: Initialize separate Actor and Critic networks ---
        self.actor = ActorNetwork(state_dim, action_dim, self.hidden_dims)
        self.critic = CriticNetwork(state_dim, self.hidden_dims)
        
        # --- CHANGE 2: Create separate optimizers ---
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

    def update(self) -> float:
        """
        Not used in One-Step Actor-Critic as updates happen inside the training loop.
        """
        pass

    def train(self, env: gym.Env, max_episodes: int = 1000, target_reward: float = 475.0, window: int = 100) -> Dict[str, Any]:
        stats = {'rewards': [], 'loss': [], 'episodes_trained': 0, 'converged': False}
        
        for episode in range(1, max_episodes + 1):
            state, _ = env.reset()
            # Fix: Add batch dimension
            state = torch.FloatTensor(state).unsqueeze(0)
            
            episode_reward = 0
            I = 1.0 
            episode_losses = []
            
            while True:
            # --- 1. Actor Step ---
                logits = self.actor(state)
                dist = Categorical(logits=logits)
                action = dist.sample()

                # --- 2. Environment Step ---
                next_state_np, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                next_state = torch.FloatTensor(next_state_np).unsqueeze(0)

                # --- 3. Critic Step ---
                value_s = self.critic(state) # Shape [1, 1]

                with torch.no_grad():
                    if done:
                        value_next_s = torch.tensor([[0.0]])
                    else:
                        value_next_s = self.critic(next_state)
                    
                    # Target is [1, 1]
                    target = reward + self.gamma * value_next_s

                # Calculate Delta (Advantage)
                # We detach value_s here to get a clean scalar/tensor for the Actor
                delta = target - value_s

                # --- 4. Update Critic ---
                # Compare [1, 1] to [1, 1]. NO WRAPPING.
                critic_loss = I * F.mse_loss(value_s, target)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                # --- 5. Update Actor ---
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()

                # We use delta.detach() so the Actor doesn't try to update the Critic
                actor_loss = - I * ((delta.detach() * log_prob) + (self.entropy_beta * entropy))

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Record total loss
                episode_losses.append(critic_loss.item() + actor_loss.item())

                # --- 6. Update I and State ---
                I *= self.gamma
                episode_reward += reward
                state = next_state

                if done:
                    break
            
            stats['rewards'].append(episode_reward)
            if episode_losses:
                stats['loss'].append(np.mean(episode_losses))
            
            # Check convergence
            if len(stats['rewards']) >= window:
                avg_reward = np.mean(stats['rewards'][-window:])
                if avg_reward >= target_reward:
                    print(f"\nConverged at episode {episode} with average reward {avg_reward:.2f}!")
                    stats['converged'] = True
                    stats['episodes_trained'] = episode
                    break
            
            stats['episodes_trained'] = episode
            
            if episode % 50 == 0:
                avg_r = np.mean(stats['rewards'][-min(episode, window):])
                print(f"Episode {episode} | MA Reward ({window} episodes): {avg_r:.2f}")
                
        return stats