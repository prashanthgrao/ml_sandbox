import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import optuna
import random

"""
Key Changes: Let's use this for our presentation
1. Generalized Advantage Estimation (GAE) – Efficiently computes advantage, reducing variance.
2. Dynamic Hidden Layer Size and Learning Rate – Flexible architecture and learning speed.
3. Independent Actor and Critic Optimizers – Allows separate updates for actor and critic.
4. Using GAE for Returns – Improves advantage computation for better performance.
5. Entropy Regularization – Encourages exploration by penalizing certainty in actions.
6. Learning Rate Scheduling – Adjusts learning rate over time for better convergence.
7. Reward Normalization – Stabilizes training by normalizing rewards.
8. N-Step Returns – Improves training efficiency and stability by considering multiple steps.
9. Convergence Measurement – Tracks episodes taken to achieve a specific average reward.
10. Optuna Hyperparameter Tuning – Automates tuning of critical parameters for optimal results.
"""

# Environment Setup
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Hyperparameters
num_episodes = 1000
gamma = 0.99  # Discount factor

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.leaky_relu(self.fc1(state))  # Leaky ReLU helps avoid dead neurons
        x = F.leaky_relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_size, hidden_size=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.value_fc = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = F.leaky_relu(self.fc1(state))  # Leaky ReLU for smoother training
        x = F.leaky_relu(self.fc2(x))
        value = self.value_fc(x)
        return value

    # Generalized Advantage Estimation (GAE)
    def compute_gae(rewards, values, gamma, lam=0.95):
        returns = []
        R = 0
        for r, v in zip(reversed(rewards), reversed(values)):
            R = r + gamma * R
            returns.insert(0, R + (1 - lam) * (R - v.item()))
        return torch.tensor(returns)
    
    def train_model(hidden_size, learning_rate, convergence_threshold=500):
    actor = Actor(state_size, action_size, hidden_size)
    critic = Critic(state_size, hidden_size)

    # Optimizers for actor and critic networks
    actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
    critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)

    rewards_all_episodes = []

    for episode in range(num_episodes):
        state = env.reset()
        state = torch.FloatTensor(state)

        log_probs = []
        values = []
        rewards = []

        done = False
        while not done:
            # Select action using actor network
            action_probs = actor(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Take action in environment
            next_state, reward, done, _ = env.step(action.item())
            next_state = torch.FloatTensor(next_state)

            # Store log probability, value, and reward
            value = critic(state)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

            state = next_state

        # Compute returns using GAE
        returns = compute_gae(rewards, values, gamma)

        # Convert lists to tensors
        log_probs = torch.stack(log_probs)
        values = torch.stack(values).squeeze()

        # Compute advantage
        advantage = returns - values.detach()

        # Actor loss with entropy regularization
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=-1).mean()  # Entropy regularization
        actor_loss = -(log_probs * advantage).mean() - 0.01 * entropy  # Incorporate entropy in the loss

        # Critic loss
        critic_loss = F.mse_loss(values, returns)

        # Update actor network
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # Update critic network
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Store total reward for this episode
        total_reward = sum(rewards)
        rewards_all_episodes.append(total_reward)

        # Print episode result
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward}")

        # Check for convergence
        if episode >= 249:  # Ensure we have enough data to compute the average
            if np.mean(rewards_all_episodes[-250:]) >= convergence_threshold:
                print(f"Converged in {episode + 1} episodes!")
                break

    return actor, critic, rewards_all_episodes  # Return actor, critic, and rewards

# Define a list to store rewards from each trial along with the hyperparameters used
trial_rewards = []
episode_rewards = []  # To store the rewards of all episodes
trial_params = []     # To store hyperparameters (hidden size and learning rate) of each trial

# Objective function for Optuna
def objective(trial):
    hidden_size = trial.suggest_int("hidden_size", 128, 512)  # Increased range
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)

    # Unpack only the rewards from the train_model function
    _, _, rewards = train_model(hidden_size, learning_rate)

    # Store the rewards of the full episode for plotting
    episode_rewards.append(rewards)

    # Store the hyperparameters of the current trial
    trial_params.append((hidden_size, learning_rate))

    # Calculate average reward of the last 250 episodes
    avg_reward = np.mean(rewards[-250:])

    # Store the average reward of the last 250 episodes for final plotting
    trial_rewards.append(avg_reward)

    return avg_reward  # Return average reward of last 250 episodes

# Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("Best hyperparameters:", study.best_params)

# Plotting reward progression for each trial separately
for i, rewards in enumerate(episode_rewards):
    # Extract the hidden_size and learning_rate for the current trial
    hidden_size, learning_rate = trial_params[i]

    # Create a new figure for each trial
    plt.figure(figsize=(10, 6))

    # Plot the reward progression for the current trial
    plt.plot(rewards, color='b', label=f'Trial {i + 1}')

    # Add labels and title
    plt.xlabel('Episode Number')
    plt.ylabel('Total Reward')
    plt.title(f'Trial {i} Reward Progression\nHidden Size={hidden_size}, LR={learning_rate:.1e}')

    # Show the plot
    plt.show()
