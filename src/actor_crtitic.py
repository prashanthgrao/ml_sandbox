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
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set device for computations (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)

# Environment Setup
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Hyperparameters
num_episodes = 1000
gamma = 0.99  # Discount factor
batch_size = 64  # Batch size for mini-batch updates

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
    values = values + [0]  # Add an extra 0 for next value
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] - values[i]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    returns = torch.tensor(advantages) + values[:-1]
    return torch.tensor(advantages), returns

# Train model with GAE, entropy regularization, and Optuna hyperparameter tuning
def train_model(hidden_size, learning_rate, convergence_threshold=500):
    actor = Actor(state_size, action_size, hidden_size).to(device)
    critic = Critic(state_size, hidden_size).to(device)

    # Optimizers and Learning Rate Schedulers
    actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
    critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)
    scheduler_actor = torch.optim.lr_scheduler.StepLR(actor_optimizer, step_size=100, gamma=0.95)
    scheduler_critic = torch.optim.lr_scheduler.StepLR(critic_optimizer, step_size=100, gamma=0.95)

    rewards_all_episodes = []
    transitions = []  # Store transitions for mini-batch updates

    for episode in range(num_episodes):
        state = env.reset()
        state = torch.FloatTensor(state).to(device)

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
            next_state = torch.FloatTensor(next_state).to(device)

            # Store log probability, value, and reward
            value = critic(state)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

            state = next_state

        # Compute GAE and returns
        advantages, returns = compute_gae(rewards, values, gamma)

        # Normalize advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert lists to tensors
        log_probs = torch.stack(log_probs)
        values = torch.stack(values).squeeze()

        # Actor loss with entropy regularization
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=-1).mean()  # Entropy regularization
        actor_loss = -(log_probs * advantages.detach()).mean() - 0.01 * entropy

        # Critic loss
        critic_loss = F.mse_loss(values, returns)

        # Backpropagate actor and critic losses
        actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
        critic_optimizer.step()

        # Step learning rate scheduler
        scheduler_actor.step()
        scheduler_critic.step()

        # Store total reward for this episode
        total_reward = sum(rewards)
        rewards_all_episodes.append(total_reward)

        # Logging and convergence check
        if episode % 10 == 0:
            logger.info(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward}")
        if episode >= 249:  # Ensure we have enough data to compute the average
            if np.mean(rewards_all_episodes[-250:]) >= convergence_threshold:
                logger.info(f"Converged in {episode + 1} episodes!")
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

logger.info("Best hyperparameters: %s", study.best_params)

# Plotting reward progression for each trial separately
for i, rewards in enumerate(episode_rewards):
    hidden_size, learning_rate = trial_params[i]

    plt.figure(figsize=(10, 6))
    plt.plot(rewards, color='b', label=f'Trial {i + 1}')
    plt.xlabel('Episode Number')
    plt.ylabel('Total Reward')
    plt.title(f'Trial {i} Reward Progression\nHidden Size={hidden_size}, LR={learning_rate:.1e}')
    plt.show()

# Train the final model with the best hyperparameters
best_hidden_size = study.best_params['hidden_size']
best_learning_rate = study.best_params['learning_rate']

logger.info(f"Training final model with hidden_size={best_hidden_size} and learning_rate={best_learning_rate}...")
actor, critic, trained_rewards = train_model(best_hidden_size, best_learning_rate)

# Save and load model functions
def save_model(actor, critic, filename_actor, filename_critic):
    torch.save(actor.state_dict(), filename_actor)
    torch.save(critic.state_dict(), filename_critic)
    logger.info(f"Models saved as {filename_actor} and {filename_critic}")

def load_model(actor, critic, filename_actor, filename_critic):
    actor.load_state_dict(torch.load(filename_actor))
    critic.load_state_dict(torch.load(filename_critic))
    actor.eval()
    critic.eval()
    logger.info(f"Models loaded from {filename_actor} and {filename_critic}")
    return actor, critic

save_model(actor, critic, "actor.pth", "critic.pth")
actor, critic = load_model(actor, critic, "actor.pth", "critic.pth")

# Evaluate the final trained model
def evaluate_model(actor, num_episodes=10):
    total_rewards = []  # List to store the total reward for each episode
    actor.eval()  # Set the actor to evaluation mode

    with torch.no_grad():  # Disable gradient computation during evaluation
        for episode in range(num_episodes):
            state = env.reset()
            state = torch.FloatTensor(state).to(device)
            episode_reward = 0
            done = False

            while not done:
                action_probs = actor(state)
                action = torch.argmax(action_probs).item()
                next_state, reward, done, _ = env.step(action)

                state = torch.FloatTensor(next_state).to(device)
                episode_reward += reward

            total_rewards.append(episode_reward)
            logger.info(f"Evaluation Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")

    return np.mean(total_rewards), total_rewards

mean_reward, total_rewards = evaluate_model(actor)
logger.info(f"Final Evaluation - Mean Reward over {num_episodes} episodes: {mean_reward}")

# Plotting the evaluated rewards
plt.figure(figsize=(10, 6))
plt.plot(eval_rewards)
plt.xlabel('Evaluation Episode')
plt.ylabel('Total Reward')
plt.title(f'Evaluation of the Trained A2C Model on {env.spec.id}')
plt.show()

# Calculate and print the average reward during evaluation
avg_eval_reward = np.mean(eval_rewards)
print(f"Average reward over {num_eval_episodes} evaluation episodes: {avg_eval_reward}")
env.close()
