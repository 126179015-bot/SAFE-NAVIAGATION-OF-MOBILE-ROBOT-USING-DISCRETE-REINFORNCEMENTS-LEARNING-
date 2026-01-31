import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# =============================================
# ENVIRONMENT CLASS
# =============================================
class DiffDriveEnv:
    def __init__(self, world_size=(10, 10), goal=(9, 9), max_steps=300):
        self.world_size = world_size
        self.goal_pos = np.array(goal)
        self.max_steps = max_steps
        self.reset()

        # Example obstacles
        self.obstacles = [
            {"pos": np.array([5.0, 5.0]), "radius": 1.0},
            {"pos": np.array([2.0, 7.0]), "radius": 0.8},
        ]

    def reset(self):
        self.robot_pos = np.array([0.5, 0.5])
        self.steps = 0
        self.done = False
        info = {}
        return self._get_state(), info

    def _get_state(self):
        # State = [x, y, goal_x, goal_y]
        return np.concatenate((self.robot_pos, self.goal_pos))

    def step(self, action):
        """Action = 0: up, 1: down, 2: left, 3: right"""
        move = {
            0: np.array([0, 0.5]),
            1: np.array([0, -0.5]),
            2: np.array([-0.5, 0]),
            3: np.array([0.5, 0]),
        }[action]

        self.robot_pos += move
        self.robot_pos = np.clip(self.robot_pos, 0, self.world_size[0])
        self.steps += 1

        done = False
        info = {"success": False, "collision": False}

        # Check collision
        for obs in self.obstacles:
            if np.linalg.norm(self.robot_pos - obs["pos"]) < obs["radius"]:
                done = True
                info["collision"] = True

        # Check goal
        if np.linalg.norm(self.robot_pos - self.goal_pos) < 0.5:
            done = True
            info["success"] = True

        # Timeout
        if self.steps >= self.max_steps:
            done = True

        return self._get_state(), done, info

# =============================================
# DQN NETWORK
# =============================================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# =============================================
# REWARD FUNCTION (FIXED)
# =============================================
def compute_reward(env, done, info, prev_dist, steps):
    """
    Computes reward and always returns (reward, curr_dist)
    """
    curr_dist = np.linalg.norm(np.array(env.robot_pos) - np.array(env.goal_pos))
    reward = 0.0

    if done:
        if info.get("success"):
            reward = 500.0 - steps * 0.5  # Strong positive reward for reaching goal
        elif info.get("collision"):
            reward = -200.0  # Heavy penalty for collision
        else:
            reward = -50.0  # Timeout
        return reward, curr_dist  # ✅ Always return 2 values

    # Smooth distance-based shaping
    if prev_dist is not None:
        progress = prev_dist - curr_dist
        reward += progress * 10.0  # Bonus for moving closer

    # Small step penalty
    reward -= 0.05

    # Obstacle penalty
    min_obs_dist = float("inf")
    for obs in env.obstacles:
        obs_dist = np.linalg.norm(np.array(env.robot_pos) - np.array(obs["pos"]))
        min_obs_dist = min(min_obs_dist, obs_dist)
    if min_obs_dist < obs["radius"] * 2:
        reward -= 1.0

    return reward, curr_dist

# =============================================
# TRAINING FUNCTION
# =============================================
def train_improved_dqn(episodes=1000, gamma=0.99, epsilon=1.0, eps_min=0.1, eps_decay=0.995,
                       lr=1e-3, batch_size=64, memory_size=50000, save_interval=200):

    env = DiffDriveEnv()
    input_dim = 4
    output_dim = 4

    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    memory = deque(maxlen=memory_size)

    def act(state, eps):
        if random.random() < eps:
            return random.randint(0, output_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = policy_net(state)
        return torch.argmax(q_values).item()

    all_rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        prev_dist = np.linalg.norm(np.array(env.robot_pos) - np.array(env.goal_pos))
        steps = 0

        while not done:
            action = act(state, epsilon)
            next_state, done, info = env.step(action)
            reward, prev_dist = compute_reward(env, done, info, prev_dist, steps)
            total_reward += reward
            memory.append((state, action, reward, next_state, done))
            state = next_state
            steps += 1

            # Replay training
            if len(memory) > batch_size:
                batch = random.sample(memory, batch_size)
                s, a, r, ns, d = zip(*batch)

                s = torch.FloatTensor(s)
                a = torch.LongTensor(a).unsqueeze(1)
                r = torch.FloatTensor(r).unsqueeze(1)
                ns = torch.FloatTensor(ns)
                d = torch.FloatTensor(d).unsqueeze(1)

                q_values = policy_net(s).gather(1, a)
                next_q_values = target_net(ns).max(1)[0].unsqueeze(1)
                expected_q = r + (gamma * next_q_values * (1 - d))

                loss = criterion(q_values, expected_q.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Decay epsilon
        epsilon = max(eps_min, epsilon * eps_decay)
        all_rewards.append(total_reward)

        # Update target network every few episodes
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Save checkpoint
        if episode % save_interval == 0 and episode != 0:
            torch.save(policy_net.state_dict(), f"dqn_checkpoint_ep{episode}.pth")
            print(f"💾 Saved model checkpoint at episode {episode}")

        print(f"Episode {episode}/{episodes} | Reward: {total_reward:.2f} | Eps: {epsilon:.3f}")

    # Save final model
    torch.save(policy_net.state_dict(), "dqn_final_model.pth")
    print("✅ Training complete, model saved as dqn_final_model.pth")

    # Plot rewards
    plt.plot(all_rewards)
    plt.title("DQN Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()

# =============================================
# MAIN
# =============================================
if __name__ == "__main__":
    train_improved_dqn(episodes=1000, save_interval=200)
