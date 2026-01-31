import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import math

# ============================================================
# ULTRASONIC SENSOR SYSTEM (SAME AS DQN)
# ============================================================

class UltrasonicSensorArray:
    """8-ray ultrasonic sensor with light beam detection"""

    def __init__(self, num_rays=8, max_range=3.0):
        self.num_rays = num_rays
        self.max_range = max_range
        self.angles = [i * (2 * math.pi / num_rays) for i in range(num_rays)]

    def scan(self, robot_pos, obstacles, world_size):
        """Scan and return 8 distance readings (normalized 0-1)"""
        readings = []

        for angle_offset in self.angles:
            min_dist = self.max_range

            for obs in obstacles:
                obs_dist = np.linalg.norm(robot_pos - obs["pos"]) - obs["radius"]
                if obs_dist < min_dist and obs_dist > 0:
                    min_dist = obs_dist

            for axis in [0, 1]:
                for boundary in [0, world_size]:
                    boundary_dist = abs(boundary - robot_pos[axis])
                    if boundary_dist < min_dist:
                        min_dist = boundary_dist

            readings.append(min_dist / self.max_range)

        return np.array(readings, dtype=np.float32)


# ============================================================
# ENVIRONMENT WITH SENSORS (SAME AS DQN - FIXED)
# ============================================================

class DiffDriveEnvWithSensors:
    """Environment with proper 15D state"""

    def __init__(self, world_size=10, goal=(9, 9), max_steps=400):
        self.world_size = world_size
        self.goal_pos = np.array(goal, dtype=np.float32)
        self.max_steps = max_steps

        self.obstacles = [
            {"pos": np.array([5.0, 5.0]), "radius": 1.0},
            {"pos": np.array([2.0, 7.0]), "radius": 0.8},
            {"pos": np.array([7.0, 3.0]), "radius": 0.7},
        ]

        self.sensors = UltrasonicSensorArray(num_rays=8, max_range=3.0)
        self.reset()

    def reset(self):
        self.robot_pos = np.array([0.5, 0.5], dtype=np.float32)
        self.steps = 0
        self.done = False
        self.prev_distance = np.linalg.norm(self.robot_pos - self.goal_pos)
        return self._get_state(), {}

    def _get_state(self):
        """
        FIXED STATE (15D EXACTLY):
        [x, y, goal_x, goal_y] (4)
        + [sensor_1...sensor_8] (8)
        + [goal_dist, goal_angle, cos_angle] (3)
        = 15 dimensions total
        """
        x_norm = self.robot_pos[0] / self.world_size
        y_norm = self.robot_pos[1] / self.world_size
        goal_x_norm = self.goal_pos[0] / self.world_size
        goal_y_norm = self.goal_pos[1] / self.world_size

        sensor_data = self.sensors.scan(self.robot_pos, self.obstacles, self.world_size)

        dx = self.goal_pos[0] - self.robot_pos[0]
        dy = self.goal_pos[1] - self.robot_pos[1]
        goal_dist = np.linalg.norm([dx, dy]) / (self.world_size * math.sqrt(2))
        goal_angle = math.atan2(dy, dx) / math.pi
        cos_angle = math.cos(math.atan2(dy, dx)) if (dx != 0 or dy != 0) else 0.0

        state = np.array([
            x_norm, y_norm, goal_x_norm, goal_y_norm,
            sensor_data[0], sensor_data[1], sensor_data[2], sensor_data[3],
            sensor_data[4], sensor_data[5], sensor_data[6], sensor_data[7],
            goal_dist, goal_angle, cos_angle
        ], dtype=np.float32)

        assert state.shape[0] == 15, f"State size is {state.shape[0]}, expected 15!"
        return state

    def step(self, action):
        """Action = 0: up, 1: down, 2: left, 3: right"""

        move = {
            0: np.array([0, 0.5]),
            1: np.array([0, -0.5]),
            2: np.array([-0.5, 0]),
            3: np.array([0.5, 0]),
        }[action]

        self.robot_pos += move
        self.robot_pos = np.clip(self.robot_pos, 0, self.world_size)

        self.steps += 1
        done = False
        info = {"success": False, "collision": False}

        for obs in self.obstacles:
            if np.linalg.norm(self.robot_pos - obs["pos"]) < obs["radius"]:
                done = True
                info["collision"] = True
                break

        if np.linalg.norm(self.robot_pos - self.goal_pos) < 0.5:
            done = True
            info["success"] = True

        if self.steps >= self.max_steps:
            done = True

        return self._get_state(), done, info


# ============================================================
# DUELING DQN NETWORK (DDQN SPECIFIC)
# ============================================================

class DuelingDQN(nn.Module):
    """
    Dueling Double DQN:
    - Dueling architecture: Value + Advantage streams
    - Shared feature extraction
    - Separate streams for state value and action advantages
    """

    def __init__(self, input_dim=15, output_dim=4):
        super(DuelingDQN, self).__init__()

        # Shared feature extraction
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Value stream (predicts state value V(s))
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Advantage stream (predicts action advantages A(s,a))
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        """
        Dueling formula:
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,)))

        This separates state evaluation from action preference
        """
        features = self.feature_layer(x)

        # Value stream output
        value = self.value_stream(features)

        # Advantage stream output
        advantages = self.advantage_stream(features)

        # Combine: Q = V + (A - mean(A))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values


# ============================================================
# TRAINING - 10,000 EPISODES (FIXED + DOUBLE DQN)
# ============================================================

def train_ddqn_ultrasonic(episodes=10000, gamma=0.99, epsilon=1.0, 
                          eps_min=0.05, eps_decay=0.9995,
                          lr=1e-3, batch_size=64, memory_size=100000,
                          save_interval=500):
    """
    ✅ FIXED DDQN Training with:
    - Dueling architecture for better stability
    - Double DQN to reduce overestimation
    - Proper 15D state
    - 10,000 episodes
    """

    print(f"\n{'='*70}")
    print(f"🚀 DDQN Training with Ultrasonic Sensors (FIXED)")
    print(f"{'='*70}")
    print(f"Episodes: {episodes}")
    print(f"Batch Size: {batch_size}")
    print(f"State Dimension: 15 (verified)")
    print(f"Architecture: Dueling Double DQN")
    print(f"Network: 15 → 256 (shared) → [128→1] + [128→4]")
    print(f"{'='*70}\n")

    env = DiffDriveEnvWithSensors()

    # Verify state size
    test_state, _ = env.reset()
    print(f"✅ State size verified: {len(test_state)}D")
    assert len(test_state) == 15, f"State size {len(test_state)} != 15!"

    # Create networks
    policy_net = DuelingDQN(input_dim=15, output_dim=4)
    target_net = DuelingDQN(input_dim=15, output_dim=4)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    criterion = nn.SmoothL1Loss()

    memory = deque(maxlen=memory_size)

    def act(state, eps):
        if random.random() < eps:
            return random.randint(0, 3)

        state_tensor = torch.FloatTensor(np.array([state]))
        with torch.no_grad():
            q_values = policy_net(state_tensor)
        return torch.argmax(q_values).item()

    all_rewards = []
    success_count = 0
    best_success_rate = 0

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        prev_dist = np.linalg.norm(env.robot_pos - env.goal_pos)
        steps = 0

        while not done:
            action = act(state, epsilon)
            next_state, done, info = env.step(action)

            # Compute reward (same as DQN)
            curr_dist = np.linalg.norm(env.robot_pos - env.goal_pos)
            reward = 0.0

            if done:
                if info.get("success"):
                    reward = 1000.0 - steps * 0.5
                elif info.get("collision"):
                    reward = -500.0
                else:
                    reward = -100.0
            else:
                progress = (prev_dist - curr_dist) * 100.0
                reward += progress

                if curr_dist < 2.0:
                    reward += (2.0 - curr_dist) * 50.0

                sensor_data = env.sensors.scan(env.robot_pos, env.obstacles, env.world_size)
                min_sensor = np.min(sensor_data)
                if min_sensor < 0.3:
                    reward -= 10.0 * (0.3 - min_sensor)

                reward -= 0.1
                prev_dist = curr_dist

            total_reward += reward
            memory.append((state, action, reward, next_state, done))
            state = next_state
            steps += 1

            # FIXED: Proper batch conversion
            if len(memory) > batch_size:
                batch = random.sample(memory, batch_size)

                states_batch = np.array([b[0] for b in batch])
                actions_batch = np.array([b[1] for b in batch])
                rewards_batch = np.array([b[2] for b in batch])
                next_states_batch = np.array([b[3] for b in batch])
                dones_batch = np.array([b[4] for b in batch])

                s = torch.FloatTensor(states_batch)
                a = torch.LongTensor(actions_batch).unsqueeze(1)
                r = torch.FloatTensor(rewards_batch).unsqueeze(1)
                ns = torch.FloatTensor(next_states_batch)
                d = torch.FloatTensor(dones_batch).unsqueeze(1)

                # DOUBLE DQN: Select action with policy_net, evaluate with target_net
                with torch.no_grad():
                    next_actions = policy_net(ns).argmax(dim=1).unsqueeze(1)
                    next_q_values = target_net(ns).gather(1, next_actions)

                q_values = policy_net(s).gather(1, a)
                expected_q = r + (gamma * next_q_values * (1 - d))

                loss = criterion(q_values, expected_q.detach())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

        epsilon = max(eps_min, epsilon * eps_decay)
        all_rewards.append(total_reward)

        if info.get("success"):
            success_count += 1

        # Update target network
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Save checkpoints
        if episode % save_interval == 0 and episode > 0:
            torch.save(policy_net.state_dict(), f"ddqn_sensor_ep{episode}.pth")

            recent_success = (success_count / (episode + 1)) * 100
            if recent_success > best_success_rate:
                best_success_rate = recent_success
                torch.save(policy_net.state_dict(), "ddqn_sensor_BEST.pth")
                print(f"✨ New best! Success: {recent_success:.1f}% (Ep {episode})")

        if episode % 100 == 0 and episode > 0:
            recent_reward = np.mean(all_rewards[-100:])
            recent_success = (success_count / (episode + 1)) * 100
            print(f"Ep {episode:5d}/{episodes} | Reward: {recent_reward:7.2f} | "
                  f"Success: {recent_success:5.1f}% | ε: {epsilon:.4f}")

        scheduler.step()

    torch.save(policy_net.state_dict(), "ddqn_sensor_final.pth")

    print(f"\n{'='*70}")
    print(f"✅ DDQN Training Complete!")
    print(f"Total Episodes: {episodes}")
    print(f"Final Success Rate: {(success_count/episodes)*100:.1f}%")
    print(f"Best Success Rate: {best_success_rate:.1f}%")
    print(f"\nSaved Models:")
    print(f"  ✓ ddqn_sensor_BEST.pth (best success: {best_success_rate:.1f}%)")
    print(f"  ✓ ddqn_sensor_final.pth (after 10K episodes)")
    print(f"{'='*70}\n")

    # Plot
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(all_rewards, alpha=0.3, label='Episode Reward', color='red')
    if len(all_rewards) >= 100:
        moving_avg = np.convolve(all_rewards, np.ones(100)/100, mode='valid')
        plt.plot(range(99, len(all_rewards)), moving_avg, 'darkred', linewidth=2, label='100-ep MA')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title('DDQN Training with Ultrasonic Sensors (10,000 Episodes)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    success_history = []
    for i in range(100, len(all_rewards), 100):
        success_history.append((success_count / i) * 100)
    plt.plot(success_history, 'darkred', linewidth=2, marker='s')
    plt.axhline(y=best_success_rate, color='r', linestyle='--', label=f'Best: {best_success_rate:.1f}%')
    plt.xlabel('Episode (x100)', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.title('Learning Progress', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ddqn_ultrasonic_training.png', dpi=150)
    print("📊 Plot saved: ddqn_ultrasonic_training.png")
    plt.show()

    return policy_net


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔧 FIXED DDQN Training with Ultrasonic Sensors")
    print("="*70)
    print("\nFixes Applied (same as DQN):")
    print("  ✅ State size verified as exactly 15D")
    print("  ✅ Proper numpy array batch conversion")
    print("  ✅ Tensor shape mismatch resolved")
    print("\nDDQN Specific Features:")
    print("  ✅ Dueling architecture (Value + Advantage streams)")
    print("  ✅ Double DQN (reduces overestimation bias)")
    print("  ✅ AdamW optimizer with weight decay")
    print("  ✅ Better stability than standard DQN")
    print("\nStarting training...")
    print("="*70 + "\n")

    model = train_ddqn_ultrasonic(episodes=10000, save_interval=500)
    print("\n✅ DDQN Training complete! Model saved.")
