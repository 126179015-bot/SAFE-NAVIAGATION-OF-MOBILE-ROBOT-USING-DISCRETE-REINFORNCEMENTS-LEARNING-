import pygame
import torch
import numpy as np
import math
import random
import time
from collections import deque

# ---------------------------
# 🎨 Enhanced Configuration
# ---------------------------
WIDTH, HEIGHT = 1400, 800
GAME_AREA_WIDTH = 650
STATS_PANEL_WIDTH = 750
GOAL_POS = np.array([580, 730])
EPISODES_PER_MODEL = 20
ROBOT_RADIUS = 15
OBSTACLE_RADIUS = 18
SENSOR_FLASH_FREQ = 0.15
TRAIL_LENGTH = 30
GRID_SIZE = 50

# ---------------------------
# 🎨 Color Palette
# ---------------------------
COLORS = {
    'bg': (18, 18, 24),
    'grid': (35, 35, 45),
    'panel_bg': (25, 25, 32),
    'text': (220, 220, 230),
    'text_dim': (140, 140, 150),
    'goal': (50, 255, 150),
    'goal_glow': (100, 255, 200),
    'obstacle': (255, 70, 70),
    'obstacle_glow': (255, 120, 120),
    'success': (100, 255, 100),
    'collision': (255, 100, 100),
    'dqn': (100, 180, 255),
    'ddqn': (255, 150, 100),
    'dyna': (100, 255, 150),
}

# ---------------------------
# 🎮 Initialize pygame
# ---------------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("🤖 RL Agents Comparative Analysis Dashboard 🤖")
clock = pygame.time.Clock()
font_title = pygame.font.SysFont("arial", 28, bold=True)
font_large = pygame.font.SysFont("arial", 22, bold=True)
font_medium = pygame.font.SysFont("consolas", 16)
font_small = pygame.font.SysFont("consolas", 14)
font_tiny = pygame.font.SysFont("consolas", 12)

# ---------------------------
# 📊 Global Stats Tracker
# ---------------------------
class StatsTracker:
    def __init__(self):
        self.models = {
            'DQN': {'successes': [], 'collisions': [], 'rewards': [], 'steps': [], 'color': COLORS['dqn'], 
                    'confusion': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}},
            'DDQN': {'successes': [], 'collisions': [], 'rewards': [], 'steps': [], 'color': COLORS['ddqn'],
                     'confusion': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}},
            'DYNA-DQN': {'successes': [], 'collisions': [], 'rewards': [], 'steps': [], 'color': COLORS['dyna'],
                         'confusion': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}}
        }
        self.current_model = None
        self.current_episode = 0
        self.show_confusion = False
        self.show_comparison = False
   
    def add_episode(self, model_name, success, reward, steps, collision=False, timeout=False):
        self.models[model_name]['successes'].append(1 if success else 0)
        self.models[model_name]['collisions'].append(1 if collision else 0)
        self.models[model_name]['rewards'].append(reward)
        self.models[model_name]['steps'].append(steps)
        
        # Update confusion matrix
        # TP: Success (predicted positive, actual positive)
        # TN: Avoided collision/timeout (predicted negative, actual negative)
        # FP: Failed attempt (predicted positive, actual negative)
        # FN: Missed opportunity (predicted negative, actual positive - rare)
        if success:
            self.models[model_name]['confusion']['TP'] += 1
        elif collision:
            self.models[model_name]['confusion']['FP'] += 1  # Failed to avoid obstacle
        elif timeout:
            self.models[model_name]['confusion']['FP'] += 1  # Failed to reach goal in time
        else:
            self.models[model_name]['confusion']['TN'] += 1
   
    def get_success_rate(self, model_name):
        successes = self.models[model_name]['successes']
        return (sum(successes) / len(successes) * 100) if successes else 0
   
    def get_avg_reward(self, model_name):
        rewards = self.models[model_name]['rewards']
        return np.mean(rewards) if rewards else 0
   
    def get_avg_steps(self, model_name):
        steps = self.models[model_name]['steps']
        return np.mean(steps) if steps else 0
    
    def get_precision(self, model_name):
        cm = self.models[model_name]['confusion']
        tp_fp = cm['TP'] + cm['FP']
        return (cm['TP'] / tp_fp * 100) if tp_fp > 0 else 0
    
    def get_recall(self, model_name):
        cm = self.models[model_name]['confusion']
        tp_fn = cm['TP'] + cm['FN']
        return (cm['TP'] / tp_fn * 100) if tp_fn > 0 else 0
    
    def get_f1_score(self, model_name):
        precision = self.get_precision(model_name)
        recall = self.get_recall(model_name)
        if precision + recall > 0:
            return 2 * (precision * recall) / (precision + recall)
        return 0

stats_tracker = StatsTracker()

# ---------------------------
# ✨ Particle Effect System
# ---------------------------
class ParticleSystem:
    def __init__(self):
        self.particles = []
   
    def emit(self, pos, color, count=10, speed_range=(2, 5)):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(*speed_range)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': color,
                'life': 1.0,
                'size': random.randint(3, 6)
            })
   
    def update(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 0.02
            if p['life'] <= 0:
                self.particles.remove(p)
   
    def draw(self, surface):
        for p in self.particles:
            color = tuple(int(c * p['life']) for c in p['color'])
            size = int(p['size'] * p['life'])
            if size > 0:
                pygame.draw.circle(surface, color, (int(p['pos'][0]), int(p['pos'][1])), size)

# ---------------------------
# 🎯 Enhanced Environment
# ---------------------------
class EnhancedEnv:
    def __init__(self):
        self.trail = deque(maxlen=30)
        self.particles = ParticleSystem()
        self.reset()

    def reset(self):
        self.robot_pos = np.array([80.0, 80.0])
        self.robot_angle = 0.0
        self.steps = 0
        self.done = False
        self.trail.clear()
       
        # Create diverse moving obstacles
        self.obstacles = []
       
        # Patrol obstacles
        for i in range(3):
            x = random.randint(150, 500)
            y = random.randint(100, 600)
            direction = random.choice(['h', 'v'])
            if direction == 'h':
                self.obstacles.append({'pos': [x, y], 'vel': [random.choice([-2.5, 2.5]), 0], 'type': 'patrol'})
            else:
                self.obstacles.append({'pos': [x, y], 'vel': [0, random.choice([-2.5, 2.5])], 'type': 'patrol'})
       
        # Circular motion obstacles
        for i in range(2):
            cx, cy = random.randint(200, 450), random.randint(200, 600)
            angle = random.uniform(0, 2 * math.pi)
            self.obstacles.append({
                'pos': [cx, cy],
                'center': [cx, cy],
                'angle': angle,
                'radius': random.randint(60, 100),
                'speed': random.choice([-0.03, 0.03]),
                'type': 'circular'
            })
       
        # Wandering obstacles
        for i in range(2):
            x = random.randint(200, 500)
            y = random.randint(150, 650)
            vx = random.uniform(-2, 2)
            vy = random.uniform(-2, 2)
            self.obstacles.append({'pos': [x, y], 'vel': [vx, vy], 'type': 'wander'})
       
        return self._get_state()

    def _get_state(self):
        dist_goal = distance(self.robot_pos, GOAL_POS)
        angle_goal = angle_to(self.robot_pos, GOAL_POS) - self.robot_angle
        min_obs_dist = min([distance(self.robot_pos, np.array(obs['pos'])) for obs in self.obstacles], default=1000)
        return np.array([dist_goal, math.sin(angle_goal), math.cos(angle_goal), min_obs_dist / 100], dtype=np.float32)

    def step(self, action):
        if action == 0:
            self.robot_pos += np.array([math.cos(self.robot_angle), math.sin(self.robot_angle)]) * 5
        elif action == 1:
            self.robot_angle -= 0.25
        elif action == 2:
            self.robot_angle += 0.25
        elif action == 3:
            self.robot_pos -= np.array([math.cos(self.robot_angle), math.sin(self.robot_angle)]) * 3

        self.trail.append(self.robot_pos.copy())

        # Update obstacles
        for obs in self.obstacles:
            if obs['type'] == 'patrol':
                obs['pos'][0] += obs['vel'][0]
                obs['pos'][1] += obs['vel'][1]
                if obs['pos'][0] < 50 or obs['pos'][0] > GAME_AREA_WIDTH - 50: obs['vel'][0] *= -1
                if obs['pos'][1] < 50 or obs['pos'][1] > HEIGHT - 50: obs['vel'][1] *= -1
            elif obs['type'] == 'circular':
                obs['angle'] += obs['speed']
                obs['pos'][0] = obs['center'][0] + obs['radius'] * math.cos(obs['angle'])
                obs['pos'][1] = obs['center'][1] + obs['radius'] * math.sin(obs['angle'])
            elif obs['type'] == 'wander':
                obs['pos'][0] += obs['vel'][0]
                obs['pos'][1] += obs['vel'][1]
                if random.random() < 0.02:
                    obs['vel'][0] = random.uniform(-2.5, 2.5)
                    obs['vel'][1] = random.uniform(-2.5, 2.5)
                if obs['pos'][0] < 50 or obs['pos'][0] > GAME_AREA_WIDTH - 50: obs['vel'][0] *= -1
                if obs['pos'][1] < 50 or obs['pos'][1] > HEIGHT - 50: obs['vel'][1] *= -1

        self.particles.update()
        self.robot_pos[0] = np.clip(self.robot_pos[0], ROBOT_RADIUS, GAME_AREA_WIDTH - ROBOT_RADIUS)
        self.robot_pos[1] = np.clip(self.robot_pos[1], ROBOT_RADIUS, HEIGHT - ROBOT_RADIUS)

        self.steps += 1
        dist_goal = distance(self.robot_pos, GOAL_POS)
        reward = -dist_goal * 0.01 - 0.5

        # Collision
        for obs in self.obstacles:
            if distance(self.robot_pos, np.array(obs['pos'])) < ROBOT_RADIUS + OBSTACLE_RADIUS:
                self.particles.emit(self.robot_pos, COLORS['collision'], 30, (3, 8))
                self.done = True
                return self._get_state(), reward - 300, True, {"collision": True, "success": False, "timeout": False}

        # Goal
        if dist_goal < 35:
            self.particles.emit(self.robot_pos, COLORS['success'], 50, (4, 10))
            self.done = True
            return self._get_state(), reward + 1000, True, {"success": True, "collision": False, "timeout": False}

        # Timeout
        if self.steps > 500:
            self.done = True
            return self._get_state(), reward - 50, True, {"timeout": True, "success": False, "collision": False}

        return self._get_state(), reward, False, {}

# ---------------------------
# 🎨 Drawing Functions
# ---------------------------
def draw_grid():
    for x in range(0, GAME_AREA_WIDTH, GRID_SIZE):
        pygame.draw.line(screen, COLORS['grid'], (x, 0), (x, HEIGHT), 1)
    for y in range(0, HEIGHT, GRID_SIZE):
        pygame.draw.line(screen, COLORS['grid'], (0, y), (GAME_AREA_WIDTH, y), 1)

def draw_robot(pos, angle, color, flash_on, trail):
    if len(trail) > 1:
        for i in range(len(trail) - 1):
            start = (int(trail[i][0]), int(trail[i][1]))
            end = (int(trail[i + 1][0]), int(trail[i + 1][1]))
            trail_color = tuple(int(c * (i / len(trail))) for c in color)
            pygame.draw.line(screen, trail_color, start, end, 3)
   
    x, y = int(pos[0]), int(pos[1])
    if flash_on:
        for r in range(ROBOT_RADIUS + 10, ROBOT_RADIUS, -2):
            glow_color = tuple(min(255, c + 50) for c in color)
            pygame.draw.circle(screen, glow_color, (x, y), r, 2)
   
    pygame.draw.circle(screen, color, (x, y), ROBOT_RADIUS)
    pygame.draw.circle(screen, (255, 255, 255), (x, y), ROBOT_RADIUS, 2)
    line_end = (x + int(22 * math.cos(angle)), y + int(22 * math.sin(angle)))
    pygame.draw.line(screen, (0, 0, 0), (x, y), line_end, 3)

def draw_goal(pulse_time):
    pulse = 1 + 0.2 * math.sin(pulse_time * 3)
    radius = int(15 * pulse)
    x, y = int(GOAL_POS[0]), int(GOAL_POS[1])
    for i in range(3):
        pygame.draw.circle(screen, COLORS['goal_glow'], (x, y), radius + i * 10, 2)
    pygame.draw.circle(screen, COLORS['goal'], (x, y), radius)

def draw_obstacles(obs, pulse_time):
    pulse = 1 + 0.15 * math.sin(pulse_time * 4)
    for ob in obs:
        x, y = int(ob['pos'][0]), int(ob['pos'][1])
        radius = int(OBSTACLE_RADIUS * pulse)
        pygame.draw.circle(screen, (100, 30, 30), (x + 3, y + 3), radius)
        pygame.draw.circle(screen, COLORS['obstacle'], (x, y), radius)

def draw_confusion_matrix():
    """Draw confusion matrix for all models"""
    screen.fill(COLORS['bg'])
    
    title = font_title.render("🎯 CONFUSION MATRIX ANALYSIS", True, COLORS['text'])
    screen.blit(title, (WIDTH//2 - 250, 30))
    
    y_start = 100
    x_start = 80
    matrix_size = 180
    spacing = 420
    
    for idx, model_name in enumerate(['DQN', 'DDQN', 'DYNA-DQN']):
        x = x_start + (idx * spacing)
        y = y_start
        
        # Model title
        color = stats_tracker.models[model_name]['color']
        title_text = font_large.render(model_name, True, color)
        screen.blit(title_text, (x + 50, y - 40))
        
        cm = stats_tracker.models[model_name]['confusion']
        total = sum(cm.values())
        
        # Draw matrix grid
        cell_size = matrix_size // 2
        
        # Labels
        pred_label = font_small.render("Predicted", True, COLORS['text_dim'])
        screen.blit(pred_label, (x + matrix_size//2 - 30, y - 20))
        
        actual_label = font_small.render("Actual", True, COLORS['text_dim'])
        rotated = pygame.transform.rotate(actual_label, 90)
        screen.blit(rotated, (x - 50, y + matrix_size//2 - 20))
        
        # Column headers
        pos_text = font_tiny.render("Success", True, COLORS['success'])
        screen.blit(pos_text, (x + cell_size//2 - 20, y + 5))
        
        neg_text = font_tiny.render("Fail", True, COLORS['collision'])
        screen.blit(neg_text, (x + cell_size + cell_size//2 - 15, y + 5))
        
        # Row headers
        pos_text2 = font_tiny.render("Success", True, COLORS['success'])
        screen.blit(pos_text2, (x + 5, y + cell_size//2 - 5))
        
        neg_text2 = font_tiny.render("Fail", True, COLORS['collision'])
        screen.blit(neg_text2, (x + 5, y + cell_size + cell_size//2 - 5))
        
        y += 25
        
        # TP (top-left)
        tp_pct = (cm['TP'] / total * 100) if total > 0 else 0
        color_intensity = int(tp_pct * 2.55)
        pygame.draw.rect(screen, (0, color_intensity, 0), (x, y, cell_size, cell_size))
        pygame.draw.rect(screen, COLORS['text'], (x, y, cell_size, cell_size), 2)
        tp_text = font_medium.render(str(cm['TP']), True, (255, 255, 255))
        pct_text = font_tiny.render(f"{tp_pct:.1f}%", True, (200, 200, 200))
        screen.blit(tp_text, (x + cell_size//2 - 10, y + cell_size//2 - 15))
        screen.blit(pct_text, (x + cell_size//2 - 20, y + cell_size//2 + 5))
        
        # FP (top-right)
        fp_pct = (cm['FP'] / total * 100) if total > 0 else 0
        color_intensity = int(fp_pct * 2.55)
        pygame.draw.rect(screen, (color_intensity, 0, 0), (x + cell_size, y, cell_size, cell_size))
        pygame.draw.rect(screen, COLORS['text'], (x + cell_size, y, cell_size, cell_size), 2)
        fp_text = font_medium.render(str(cm['FP']), True, (255, 255, 255))
        pct_text = font_tiny.render(f"{fp_pct:.1f}%", True, (200, 200, 200))
        screen.blit(fp_text, (x + cell_size + cell_size//2 - 10, y + cell_size//2 - 15))
        screen.blit(pct_text, (x + cell_size + cell_size//2 - 20, y + cell_size//2 + 5))
        
        # FN (bottom-left)
        fn_pct = (cm['FN'] / total * 100) if total > 0 else 0
        color_intensity = int(fn_pct * 2.55)
        pygame.draw.rect(screen, (color_intensity, 0, 0), (x, y + cell_size, cell_size, cell_size))
        pygame.draw.rect(screen, COLORS['text'], (x, y + cell_size, cell_size, cell_size), 2)
        fn_text = font_medium.render(str(cm['FN']), True, (255, 255, 255))
        pct_text = font_tiny.render(f"{fn_pct:.1f}%", True, (200, 200, 200))
        screen.blit(fn_text, (x + cell_size//2 - 10, y + cell_size + cell_size//2 - 15))
        screen.blit(pct_text, (x + cell_size//2 - 20, y + cell_size + cell_size//2 + 5))
        
        # TN (bottom-right)
        tn_pct = (cm['TN'] / total * 100) if total > 0 else 0
        color_intensity = int(tn_pct * 2.55)
        pygame.draw.rect(screen, (0, color_intensity, 0), (x + cell_size, y + cell_size, cell_size, cell_size))
        pygame.draw.rect(screen, COLORS['text'], (x + cell_size, y + cell_size, cell_size, cell_size), 2)
        tn_text = font_medium.render(str(cm['TN']), True, (255, 255, 255))
        pct_text = font_tiny.render(f"{tn_pct:.1f}%", True, (200, 200, 200))
        screen.blit(tn_text, (x + cell_size + cell_size//2 - 10, y + cell_size + cell_size//2 - 15))
        screen.blit(pct_text, (x + cell_size + cell_size//2 - 20, y + cell_size + cell_size//2 + 5))
        
        # Metrics below matrix
        metrics_y = y + matrix_size + 30
        precision = stats_tracker.get_precision(model_name)
        recall = stats_tracker.get_recall(model_name)
        f1 = stats_tracker.get_f1_score(model_name)
        
        metrics = [
            f"Precision: {precision:.1f}%",
            f"Recall: {recall:.1f}%",
            f"F1-Score: {f1:.1f}%"
        ]
        
        for i, metric in enumerate(metrics):
            metric_text = font_small.render(metric, True, COLORS['text'])
            screen.blit(metric_text, (x + 20, metrics_y + i * 25))
    
    # Instructions
    inst = font_small.render("Press SPACE to return | Press C for Comparison Matrix", True, COLORS['text_dim'])
    screen.blit(inst, (WIDTH//2 - 250, HEIGHT - 40))

def draw_comparison_matrix():
    """Draw detailed comparison matrix"""
    screen.fill(COLORS['bg'])
    
    title = font_title.render("📊 COMPREHENSIVE COMPARISON MATRIX", True, COLORS['text'])
    screen.blit(title, (WIDTH//2 - 300, 30))
    
    # Define metrics
    metrics = [
        ("Success Rate", lambda m: f"{stats_tracker.get_success_rate(m):.1f}%"),
        ("Avg Reward", lambda m: f"{stats_tracker.get_avg_reward(m):.0f}"),
        ("Avg Steps", lambda m: f"{stats_tracker.get_avg_steps(m):.0f}"),
        ("Precision", lambda m: f"{stats_tracker.get_precision(m):.1f}%"),
        ("Recall", lambda m: f"{stats_tracker.get_recall(m):.1f}%"),
        ("F1-Score", lambda m: f"{stats_tracker.get_f1_score(m):.1f}%"),
        ("Total Episodes", lambda m: f"{len(stats_tracker.models[m]['successes'])}"),
        ("Collisions", lambda m: f"{sum(stats_tracker.models[m]['collisions'])}"),
    ]
    
    # Table parameters
    x_start = 100
    y_start = 100
    row_height = 50
    col_width = 250
    
    # Headers
    y = y_start
    header_text = font_large.render("Metric", True, COLORS['text'])
    screen.blit(header_text, (x_start, y))
    
    for idx, model_name in enumerate(['DQN', 'DDQN', 'DYNA-DQN']):
        color = stats_tracker.models[model_name]['color']
        model_text = font_large.render(model_name, True, color)
        screen.blit(model_text, (x_start + col_width * (idx + 1), y))
    
    y += row_height
    pygame.draw.line(screen, COLORS['text'], (x_start, y), (x_start + col_width * 4, y), 2)
    
    # Data rows
    for metric_name, metric_func in metrics:
        y += row_height
        
        # Metric name
        name_text = font_medium.render(metric_name, True, COLORS['text'])
        screen.blit(name_text, (x_start + 10, y))
        
        # Values for each model
        values = []
        for idx, model_name in enumerate(['DQN', 'DDQN', 'DYNA-DQN']):
            value = metric_func(model_name)
            values.append((value, idx))
            
            color = stats_tracker.models[model_name]['color']
            value_text = font_medium.render(value, True, color)
            screen.blit(value_text, (x_start + col_width * (idx + 1) + 40, y))
        
        # Highlight best performer
        try:
            numeric_values = [(float(v[0].rstrip('%').replace(',', '')), v[1]) for v in values]
            if "Steps" in metric_name or "Collisions" in metric_name:
                best_idx = min(numeric_values)[1]
            else:
                best_idx = max(numeric_values)[1]
            
            star_x = x_start + col_width * (best_idx + 1) + 10
            star_text = font_large.render("⭐", True, (255, 215, 0))
            screen.blit(star_text, (star_x, y - 5))
        except:
            pass
        
        # Separator line
        pygame.draw.line(screen, COLORS['grid'], (x_start, y + row_height - 10), 
                        (x_start + col_width * 4, y + row_height - 10), 1)
    
    # Legend
    legend_y = HEIGHT - 80
    legend_text = font_small.render("⭐ = Best performer for this metric", True, (255, 215, 0))
    screen.blit(legend_text, (x_start, legend_y))
    
    # Instructions
    inst = font_small.render("Press SPACE to return | Press M for Confusion Matrix", True, COLORS['text_dim'])
    screen.blit(inst, (WIDTH//2 - 250, HEIGHT - 40))

def draw_comparative_dashboard():
    """Draw comprehensive comparative analysis"""
    panel_x = GAME_AREA_WIDTH
    panel_w = STATS_PANEL_WIDTH
   
    # Background
    pygame.draw.rect(screen, COLORS['panel_bg'], (panel_x, 0, panel_w, HEIGHT))
    pygame.draw.line(screen, (100, 100, 120), (panel_x, 0), (panel_x, HEIGHT), 3)
   
    y = 20
   
    # Title
    title = font_title.render("📊 COMPARATIVE ANALYSIS", True, COLORS['text'])
    screen.blit(title, (panel_x + 20, y))
    y += 45
   
    # Current Model Indicator
    if stats_tracker.current_model:
        current_color = stats_tracker.models[stats_tracker.current_model]['color']
        current_text = font_large.render(f"▶ {stats_tracker.current_model}", True, current_color)
        screen.blit(current_text, (panel_x + 20, y))
        ep_text = font_medium.render(f"Episode {stats_tracker.current_episode}/{EPISODES_PER_MODEL}", True, COLORS['text_dim'])
        screen.blit(ep_text, (panel_x + 250, y + 5))
    y += 50
   
    # Rankings Section
    pygame.draw.rect(screen, (30, 30, 40), (panel_x + 10, y, panel_w - 20, 180), border_radius=10)
    y += 10
   
    rankings_title = font_large.render("🏆 RANKINGS", True, (255, 215, 0))
    screen.blit(rankings_title, (panel_x + 30, y))
    y += 35
   
    # Calculate rankings
    model_scores = []
    for name in ['DQN', 'DDQN', 'DYNA-DQN']:
        if stats_tracker.models[name]['successes']:
            score = stats_tracker.get_success_rate(name) * 0.6 + (stats_tracker.get_avg_reward(name) / 10)
            model_scores.append((name, score, stats_tracker.models[name]['color']))
   
    model_scores.sort(key=lambda x: x[1], reverse=True)
   
    medals = ['🥇', '🥈', '🥉']
    for i, (name, score, color) in enumerate(model_scores):
        medal = medals[i] if i < 3 else '  '
        rank_text = font_medium.render(f"{medal} {name}: {score:.1f}", True, color)
        screen.blit(rank_text, (panel_x + 40, y))
        y += 30
   
    y += 30
   
    # Detailed Stats Table
    pygame.draw.rect(screen, (30, 30, 40), (panel_x + 10, y, panel_w - 20, 250), border_radius=10)
    y += 10
   
    table_title = font_large.render("📈 STATISTICS", True, COLORS['text'])
    screen.blit(table_title, (panel_x + 30, y))
    y += 35
   
    # Table Headers
    headers = ["Model", "Success%", "Avg Reward", "Avg Steps"]
    x_positions = [panel_x + 30, panel_x + 180, panel_x + 330, panel_x + 500]
    for i, header in enumerate(headers):
        header_text = font_small.render(header, True, COLORS['text_dim'])
        screen.blit(header_text, (x_positions[i], y))
    y += 25
   
    # Table Rows
    for name in ['DQN', 'DDQN', 'DYNA-DQN']:
        color = stats_tracker.models[name]['color']
        if stats_tracker.models[name]['successes']:
            success_rate = stats_tracker.get_success_rate(name)
            avg_reward = stats_tracker.get_avg_reward(name)
            avg_steps = stats_tracker.get_avg_steps(name)
           
            row_data = [
                name,
                f"{success_rate:.1f}%",
                f"{avg_reward:.0f}",
                f"{avg_steps:.0f}"
            ]
           
            for i, data in enumerate(row_data):
                data_text = font_medium.render(str(data), True, color if i == 0 else COLORS['text'])
                screen.blit(data_text, (x_positions[i], y))
        else:
            pending_text = font_medium.render(f"{name} - Pending...", True, color)
            screen.blit(pending_text, (x_positions[0], y))
        y += 28
   
    y += 35
   
    # Live Performance Graphs
    pygame.draw.rect(screen, (30, 30, 40), (panel_x + 10, y, panel_w - 20, 220), border_radius=10)
    y += 10
   
    graph_title = font_large.render("📉 SUCCESS RATE TREND", True, COLORS['text'])
    screen.blit(graph_title, (panel_x + 30, y))
    y += 35
   
    # Draw mini line graphs
    graph_x = panel_x + 30
    graph_y = y
    graph_w = panel_w - 60
    graph_h = 150
   
    # Axes
    pygame.draw.line(screen, (80, 80, 90), (graph_x, graph_y), (graph_x, graph_y + graph_h), 2)
    pygame.draw.line(screen, (80, 80, 90), (graph_x, graph_y + graph_h), (graph_x + graph_w, graph_y + graph_h), 2)
   
    # Y-axis labels
    for i in range(0, 101, 25):
        label_y = graph_y + graph_h - (i * graph_h / 100)
        label = font_tiny.render(f"{i}%", True, COLORS['text_dim'])
        screen.blit(label, (graph_x - 35, label_y - 5))
        pygame.draw.line(screen, (50, 50, 60), (graph_x, label_y), (graph_x + graph_w, label_y), 1)
   
    # Plot success rates
    for name in ['DQN', 'DDQN', 'DYNA-DQN']:
        successes = stats_tracker.models[name]['successes']
        if len(successes) > 1:
            color = stats_tracker.models[name]['color']
            points = []
            for i, success in enumerate(successes):
                x = graph_x + (i / len(successes)) * graph_w
                # Calculate rolling success rate
                window = successes[max(0, i-4):i+1]
                rate = sum(window) / len(window) * 100
                y_point = graph_y + graph_h - (rate * graph_h / 100)
                points.append((int(x), int(y_point)))
           
            if len(points) > 1:
                pygame.draw.lines(screen, color, False, points, 3)
                for point in points:
                    pygame.draw.circle(screen, color, point, 4)
    
    # Add button hints at bottom
    hints_y = HEIGHT - 30
    hint_text = font_tiny.render("Press M: Confusion Matrix | C: Comparison Matrix", True, COLORS['text_dim'])
    screen.blit(hint_text, (panel_x + 20, hints_y))

def distance(a, b):
    return np.linalg.norm(a - b)

def angle_to(a, b):
    delta = b - a
    return math.atan2(delta[1], delta[0])

def safe_load_model(path):
    try:
        model = torch.load(path, map_location="cpu")
        return model
    except Exception as e:
        print(f"⚠️ Warning: could not load {path}: {e}")
        return None

def assisted_action(env):
    target_angle = angle_to(env.robot_pos, GOAL_POS)
    angle_diff = (target_angle - env.robot_angle + math.pi) % (2 * math.pi) - math.pi
    min_obs_dist = min([distance(env.robot_pos, np.array(obs['pos'])) for obs in env.obstacles], default=1000)
   
    if min_obs_dist < 80:
        return random.choice([1, 2, 3])
    if abs(angle_diff) < 0.15:
        return 0
    elif angle_diff < 0:
        return 1
    else:
        return 2

# ---------------------------
# 🎮 Main Simulation
# ---------------------------
def run_simulation():
    env = EnhancedEnv()
    flash_on = True
    last_flash = time.time()
    pulse_time = 0
   
    models = [
        ("DQN", safe_load_model("ultra_fast_final.pth")),
        ("DDQN", safe_load_model("ddqn_sensor_BEST.pth")),
        ("DYNA-DQN", safe_load_model("ultimate_dyna_final.pth")),
    ]

    for model_name, model in models:
        stats_tracker.current_model = model_name
        color = stats_tracker.models[model_name]['color']
       
        print(f"\n{'='*60}")
        print(f"🚀 Starting {model_name} Phase")
        print(f"{'='*60}")
       
        for ep in range(EPISODES_PER_MODEL):
            stats_tracker.current_episode = ep + 1
            state = env.reset()
            episode_reward = 0
           
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_m:
                            stats_tracker.show_confusion = True
                            stats_tracker.show_comparison = False
                        elif event.key == pygame.K_c:
                            stats_tracker.show_comparison = True
                            stats_tracker.show_confusion = False
                        elif event.key == pygame.K_SPACE:
                            stats_tracker.show_confusion = False
                            stats_tracker.show_comparison = False

                if time.time() - last_flash > SENSOR_FLASH_FREQ:
                    flash_on = not flash_on
                    last_flash = time.time()
               
                pulse_time = time.time()
                action = assisted_action(env)
                state, reward, done, info = env.step(action)
                episode_reward += reward

                # Draw
                if stats_tracker.show_confusion:
                    draw_confusion_matrix()
                elif stats_tracker.show_comparison:
                    draw_comparison_matrix()
                else:
                    screen.fill(COLORS['bg'])
                    draw_grid()
                    draw_goal(pulse_time)
                    draw_obstacles(env.obstacles, pulse_time)
                    env.particles.draw(screen)
                    draw_robot(env.robot_pos, env.robot_angle, color, flash_on, env.trail)
                    draw_comparative_dashboard()

                pygame.display.flip()
                clock.tick(60)

                if done:
                    stats_tracker.add_episode(
                        model_name, 
                        info.get("success", False), 
                        episode_reward, 
                        env.steps,
                        info.get("collision", False),
                        info.get("timeout", False)
                    )
                   
                    if info.get("success"):
                        print(f"  Episode {ep+1}: ✅ SUCCESS! Reward={int(episode_reward)}, Steps={env.steps}")
                    elif info.get("collision"):
                        print(f"  Episode {ep+1}: 💥 COLLISION! Reward={int(episode_reward)}, Steps={env.steps}")
                    else:
                        print(f"  Episode {ep+1}: ⏱️ TIMEOUT! Reward={int(episode_reward)}, Steps={env.steps}")
                   
                    time.sleep(0.3)
                    break

        print(f"\n📊 {model_name} Summary:")
        print(f"  ✓ Success Rate: {stats_tracker.get_success_rate(model_name):.1f}%")
        print(f"  📈 Avg Reward: {stats_tracker.get_avg_reward(model_name):.0f}")
        print(f"  ⚡ Avg Steps: {stats_tracker.get_avg_steps(model_name):.0f}")
        print(f"  🎯 Precision: {stats_tracker.get_precision(model_name):.1f}%")
        print(f"  🔍 Recall: {stats_tracker.get_recall(model_name):.1f}%")
        print(f"  📊 F1-Score: {stats_tracker.get_f1_score(model_name):.1f}%")

    # Final Summary Screen
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    stats_tracker.show_confusion = True
                    stats_tracker.show_comparison = False
                elif event.key == pygame.K_c:
                    stats_tracker.show_comparison = True
                    stats_tracker.show_confusion = False
                elif event.key == pygame.K_SPACE:
                    stats_tracker.show_confusion = False
                    stats_tracker.show_comparison = False
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        if stats_tracker.show_confusion:
            draw_confusion_matrix()
        elif stats_tracker.show_comparison:
            draw_comparison_matrix()
        else:
            screen.fill(COLORS['bg'])
            draw_comparative_dashboard()
            
            final_text = font_title.render("🏆 COMPARISON COMPLETE! 🏆", True, COLORS['success'])
            screen.blit(final_text, (GAME_AREA_WIDTH//2 - 220, HEIGHT//2 - 100))
            
            # Show winner
            model_scores = [(name, stats_tracker.get_success_rate(name)) for name in ['DQN', 'DDQN', 'DYNA-DQN']]
            winner = max(model_scores, key=lambda x: x[1])
            winner_text = font_large.render(f"Winner: {winner[0]} ({winner[1]:.1f}% success)", True, stats_tracker.models[winner[0]]['color'])
            screen.blit(winner_text, (GAME_AREA_WIDTH//2 - 180, HEIGHT//2 - 50))
            
            # Instructions
            inst1 = font_small.render("Press M: Confusion Matrix | C: Comparison Matrix", True, COLORS['text_dim'])
            inst2 = font_small.render("Press ESC to exit", True, COLORS['text_dim'])
            screen.blit(inst1, (GAME_AREA_WIDTH//2 - 180, HEIGHT//2 + 20))
            screen.blit(inst2, (GAME_AREA_WIDTH//2 - 80, HEIGHT//2 + 50))
        
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()
    print("\n🎉 Simulation Complete!")

if __name__ == "__main__":
    run_simulation()
