import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

class NavigationEnv(gym.Env):
    """
    Custom 2D Navigation Environment for Continuous Control
    Agent must navigate to target while avoiding obstacles
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, render_mode=None, continuous=True):
        super().__init__()
        
        # Environment parameters
        self.width = 800
        self.height = 600
        self.continuous = continuous
        self.max_steps = 500
        self.current_step = 0
        
        # Agent parameters
        self.agent_radius = 15
        self.agent_max_speed = 5.0
        self.agent_pos = np.array([100.0, 300.0])
        self.agent_vel = np.array([0.0, 0.0])
        
        # Target parameters
        self.target_pos = np.array([700.0, 300.0])
        self.target_radius = 20
        self.goal_threshold = 30
        
        # Obstacles (x, y, radius)
        self.obstacles = [
            np.array([200.0, 200.0, 40.0]),
            np.array([300.0, 400.0, 50.0]),
            np.array([450.0, 250.0, 45.0]),
            np.array([550.0, 450.0, 35.0]),
            np.array([600.0, 150.0, 40.0])
        ]
        
        # Metrics
        self.collision_count = 0
        self.total_reward = 0
        
        # Define action and observation space
        if continuous:
            # Continuous: [linear_velocity, angular_velocity]
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )
        else:
            # Discrete: 5 actions (forward, left, right, back, stop)
            self.action_space = spaces.Discrete(5)
        
        # Observation: [dist_to_target, angle_to_target, vel_x, vel_y, 
        #               dist_obs1, angle_obs1, ..., dist_obs5, angle_obs5]
        obs_dim = 4 + 2 * len(self.obstacles)
        self.observation_space = spaces.Box(
            low=np.float32(-np.inf),
            high=np.float32(np.inf),
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Rendering
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset agent position randomly
        self.agent_pos = np.array([
            np.random.uniform(50, 150),
            np.random.uniform(250, 350)
        ])
        self.agent_vel = np.array([0.0, 0.0])
        
        # Randomize target position
        self.target_pos = np.array([
            np.random.uniform(650, 750),
            np.random.uniform(250, 350)
        ])
        
        self.current_step = 0
        self.collision_count = 0
        self.total_reward = 0
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action):
        self.current_step += 1
        
        # Apply action
        if self.continuous:
            # Continuous control
            # Handle different action types (numpy array, list, torch tensor, etc.)
            if hasattr(action, 'cpu'):  # PyTorch tensor
                action = action.cpu().detach().numpy()
            action = np.asarray(action, dtype=np.float32).flatten()
            
            # Ensure we have exactly 2 action values
            if len(action) != 2:
                raise ValueError(f"Expected 2 action values, got {len(action)}")
            
            linear_vel = float(action[0]) * self.agent_max_speed
            angular_vel = float(action[1]) * 0.3  # max rotation speed
            
            # Update velocity
            current_angle = math.atan2(self.agent_vel[1], self.agent_vel[0])
            new_angle = current_angle + angular_vel
            speed = np.linalg.norm(self.agent_vel) * 0.9 + linear_vel * 0.1
            speed = float(np.clip(speed, 0, self.agent_max_speed))
            
            self.agent_vel[0] = float(speed * math.cos(new_angle))
            self.agent_vel[1] = float(speed * math.sin(new_angle))
        else:
            # Discrete control
            action = int(action)  # Ensure action is integer
            if action == 0:  # Forward
                direction = self.target_pos - self.agent_pos
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                self.agent_vel = direction * self.agent_max_speed
            elif action == 1:  # Left
                self.agent_vel = np.array([-self.agent_max_speed, 0.0])
            elif action == 2:  # Right
                self.agent_vel = np.array([self.agent_max_speed, 0.0])
            elif action == 3:  # Back
                self.agent_vel = np.array([0.0, self.agent_max_speed])
            elif action == 4:  # Stop
                self.agent_vel = np.array([0.0, 0.0])
        
        # Update position
        self.agent_pos = self.agent_pos + self.agent_vel
        
        # Keep agent within bounds
        self.agent_pos[0] = float(np.clip(self.agent_pos[0], self.agent_radius, 
                                          self.width - self.agent_radius))
        self.agent_pos[1] = float(np.clip(self.agent_pos[1], self.agent_radius, 
                                          self.height - self.agent_radius))
        
        # Check collisions and calculate reward
        reward = 0
        terminated = False
        truncated = False
        
        # Distance to target
        dist_to_target = np.linalg.norm(self.agent_pos - self.target_pos)
        
        # Reward shaping
        reward -= 0.01  # Small step penalty
        reward -= dist_to_target * 0.001  # Distance penalty
        
        # Check if reached target
        if dist_to_target < self.goal_threshold:
            reward += 100
            terminated = True
        
        # Check obstacle collisions
        collision = False
        for obs in self.obstacles:
            dist_to_obs = np.linalg.norm(self.agent_pos - obs[:2])
            if dist_to_obs < (self.agent_radius + obs[2]):
                reward -= 10
                collision = True
                self.collision_count += 1
                # Push agent away from obstacle
                push_dir = self.agent_pos - obs[:2]
                push_dir = push_dir / (np.linalg.norm(push_dir) + 1e-8)
                self.agent_pos += push_dir * 2
        
        # Check wall collisions
        if (self.agent_pos[0] <= self.agent_radius or 
            self.agent_pos[0] >= self.width - self.agent_radius or
            self.agent_pos[1] <= self.agent_radius or 
            self.agent_pos[1] >= self.height - self.agent_radius):
            reward -= 5
            collision = True
            self.collision_count += 1
        
        # Episode timeout
        if self.current_step >= self.max_steps:
            truncated = True
        
        self.total_reward += reward
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        # Distance and angle to target
        target_vec = self.target_pos - self.agent_pos
        dist_to_target = np.linalg.norm(target_vec)
        angle_to_target = math.atan2(target_vec[1], target_vec[0])
        
        obs = [
            dist_to_target / 1000.0,  # Normalize
            angle_to_target / math.pi,
            self.agent_vel[0] / self.agent_max_speed,
            self.agent_vel[1] / self.agent_max_speed
        ]
        
        # Distance and angle to each obstacle
        for obstacle in self.obstacles:
            obs_vec = obstacle[:2] - self.agent_pos
            dist_to_obs = np.linalg.norm(obs_vec)
            angle_to_obs = math.atan2(obs_vec[1], obs_vec[0])
            obs.extend([
                dist_to_obs / 1000.0,
                angle_to_obs / math.pi
            ])
        
        return np.array(obs, dtype=np.float32)
    
    def _get_info(self):
        return {
            'collision_count': self.collision_count,
            'total_reward': self.total_reward,
            'distance_to_goal': np.linalg.norm(self.agent_pos - self.target_pos),
            'steps': self.current_step
        }
    
    def render(self):
        if self.render_mode is None:
            return
        
        if self.screen is None:
            pygame.init()
            if self.render_mode == 'human':
                self.screen = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption("RL Navigation")
            else:
                self.screen = pygame.Surface((self.width, self.height))
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # Draw
        self.screen.fill((255, 255, 255))
        
        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.circle(self.screen, (100, 100, 100), 
                             obs[:2].astype(int), int(obs[2]))
        
        # Draw target
        pygame.draw.circle(self.screen, (0, 255, 0), 
                         self.target_pos.astype(int), self.target_radius)
        
        # Draw agent
        pygame.draw.circle(self.screen, (255, 0, 0), 
                         self.agent_pos.astype(int), self.agent_radius)
        
        # Draw velocity vector
        vel_end = self.agent_pos + self.agent_vel * 5
        pygame.draw.line(self.screen, (0, 0, 255), 
                        self.agent_pos.astype(int), vel_end.astype(int), 3)
        
        if self.render_mode == 'human':
            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
    
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None