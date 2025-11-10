# Reinforcement Learning Based Continuous Control for Autonomous Navigation

## Project Overview

This project implements and compares multiple RL algorithms for autonomous navigation:
- **DQN** (Deep Q-Network) - Discrete control
- **DDPG** (Deep Deterministic Policy Gradient) - Continuous control
- **PPO** (Proximal Policy Optimization) - Continuous control
- **SAC** (Soft Actor-Critic) - Continuous control

## Project Structure

```
project/
├── environment.py          # Custom 2D navigation environment
├── dqn_agent.py           # DQN implementation
├── ddpg_agent.py          # DDPG implementation
├── ppo_agent.py           # PPO implementation
├── sac_agent.py           # SAC implementation
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── requirements.txt       # Dependencies
└── results/              # Saved models and results
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train all algorithms:
```bash
python train.py --algorithm all --episodes 500
```

Train specific algorithm:
```bash
python train.py --algorithm sac --episodes 300
```

Train with rendering:
```bash
python train.py --algorithm ddpg --episodes 200 --render
```

### Evaluation

Evaluate all trained models:
```bash
python evaluate.py --algorithm all --episodes 100 --render
```

Evaluate specific model:
```bash
python evaluate.py --algorithm ppo --episodes 50 --render
```

## Environment Details

### Observation Space (14-dimensional)
- Distance to target (normalized)
- Angle to target
- Agent velocity (x, y)
- Distance and angle to 5 obstacles (10 values)

### Action Space
- **Discrete (DQN)**: 5 actions [forward, left, right, back, stop]
- **Continuous (DDPG/PPO/SAC)**: 2D [linear_velocity, angular_velocity]

### Rewards
- Step penalty: -0.01
- Distance penalty: -0.001 × distance
- Collision penalty: -10
- Wall collision: -5
- Goal reached: +100

## Performance Metrics

The project evaluates algorithms based on:
1. **Average Reward**: Cumulative reward per episode
2. **Collision Count**: Number of obstacle/wall collisions
3. **Steps to Goal**: Episode length
4. **Success Rate**: Percentage of successful goal reaches
5. **Time to Goal**: Real-time duration to reach goal
6. **Convergence Rate**: Learning speed

## Algorithm Comparison

### On-Policy vs Off-Policy

**On-Policy (PPO)**:
- Learns from current policy
- More stable but sample inefficient
- Better for continuous control
- Advantage: Monotonic improvement guarantee

**Off-Policy (DDPG, SAC, DQN)**:
- Learns from replay buffer
- More sample efficient
- Can be unstable
- Advantage: Better data utilization

### Discrete vs Continuous Control

**Discrete (DQN)**:
- Simple action selection
- Easier to train
- Less flexible movement
- Good for grid-based navigation

**Continuous (DDPG/PPO/SAC)**:
- Smooth trajectories
- More natural control
- Requires careful tuning
- Better for realistic robotics

## Hyperparameters

### DQN
- Learning rate: 1e-3
- Gamma: 0.99
- Epsilon: 1.0 → 0.01
- Buffer size: 100,000
- Batch size: 64

### DDPG
- Actor LR: 1e-4
- Critic LR: 1e-3
- Gamma: 0.99
- Tau: 0.005
- Batch size: 64

### PPO
- Learning rate: 3e-4
- Gamma: 0.99
- GAE lambda: 0.95
- Clip epsilon: 0.2
- Epochs: 10

### SAC
- Learning rate: 3e-4
- Gamma: 0.99
- Tau: 0.005
- Auto-tune alpha: True
- Batch size: 256

## Expected Results

After 500 episodes:
- **SAC**: Highest success rate (~80-90%), smooth policies
- **PPO**: Good stability (~70-80%), consistent performance
- **DDPG**: Fast learning (~60-70%), can be unstable
- **DQN**: Moderate performance (~50-60%), discrete actions

## Extension to Self-Driving Car

To extend this to a self-driving scenario:

1. Modify `environment.py`:
   - Add lane boundaries
   - Implement traffic rules
   - Add other vehicles

2. Update observation space:
   - Lane position
   - Relative velocities
   - Traffic density

3. Modify rewards:
   - Lane keeping reward
   - Speed maintenance
   - Overtaking bonus
   - Collision penalty

4. Adjust action space:
   - Steering angle
   - Acceleration/braking

## Troubleshooting

**Issue**: Agent not learning
- Solution: Check reward shaping, adjust learning rates

**Issue**: High collision rate
- Solution: Increase collision penalty, improve state representation

**Issue**: Slow convergence
- Solution: Tune hyperparameters, increase buffer size

**Issue**: PyGame window not closing
- Solution: Call `env.close()` properly

## References

1. Schulman et al., "Proximal Policy Optimization Algorithms", 2017
2. Lillicrap et al., "Continuous control with deep reinforcement learning", 2015
3. Haarnoja et al., "Soft Actor-Critic", 2018
4. Mnih et al., "Human-level control through deep reinforcement learning", 2015

## Contact

Saujan Prakash Niroula (122531001)
Chandrakanta Dhurwey (112101009)
IIT Palakkad - EE5531 Reinforcement Learning Based Control

## License

This project is for educational purposes as part of EE5531 course work.#
