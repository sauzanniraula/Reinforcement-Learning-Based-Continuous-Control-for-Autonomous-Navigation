import numpy as np
import matplotlib.pyplot as plt
from environment import NavigationEnv
from dqn_agent import DQNAgent
from ddpg_agent import DDPGAgent
from ppo_agent import PPOAgent
from sac_agent import SACAgent
import argparse
import os
from datetime import datetime

def train_dqn(env, agent, num_episodes=500, render_freq=50):
    """Train DQN agent"""
    rewards_history = []
    collisions_history = []
    steps_history = []
    loss_history = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        episode_loss = []
        
        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.memory.push(state, action, reward, next_state, float(done))
            loss = agent.train()
            
            if loss is not None:
                episode_loss.append(loss)
            
            state = next_state
            episode_reward += reward
            
            if episode % render_freq == 0:
                env.render()
        
        rewards_history.append(episode_reward)
        collisions_history.append(info['collision_count'])
        steps_history.append(info['steps'])
        
        if episode_loss:
            loss_history.append(np.mean(episode_loss))
        
        if episode % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            avg_collisions = np.mean(collisions_history[-10:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                  f"Avg Collisions: {avg_collisions:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    return rewards_history, collisions_history, steps_history, loss_history

def train_ddpg(env, agent, num_episodes=500, render_freq=50):
    """Train DDPG agent"""
    rewards_history = []
    collisions_history = []
    steps_history = []
    actor_loss_history = []
    critic_loss_history = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        agent.noise.reset()
        episode_actor_loss = []
        episode_critic_loss = []
        
        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.memory.push(state, action, reward, next_state, float(done))
            actor_loss, critic_loss = agent.train()
            
            if actor_loss is not None:
                episode_actor_loss.append(actor_loss)
                episode_critic_loss.append(critic_loss)
            
            state = next_state
            episode_reward += reward
            
            if episode % render_freq == 0:
                env.render()
        
        rewards_history.append(episode_reward)
        collisions_history.append(info['collision_count'])
        steps_history.append(info['steps'])
        
        if episode_actor_loss:
            actor_loss_history.append(np.mean(episode_actor_loss))
            critic_loss_history.append(np.mean(episode_critic_loss))
        
        if episode % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            avg_collisions = np.mean(collisions_history[-10:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                  f"Avg Collisions: {avg_collisions:.2f}")
    
    return rewards_history, collisions_history, steps_history, actor_loss_history

def train_ppo(env, agent, num_episodes=500, rollout_length=2048, render_freq=50):
    """Train PPO agent"""
    rewards_history = []
    collisions_history = []
    steps_history = []
    loss_history = []
    
    state, _ = env.reset()
    episode_reward = 0
    episode_collisions = 0
    episode_steps = 0
    episode_count = 0
    
    for step in range(num_episodes * rollout_length):
        action, log_prob, value = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        agent.buffer.add(state, action, reward, log_prob, value, float(done))
        
        state = next_state
        episode_reward += reward
        episode_steps += 1
        
        if done:
            episode_collisions = info['collision_count']
            rewards_history.append(episode_reward)
            collisions_history.append(episode_collisions)
            steps_history.append(episode_steps)
            
            episode_count += 1
            if episode_count % 10 == 0:
                avg_reward = np.mean(rewards_history[-10:])
                avg_collisions = np.mean(collisions_history[-10:])
                print(f"Episode {episode_count}, Avg Reward: {avg_reward:.2f}, "
                      f"Avg Collisions: {avg_collisions:.2f}")
            
            state, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
            
            if episode_count >= num_episodes:
                break
        
        # Update policy
        if (step + 1) % rollout_length == 0:
            loss = agent.train(state)
            if loss is not None:
                loss_history.append(loss)
        
        if episode_count > 0 and episode_count % render_freq == 0 and not done:
            env.render()
    
    return rewards_history, collisions_history, steps_history, loss_history

def train_sac(env, agent, num_episodes=500, render_freq=50):
    """Train SAC agent"""
    rewards_history = []
    collisions_history = []
    steps_history = []
    policy_loss_history = []
    critic_loss_history = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        episode_policy_loss = []
        episode_critic_loss = []
        
        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.memory.push(state, action, reward, next_state, float(done))
            policy_loss, critic_loss, _ = agent.train()
            
            if policy_loss is not None:
                episode_policy_loss.append(policy_loss)
                episode_critic_loss.append(critic_loss)
            
            state = next_state
            episode_reward += reward
            
            if episode % render_freq == 0:
                env.render()
        
        rewards_history.append(episode_reward)
        collisions_history.append(info['collision_count'])
        steps_history.append(info['steps'])
        
        if episode_policy_loss:
            policy_loss_history.append(np.mean(episode_policy_loss))
            critic_loss_history.append(np.mean(episode_critic_loss))
        
        if episode % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            avg_collisions = np.mean(collisions_history[-10:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                  f"Avg Collisions: {avg_collisions:.2f}, Alpha: {agent.alpha:.3f}")
    
    return rewards_history, collisions_history, steps_history, policy_loss_history

def plot_results(results_dict, save_dir='results'):
    """Plot and save training results"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot rewards
    for name, (rewards, _, _, _) in results_dict.items():
        if rewards:
            window = 10
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(smoothed, label=name)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Average Reward per Episode')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot collisions
    for name, (_, collisions, _, _) in results_dict.items():
        if collisions:
            window = 10
            smoothed = np.convolve(collisions, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(smoothed, label=name)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Collisions')
    axes[0, 1].set_title('Average Collisions per Episode')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot steps
    for name, (_, _, steps, _) in results_dict.items():
        if steps:
            window = 10
            smoothed = np.convolve(steps, np.ones(window)/window, mode='valid')
            axes[1, 0].plot(smoothed, label=name)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps')
    axes[1, 0].set_title('Steps to Goal per Episode')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot convergence rate (success rate)
    for name, (rewards, _, _, _) in results_dict.items():
        if rewards:
            success_threshold = 50  # Define success threshold
            window = 50
            success_rate = []
            for i in range(window, len(rewards)):
                successes = sum(1 for r in rewards[i-window:i] if r > success_threshold)
                success_rate.append(successes / window * 100)
            axes[1, 1].plot(success_rate, label=name)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Success Rate (%)')
    axes[1, 1].set_title('Success Rate (50-episode window)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_comparison.png'), dpi=300)
    print(f"Results saved to {save_dir}/training_comparison.png")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train RL agents for navigation')
    parser.add_argument('--algorithm', type=str, default='all', 
                       choices=['dqn', 'ddpg', 'ppo', 'sac', 'all'],
                       help='Algorithm to train')
    parser.add_argument('--episodes', type=int, default=500,
                       help='Number of training episodes')
    parser.add_argument('--render', action='store_true',
                       help='Render environment during training')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    results = {}
    
    # Train DQN (Discrete actions)
    if args.algorithm in ['dqn', 'all']:
        print("\n" + "="*50)
        print("Training DQN (Discrete Control)")
        print("="*50)
        env = NavigationEnv(render_mode='human' if args.render else None, continuous=False)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        agent = DQNAgent(state_dim, action_dim)
        results['DQN'] = train_dqn(env, agent, num_episodes=args.episodes)
        agent.save(os.path.join(args.save_dir, 'dqn_model.pt'))
        env.close()
    
    # Train DDPG (Continuous actions)
    if args.algorithm in ['ddpg', 'all']:
        print("\n" + "="*50)
        print("Training DDPG (Continuous Control)")
        print("="*50)
        env = NavigationEnv(render_mode='human' if args.render else None, continuous=True)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        agent = DDPGAgent(state_dim, action_dim)
        results['DDPG'] = train_ddpg(env, agent, num_episodes=args.episodes)
        agent.save(os.path.join(args.save_dir, 'ddpg_model.pt'))
        env.close()
    
    # Train PPO (Continuous actions)
    if args.algorithm in ['ppo', 'all']:
        print("\n" + "="*50)
        print("Training PPO (Continuous Control)")
        print("="*50)
        env = NavigationEnv(render_mode='human' if args.render else None, continuous=True)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        agent = PPOAgent(state_dim, action_dim)
        results['PPO'] = train_ppo(env, agent, num_episodes=args.episodes)
        agent.save(os.path.join(args.save_dir, 'ppo_model.pt'))
        env.close()
    
    # Train SAC (Continuous actions)
    if args.algorithm in ['sac', 'all']:
        print("\n" + "="*50)
        print("Training SAC (Continuous Control)")
        print("="*50)
        env = NavigationEnv(render_mode='human' if args.render else None, continuous=True)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        agent = SACAgent(state_dim, action_dim)
        results['SAC'] = train_sac(env, agent, num_episodes=args.episodes)
        agent.save(os.path.join(args.save_dir, 'sac_model.pt'))
        env.close()
    
    # Plot comparative results
    if len(results) > 0:
        plot_results(results, save_dir=args.save_dir)

if __name__ == '__main__':
    main()