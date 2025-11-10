import numpy as np
import matplotlib.pyplot as plt
from environment import NavigationEnv
from dqn_agent import DQNAgent
from ddpg_agent import DDPGAgent
from ppo_agent import PPOAgent
from sac_agent import SACAgent
import argparse
import os
import time

def evaluate_agent(env, agent, num_episodes=100, render=True):
    """Evaluate a trained agent"""
    total_rewards = []
    total_collisions = []
    total_steps = []
    success_count = 0
    times_to_goal = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        start_time = time.time()
        
        while not done:
            # Select action without exploration
            if isinstance(agent, DQNAgent):
                action = agent.select_action(state, training=False)
            elif isinstance(agent, PPOAgent):
                # PPO returns (action, log_prob, value) tuple
                action, _, _ = agent.select_action(state, training=False)
            else:
                action = agent.select_action(state, training=False)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            
            if render:
                env.render()
        
        end_time = time.time()
        episode_time = end_time - start_time
        
        total_rewards.append(episode_reward)
        total_collisions.append(info['collision_count'])
        total_steps.append(info['steps'])
        
        # Check if goal was reached
        if info['distance_to_goal'] < 30:
            success_count += 1
            times_to_goal.append(episode_time)
        
        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
              f"Collisions={info['collision_count']}, Steps={info['steps']}, "
              f"Distance to Goal={info['distance_to_goal']:.2f}")
    
    # Calculate statistics
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    avg_collisions = np.mean(total_collisions)
    avg_steps = np.mean(total_steps)
    success_rate = (success_count / num_episodes) * 100
    avg_time_to_goal = np.mean(times_to_goal) if times_to_goal else float('inf')
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average Collisions: {avg_collisions:.2f}")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Time to Goal: {avg_time_to_goal:.2f}s")
    print("="*50)
    
    return {
        'rewards': total_rewards,
        'collisions': total_collisions,
        'steps': total_steps,
        'success_rate': success_rate,
        'avg_time_to_goal': avg_time_to_goal
    }

def compare_algorithms(results_dict, save_dir='results'):
    """Create comparison visualizations"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    algorithms = list(results_dict.keys())
    
    # 1. Average Rewards
    avg_rewards = [np.mean(results_dict[alg]['rewards']) for alg in algorithms]
    std_rewards = [np.std(results_dict[alg]['rewards']) for alg in algorithms]
    axes[0, 0].bar(algorithms, avg_rewards, yerr=std_rewards, capsize=5)
    axes[0, 0].set_ylabel('Average Reward')
    axes[0, 0].set_title('Average Reward Comparison')
    axes[0, 0].grid(True, axis='y')
    
    # 2. Average Collisions
    avg_collisions = [np.mean(results_dict[alg]['collisions']) for alg in algorithms]
    axes[0, 1].bar(algorithms, avg_collisions, color='orange')
    axes[0, 1].set_ylabel('Average Collisions')
    axes[0, 1].set_title('Average Collision Count')
    axes[0, 1].grid(True, axis='y')
    
    # 3. Average Steps
    avg_steps = [np.mean(results_dict[alg]['steps']) for alg in algorithms]
    axes[0, 2].bar(algorithms, avg_steps, color='green')
    axes[0, 2].set_ylabel('Average Steps')
    axes[0, 2].set_title('Average Steps to Goal/Termination')
    axes[0, 2].grid(True, axis='y')
    
    # 4. Success Rate
    success_rates = [results_dict[alg]['success_rate'] for alg in algorithms]
    axes[1, 0].bar(algorithms, success_rates, color='purple')
    axes[1, 0].set_ylabel('Success Rate (%)')
    axes[1, 0].set_title('Success Rate Comparison')
    axes[1, 0].set_ylim([0, 100])
    axes[1, 0].grid(True, axis='y')
    
    # 5. Time to Goal
    times = [results_dict[alg]['avg_time_to_goal'] for alg in algorithms]
    times = [t if t != float('inf') else 0 for t in times]  # Handle inf
    axes[1, 1].bar(algorithms, times, color='red')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].set_title('Average Time to Reach Goal')
    axes[1, 1].grid(True, axis='y')
    
    # 6. Reward Distribution (Box plot)
    reward_data = [results_dict[alg]['rewards'] for alg in algorithms]
    axes[1, 2].boxplot(reward_data, labels=algorithms)
    axes[1, 2].set_ylabel('Reward')
    axes[1, 2].set_title('Reward Distribution')
    axes[1, 2].grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'algorithm_comparison.png'), dpi=300)
    print(f"\nComparison plot saved to {save_dir}/algorithm_comparison.png")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained RL agents')
    parser.add_argument('--algorithm', type=str, default='all',
                       choices=['dqn', 'ddpg', 'ppo', 'sac', 'all'],
                       help='Algorithm to evaluate')
    parser.add_argument('--model_dir', type=str, default='results',
                       help='Directory containing trained models')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                       help='Render environment during evaluation')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    results = {}
    
    # Evaluate DQN
    if args.algorithm in ['dqn', 'all']:
        model_path = os.path.join(args.model_dir, 'dqn_model.pt')
        if os.path.exists(model_path):
            print("\n" + "="*50)
            print("Evaluating DQN")
            print("="*50)
            env = NavigationEnv(render_mode='human' if args.render else None, continuous=False)
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            
            agent = DQNAgent(state_dim, action_dim)
            agent.load(model_path)
            agent.epsilon = 0.0  # No exploration during evaluation
            
            results['DQN'] = evaluate_agent(env, agent, num_episodes=args.episodes, render=args.render)
            env.close()
        else:
            print(f"DQN model not found at {model_path}")
    
    # Evaluate DDPG
    if args.algorithm in ['ddpg', 'all']:
        model_path = os.path.join(args.model_dir, 'ddpg_model.pt')
        if os.path.exists(model_path):
            print("\n" + "="*50)
            print("Evaluating DDPG")
            print("="*50)
            env = NavigationEnv(render_mode='human' if args.render else None, continuous=True)
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            
            agent = DDPGAgent(state_dim, action_dim)
            agent.load(model_path)
            
            results['DDPG'] = evaluate_agent(env, agent, num_episodes=args.episodes, render=args.render)
            env.close()
        else:
            print(f"DDPG model not found at {model_path}")
    
    # Evaluate PPO
    if args.algorithm in ['ppo', 'all']:
        model_path = os.path.join(args.model_dir, 'ppo_model.pt')
        if os.path.exists(model_path):
            print("\n" + "="*50)
            print("Evaluating PPO")
            print("="*50)
            env = NavigationEnv(render_mode='human' if args.render else None, continuous=True)
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            
            agent = PPOAgent(state_dim, action_dim)
            agent.load(model_path)
            
            results['PPO'] = evaluate_agent(env, agent, num_episodes=args.episodes, render=args.render)
            env.close()
        else:
            print(f"PPO model not found at {model_path}")
    
    # Evaluate SAC
    if args.algorithm in ['sac', 'all']:
        model_path = os.path.join(args.model_dir, 'sac_model.pt')
        if os.path.exists(model_path):
            print("\n" + "="*50)
            print("Evaluating SAC")
            print("="*50)
            env = NavigationEnv(render_mode='human' if args.render else None, continuous=True)
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            
            agent = SACAgent(state_dim, action_dim)
            agent.load(model_path)
            
            results['SAC'] = evaluate_agent(env, agent, num_episodes=args.episodes, render=args.render)
            env.close()
        else:
            print(f"SAC model not found at {model_path}")
    
    # Create comparative visualizations
    if len(results) > 1:
        compare_algorithms(results, save_dir=args.save_dir)

if __name__ == '__main__':
    main()