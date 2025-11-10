import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os

class PerformanceAnalyzer:
    """Comprehensive analysis and report generation for RL navigation project"""
    
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        self.algorithms = ['DQN', 'DDPG', 'PPO', 'SAC']
        sns.set_style("whitegrid")
        
    def analyze_convergence(self, training_data):
        """Analyze convergence characteristics of each algorithm"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        for idx, (name, data) in enumerate(training_data.items()):
            row, col = idx // 2, idx % 2
            rewards, collisions, steps, losses = data
            
            # Smooth the data
            window = 20
            if len(rewards) >= window:
                smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
                episodes = np.arange(len(smoothed_rewards))
                
                axes[row, col].plot(episodes, smoothed_rewards, linewidth=2, label='Smoothed Reward')
                axes[row, col].fill_between(episodes, 
                                           smoothed_rewards - np.std(rewards[:len(smoothed_rewards)]),
                                           smoothed_rewards + np.std(rewards[:len(smoothed_rewards)]),
                                           alpha=0.3)
                
                # Mark convergence point (when reward stabilizes)
                convergence_threshold = 0.9 * max(smoothed_rewards)
                convergence_idx = np.where(smoothed_rewards >= convergence_threshold)[0]
                if len(convergence_idx) > 0:
                    conv_episode = convergence_idx[0]
                    axes[row, col].axvline(conv_episode, color='r', linestyle='--', 
                                          label=f'Convergence: Ep {conv_episode}')
                
                axes[row, col].set_title(f'{name} Convergence Analysis', fontsize=14, fontweight='bold')
                axes[row, col].set_xlabel('Episode', fontsize=12)
                axes[row, col].set_ylabel('Average Reward', fontsize=12)
                axes[row, col].legend()
                axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'convergence_analysis.png'), dpi=300)
        plt.close()
        
    def analyze_sample_efficiency(self, training_data):
        """Analyze sample efficiency (learning speed)"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for name, data in training_data.items():
            rewards = data[0]
            window = 50
            
            if len(rewards) >= window:
                # Calculate cumulative success rate
                success_threshold = 50
                cumulative_successes = []
                for i in range(window, len(rewards)):
                    successes = sum(1 for r in rewards[:i] if r > success_threshold)
                    cumulative_successes.append((successes / i) * 100)
                
                episodes = np.arange(window, len(rewards))
                ax.plot(episodes, cumulative_successes, linewidth=2.5, label=name, marker='o', 
                       markersize=3, markevery=len(episodes)//10)
        
        ax.set_title('Sample Efficiency Comparison', fontsize=16, fontweight='bold')
        ax.set_xlabel('Episode', fontsize=14)
        ax.set_ylabel('Cumulative Success Rate (%)', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'sample_efficiency.png'), dpi=300)
        plt.close()
        
    def analyze_stability(self, training_data):
        """Analyze training stability (variance in performance)"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Reward variance over time
        for name, data in training_data.items():
            rewards = data[0]
            window = 50
            
            if len(rewards) >= window:
                variances = []
                for i in range(window, len(rewards)):
                    var = np.var(rewards[i-window:i])
                    variances.append(var)
                
                episodes = np.arange(window, len(rewards))
                axes[0].plot(episodes, variances, linewidth=2, label=name, alpha=0.7)
        
        axes[0].set_title('Training Stability (Reward Variance)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Episode', fontsize=12)
        axes[0].set_ylabel('Variance', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        # Coefficient of variation
        cv_data = {}
        for name, data in training_data.items():
            rewards = data[0]
            if len(rewards) >= 100:
                # Calculate CV for latter half (more stable period)
                latter_half = rewards[len(rewards)//2:]
                cv = np.std(latter_half) / (np.mean(latter_half) + 1e-8)
                cv_data[name] = cv
        
        names = list(cv_data.keys())
        cvs = list(cv_data.values())
        colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
        
        axes[1].bar(names, cvs, color=colors, edgecolor='black', linewidth=1.5)
        axes[1].set_title('Coefficient of Variation (Lower = More Stable)', 
                         fontsize=14, fontweight='bold')
        axes[1].set_ylabel('CV', fontsize=12)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'stability_analysis.png'), dpi=300)
        plt.close()
        
    def analyze_exploration_exploitation(self, training_data):
        """Analyze exploration vs exploitation trade-off"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        for idx, (name, data) in enumerate(training_data.items()):
            row, col = idx // 2, idx % 2
            rewards, collisions, steps, _ = data
            
            # Early vs late performance
            split_point = len(rewards) // 2
            early_rewards = rewards[:split_point]
            late_rewards = rewards[split_point:]
            
            axes[row, col].hist(early_rewards, bins=30, alpha=0.5, label='Early Episodes', 
                               color='blue', edgecolor='black')
            axes[row, col].hist(late_rewards, bins=30, alpha=0.5, label='Late Episodes', 
                               color='red', edgecolor='black')
            
            axes[row, col].set_title(f'{name}: Exploration to Exploitation', 
                                    fontsize=14, fontweight='bold')
            axes[row, col].set_xlabel('Reward', fontsize=12)
            axes[row, col].set_ylabel('Frequency', fontsize=12)
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'exploration_exploitation.png'), dpi=300)
        plt.close()
        
    def generate_statistical_report(self, eval_results):
        """Generate detailed statistical report"""
        report = []
        report.append("="*80)
        report.append("STATISTICAL ANALYSIS REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        for name, data in eval_results.items():
            report.append(f"\n{name} Algorithm:")
            report.append("-" * 60)
            
            rewards = data['rewards']
            collisions = data['collisions']
            steps = data['steps']
            
            # Basic statistics
            report.append(f"  Reward Statistics:")
            report.append(f"    Mean: {np.mean(rewards):.2f}")
            report.append(f"    Median: {np.median(rewards):.2f}")
            report.append(f"    Std Dev: {np.std(rewards):.2f}")
            report.append(f"    Min: {np.min(rewards):.2f}")
            report.append(f"    Max: {np.max(rewards):.2f}")
            report.append(f"    25th Percentile: {np.percentile(rewards, 25):.2f}")
            report.append(f"    75th Percentile: {np.percentile(rewards, 75):.2f}")
            
            report.append(f"\n  Collision Statistics:")
            report.append(f"    Mean: {np.mean(collisions):.2f}")
            report.append(f"    Median: {np.median(collisions):.2f}")
            report.append(f"    Zero Collision Episodes: {sum(c == 0 for c in collisions)}/{len(collisions)}")
            
            report.append(f"\n  Efficiency Metrics:")
            report.append(f"    Avg Steps: {np.mean(steps):.2f}")
            report.append(f"    Success Rate: {data['success_rate']:.2f}%")
            report.append(f"    Avg Time to Goal: {data['avg_time_to_goal']:.2f}s")
            
            # Performance consistency
            consistency = 1 - (np.std(rewards) / (np.mean(rewards) + 1e-8))
            report.append(f"\n  Consistency Score: {consistency:.3f} (1.0 = perfect)")
        
        report.append("\n" + "="*80)
        
        # Save report
        report_path = os.path.join(self.results_dir, 'statistical_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print('\n'.join(report))
        return report
    
    def create_comparative_heatmap(self, eval_results):
        """Create heatmap comparing all algorithms across metrics"""
        algorithms = list(eval_results.keys())
        metrics = ['Avg Reward', 'Success Rate', 'Avg Collisions', 
                  'Avg Steps', 'Consistency']
        
        # Normalize metrics for comparison
        data_matrix = []
        for alg in algorithms:
            results = eval_results[alg]
            row = [
                np.mean(results['rewards']),
                results['success_rate'],
                -np.mean(results['collisions']),  # Negative because lower is better
                -np.mean(results['steps']),  # Negative because lower is better
                1 - (np.std(results['rewards']) / (np.mean(results['rewards']) + 1e-8))
            ]
            data_matrix.append(row)
        
        # Normalize to 0-1 range
        data_matrix = np.array(data_matrix)
        for j in range(data_matrix.shape[1]):
            col = data_matrix[:, j]
            data_matrix[:, j] = (col - col.min()) / (col.max() - col.min() + 1e-8)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto')
        
        # Set ticks
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(algorithms)))
        ax.set_xticklabels(metrics, fontsize=12)
        ax.set_yticklabels(algorithms, fontsize=12)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add values to cells
        for i in range(len(algorithms)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{data_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=10)
        
        ax.set_title("Algorithm Performance Heatmap (Normalized Metrics)", 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Performance Score', rotation=270, labelpad=20, fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'performance_heatmap.png'), dpi=300)
        plt.close()
    
    def generate_full_report(self, training_data, eval_results):
        """Generate complete analysis report"""
        print("\nGenerating comprehensive analysis report...")
        
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 1. Convergence Analysis
        print("  - Analyzing convergence characteristics...")
        self.analyze_convergence(training_data)
        
        # 2. Sample Efficiency
        print("  - Analyzing sample efficiency...")
        self.analyze_sample_efficiency(training_data)
        
        # 3. Stability Analysis
        print("  - Analyzing training stability...")
        self.analyze_stability(training_data)
        
        # 4. Exploration vs Exploitation
        print("  - Analyzing exploration-exploitation trade-off...")
        self.analyze_exploration_exploitation(training_data)
        
        # 5. Statistical Report
        print("  - Generating statistical report...")
        self.generate_statistical_report(eval_results)
        
        # 6. Comparative Heatmap
        print("  - Creating performance heatmap...")
        self.create_comparative_heatmap(eval_results)
        
        print(f"\nAll analysis reports saved to '{self.results_dir}/' directory")
        print("\nGenerated files:")
        print("  - convergence_analysis.png")
        print("  - sample_efficiency.png")
        print("  - stability_analysis.png")
        print("  - exploration_exploitation.png")
        print("  - performance_heatmap.png")
        print("  - statistical_report.txt")

# Example usage
if __name__ == '__main__':
    # This would typically be called after training and evaluation
    # For demonstration, creating dummy data
    
    analyzer = PerformanceAnalyzer()
    
    # Dummy training data (replace with actual data from train.py)
    training_data = {
        'DQN': ([np.random.randn() * 10 + 50 + i*0.1 for i in range(500)], [], [], []),
        'DDPG': ([np.random.randn() * 8 + 60 + i*0.12 for i in range(500)], [], [], []),
        'PPO': ([np.random.randn() * 6 + 55 + i*0.11 for i in range(500)], [], [], []),
        'SAC': ([np.random.randn() * 7 + 65 + i*0.13 for i in range(500)], [], [], [])
    }
    
    # Dummy evaluation results (replace with actual data from evaluate.py)
    eval_results = {
        'DQN': {'rewards': [50 + np.random.randn()*10 for _ in range(100)],
                'collisions': [np.random.randint(0, 3) for _ in range(100)],
                'steps': [np.random.randint(100, 200) for _ in range(100)],
                'success_rate': 60, 'avg_time_to_goal': 15.5},
        'DDPG': {'rewards': [60 + np.random.randn()*8 for _ in range(100)],
                 'collisions': [np.random.randint(0, 2) for _ in range(100)],
                 'steps': [np.random.randint(90, 180) for _ in range(100)],
                 'success_rate': 70, 'avg_time_to_goal': 13.2},
        'PPO': {'rewards': [55 + np.random.randn()*6 for _ in range(100)],
                'collisions': [np.random.randint(0, 2) for _ in range(100)],
                'steps': [np.random.randint(95, 190) for _ in range(100)],
                'success_rate': 75, 'avg_time_to_goal': 14.1},
        'SAC': {'rewards': [65 + np.random.randn()*7 for _ in range(100)],
                'collisions': [np.random.randint(0, 2) for _ in range(100)],
                'steps': [np.random.randint(85, 170) for _ in range(100)],
                'success_rate': 85, 'avg_time_to_goal': 11.8}
    }
    
    analyzer.generate_full_report(training_data, eval_results)