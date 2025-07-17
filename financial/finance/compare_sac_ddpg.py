import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import dill
from sac_agent import SACAgent
from ddpg_agent import Agent as DDPGAgent
from syntheticChrissAlmgren import MarketEnvironment

def train_agent(agent_type, episodes=500, max_steps=60, seed=42):
    """
    Train either SAC or DDPG agent.
    
    Params
    ======
        agent_type (str): 'sac' or 'ddpg'
        episodes (int): number of training episodes
        max_steps (int): maximum steps per episode
        seed (int): random seed for reproducibility
    """
    
    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Initialize environment with GBM price model
    env = MarketEnvironment(
        randomSeed=seed,
        price_model='gbm',
        mu=0.0,
        alpha=2.0,
        leftover_penalty=1e-3
    )
    
    # Get state and action dimensions
    state_size = env.observation_space_dimension()
    action_size = env.action_space_dimension()
    
    # Initialize agent
    if agent_type.lower() == 'sac':
        agent = SACAgent(state_size, action_size, seed, automatic_entropy_tuning=True)
        print(f"Training SAC agent...")
    elif agent_type.lower() == 'ddpg':
        agent = DDPGAgent(state_size, action_size, seed)
        print(f"Training DDPG agent...")
    else:
        raise ValueError("agent_type must be 'sac' or 'ddpg'")
    
    # Training metrics
    scores = []
    avg_scores = []
    implementation_shortfalls = []
    
    for episode in range(1, episodes + 1):
        # Reset environment
        state = env.reset(seed=seed + episode)
        env.start_transactions()
        
        score = 0
        episode_shortfall = 0
        
        for step in range(max_steps):
            # Select action
            action = agent.act(state, add_noise=True)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Store experience in replay buffer and learn
            agent.step(state, action, reward, next_state, done)
            
            # Update state and score
            state = next_state
            score += reward[0]
            
            if done:
                episode_shortfall = info.implementation_shortfall
                break
        
        # Store metrics
        scores.append(score)
        implementation_shortfalls.append(episode_shortfall)
        
        # Calculate average score over last 100 episodes
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        
        # Print progress
        if episode % 100 == 0:
            print(f'Episode {episode}/{episodes} | Score: {score:.4f} | Avg Score: {avg_score:.4f} | '
                  f'Shortfall: {episode_shortfall:.2f}')
    
    return agent, scores, avg_scores, implementation_shortfalls

def evaluate_agent(agent, agent_type, episodes=100, seed=42):
    """
    Evaluate a trained agent.
    
    Params
    ======
        agent: trained agent
        agent_type (str): 'sac' or 'ddpg'
        episodes (int): number of evaluation episodes
        seed (int): random seed for reproducibility
    """
    
    # Initialize environment
    env = MarketEnvironment(
        randomSeed=seed,
        price_model='gbm',
        mu=0.0,
        alpha=2.0,
        leftover_penalty=1e-3
    )
    
    evaluation_scores = []
    evaluation_shortfalls = []
    
    print(f"Evaluating {agent_type.upper()} agent...")
    
    for episode in range(episodes):
        # Reset environment
        state = env.reset(seed=seed + episode + 1000)  # Different seed for evaluation
        env.start_transactions()
        
        score = 0
        episode_shortfall = 0
        
        for step in range(60):  # Max 60 steps
            # Select action (no noise during evaluation)
            action = agent.act(state, add_noise=False)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Update state and score
            state = next_state
            score += reward[0]
            
            if done:
                episode_shortfall = info.implementation_shortfall
                break
        
        evaluation_scores.append(score)
        evaluation_shortfalls.append(episode_shortfall)
    
    # Print evaluation results
    print(f"{agent_type.upper()} Evaluation Results ({episodes} episodes):")
    print(f"Average Score: {np.mean(evaluation_scores):.4f} ± {np.std(evaluation_scores):.4f}")
    print(f"Average Implementation Shortfall: {np.mean(evaluation_shortfalls):.2f} ± {np.std(evaluation_shortfalls):.2f}")
    
    return evaluation_scores, evaluation_shortfalls

def plot_comparison(sac_scores, sac_avg_scores, sac_shortfalls, 
                   ddpg_scores, ddpg_avg_scores, ddpg_shortfalls):
    """Plot comparison between SAC and DDPG."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training scores
    ax1.plot(sac_scores, alpha=0.6, label='SAC Episode Score', color='blue')
    ax1.plot(sac_avg_scores, label='SAC Average Score', color='darkblue', linewidth=2)
    ax1.plot(ddpg_scores, alpha=0.6, label='DDPG Episode Score', color='red')
    ax1.plot(ddpg_avg_scores, label='DDPG Average Score', color='darkred', linewidth=2)
    ax1.set_title('Training Scores Comparison')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.legend()
    ax1.grid(True)
    
    # Plot implementation shortfalls
    ax2.plot(sac_shortfalls, alpha=0.6, label='SAC Shortfall', color='blue')
    ax2.plot(ddpg_shortfalls, alpha=0.6, label='DDPG Shortfall', color='red')
    ax2.set_title('Implementation Shortfalls Comparison')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Shortfall ($)')
    ax2.legend()
    ax2.grid(True)
    
    # Plot average scores comparison
    ax3.plot(sac_avg_scores, label='SAC', color='blue', linewidth=2)
    ax3.plot(ddpg_avg_scores, label='DDPG', color='red', linewidth=2)
    ax3.set_title('Average Scores (100 episodes)')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Average Score')
    ax3.legend()
    ax3.grid(True)
    
    # Plot shortfall comparison (smoothed)
    sac_smooth = np.convolve(sac_shortfalls, np.ones(20)/20, mode='valid')
    ddpg_smooth = np.convolve(ddpg_shortfalls, np.ones(20)/20, mode='valid')
    ax4.plot(sac_smooth, label='SAC (smoothed)', color='blue', linewidth=2)
    ax4.plot(ddpg_smooth, label='DDPG (smoothed)', color='red', linewidth=2)
    ax4.set_title('Implementation Shortfalls (Smoothed)')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Shortfall ($)')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('sac_ddpg_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main comparison function."""
    
    print("=== SAC vs DDPG Comparison ===")
    print("Environment: Synthetic Chriss-Almgren with GBM price model")
    print("Fee: 1% (commission)")
    print("=" * 50)
    
    # Train SAC agent
    print("\nTraining SAC agent...")
    sac_agent, sac_scores, sac_avg_scores, sac_shortfalls = train_agent(
        'sac', episodes=500, max_steps=60, seed=42
    )
    
    # Train DDPG agent
    print("\nTraining DDPG agent...")
    ddpg_agent, ddpg_scores, ddpg_avg_scores, ddpg_shortfalls = train_agent(
        'ddpg', episodes=500, max_steps=60, seed=42
    )
    
    # Save trained agents
    print("\nSaving trained agents...")
    with open('trained_sac_agent_comparison.pkl', 'wb') as f:
        pickle.dump(sac_agent, f)
    with open('trained_ddpg_agent_comparison.pkl', 'wb') as f:
        pickle.dump(ddpg_agent, f)
    
    # Evaluate agents
    print("\nEvaluating agents...")
    sac_eval_scores, sac_eval_shortfalls = evaluate_agent(sac_agent, 'sac', episodes=100, seed=42)
    ddpg_eval_scores, ddpg_eval_shortfalls = evaluate_agent(ddpg_agent, 'ddpg', episodes=100, seed=42)
    
    # Plot comparison
    print("\nGenerating comparison plots...")
    plot_comparison(sac_scores, sac_avg_scores, sac_shortfalls,
                   ddpg_scores, ddpg_avg_scores, ddpg_shortfalls)
    
    # Print final comparison
    print("\n=== Final Comparison ===")
    print(f"SAC - Training Avg Score: {np.mean(sac_avg_scores[-100:]):.4f}")
    print(f"DDPG - Training Avg Score: {np.mean(ddpg_avg_scores[-100:]):.4f}")
    print(f"SAC - Evaluation Avg Score: {np.mean(sac_eval_scores):.4f}")
    print(f"DDPG - Evaluation Avg Score: {np.mean(ddpg_eval_scores):.4f}")
    print(f"SAC - Evaluation Avg Shortfall: {np.mean(sac_eval_shortfalls):.2f}")
    print(f"DDPG - Evaluation Avg Shortfall: {np.mean(ddpg_eval_shortfalls):.2f}")
    
    print("\nComparison completed!")
    print("Files saved:")
    print("- trained_sac_agent_comparison.pkl")
    print("- trained_ddpg_agent_comparison.pkl")
    print("- sac_ddpg_comparison.png")

if __name__ == "__main__":
    main() 