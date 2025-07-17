import numpy as np
import torch
import matplotlib.pyplot as plt
import dill
import pickle
from sac_agent import SACAgent
from syntheticChrissAlmgren import MarketEnvironment

def train_sac_agent(episodes=1000, max_steps=60, seed=42):
    """
    Train SAC agent on the synthetic Chriss-Almgren environment.
    
    Params
    ======
        episodes (int): number of training episodes
        max_steps (int): maximum steps per episode
        seed (int): random seed for reproducibility
    """
    
    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Initialize environment with GBM price model and 1% fee
    env = MarketEnvironment(
        randomSeed=seed,
        price_model='gbm',  # Use GBM price model
        mu=0.0,  # No drift in GBM
        alpha=2.0,
        leftover_penalty=1e-3
    )
    
    # Get state and action dimensions
    state_size = env.observation_space_dimension()
    action_size = env.action_space_dimension()
    
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    
    # Initialize SAC agent
    agent = SACAgent(state_size, action_size, seed, automatic_entropy_tuning=True)
    
    # Training metrics
    scores = []
    avg_scores = []
    implementation_shortfalls = []
    expected_shortfalls = []
    
    print("Starting SAC training...")
    
    for episode in range(1, episodes + 1):
        # Reset environment
        state = env.reset(seed=seed + episode)
        env.start_transactions()
        
        score = 0
        episode_shortfall = 0
        episode_expected_shortfall = 0
        
        for step in range(max_steps):
            # Select action
            action = agent.act(state, add_noise=True)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Store experience in replay buffer and learn
            agent.step(state, action, reward, next_state, done)
            
            # Update state and score
            state = next_state
            score += reward[0]  # reward is returned as array
            
            # Store episode information
            if done:
                episode_shortfall = info.implementation_shortfall
                episode_expected_shortfall = info.expected_shortfall if hasattr(info, 'expected_shortfall') else 0.0
                break
        
        # Store metrics
        scores.append(score)
        implementation_shortfalls.append(episode_shortfall)
        expected_shortfalls.append(episode_expected_shortfall)
        
        # Calculate average score over last 100 episodes
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        
        # Print progress
        if episode % 100 == 0:
            print(f'Episode {episode}/{episodes} | Score: {score:.4f} | Avg Score: {avg_score:.4f} | '
                  f'Shortfall: {episode_shortfall:.2f} | Expected: {episode_expected_shortfall:.2f}')
    
    print("Training completed!")
    
    # Save the trained agent
    print("Saving trained agent...")
    with open('trained_sac_agent.pkl', 'wb') as f:
        pickle.dump(agent, f)
    
    with open('trained_sac_agent.dill', 'wb') as f:
        dill.dump(agent, f)
    
    # Plot training results
    plot_training_results(scores, avg_scores, implementation_shortfalls, expected_shortfalls)
    
    return agent, scores, avg_scores, implementation_shortfalls, expected_shortfalls

def plot_training_results(scores, avg_scores, implementation_shortfalls, expected_shortfalls):
    """Plot training results."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot scores
    ax1.plot(scores, alpha=0.6, label='Episode Score')
    ax1.plot(avg_scores, label='Average Score (100 episodes)')
    ax1.set_title('Training Scores')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.legend()
    ax1.grid(True)
    
    # Plot implementation shortfalls
    ax2.plot(implementation_shortfalls, alpha=0.6, label='Implementation Shortfall')
    ax2.set_title('Implementation Shortfalls')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Shortfall ($)')
    ax2.legend()
    ax2.grid(True)
    
    # Plot expected shortfalls
    ax3.plot(expected_shortfalls, alpha=0.6, label='Expected Shortfall')
    ax3.set_title('Expected Shortfalls')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Expected Shortfall ($)')
    ax3.legend()
    ax3.grid(True)
    
    # Plot comparison
    ax4.plot(implementation_shortfalls, alpha=0.6, label='Implementation Shortfall')
    ax4.plot(expected_shortfalls, alpha=0.6, label='Expected Shortfall')
    ax4.set_title('Shortfall Comparison')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Shortfall ($)')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('sac_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_sac_agent(agent, episodes=100, seed=42):
    """
    Evaluate the trained SAC agent.
    
    Params
    ======
        agent: trained SAC agent
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
    evaluation_expected_shortfalls = []
    
    print("Evaluating SAC agent...")
    
    for episode in range(episodes):
        # Reset environment
        state = env.reset(seed=seed + episode + 1000)  # Different seed for evaluation
        env.start_transactions()
        
        score = 0
        episode_shortfall = 0
        episode_expected_shortfall = 0
        
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
                episode_expected_shortfall = info.expected_shortfall if hasattr(info, 'expected_shortfall') else 0.0
                break
        
        evaluation_scores.append(score)
        evaluation_shortfalls.append(episode_shortfall)
        evaluation_expected_shortfalls.append(episode_expected_shortfall)
    
    # Print evaluation results
    print(f"Evaluation Results ({episodes} episodes):")
    print(f"Average Score: {np.mean(evaluation_scores):.4f} ± {np.std(evaluation_scores):.4f}")
    print(f"Average Implementation Shortfall: {np.mean(evaluation_shortfalls):.2f} ± {np.std(evaluation_shortfalls):.2f}")
    print(f"Average Expected Shortfall: {np.mean(evaluation_expected_shortfalls):.2f} ± {np.std(evaluation_expected_shortfalls):.2f}")
    
    return evaluation_scores, evaluation_shortfalls, evaluation_expected_shortfalls

if __name__ == "__main__":
    # Train the SAC agent
    print("=== SAC Agent Training ===")
    print("Environment: Synthetic Chriss-Almgren with GBM price model")
    print("Fee: 1% (commission)")
    print("=" * 50)
    
    agent, scores, avg_scores, shortfalls, expected_shortfalls = train_sac_agent(
        episodes=1000,
        max_steps=60,
        seed=42
    )
    
    # Evaluate the trained agent
    print("\n=== SAC Agent Evaluation ===")
    eval_scores, eval_shortfalls, eval_expected_shortfalls = evaluate_sac_agent(
        agent,
        episodes=100,
        seed=42
    )
    
    print("\nTraining and evaluation completed!")
    print("Files saved:")
    print("- trained_sac_agent.pkl")
    print("- trained_sac_agent.dill")
    print("- sac_training_results.png") 