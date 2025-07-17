import numpy as np
import torch
import matplotlib.pyplot as plt
import dill
import pickle
from sac_agent import SACAgent
from syntheticChrissAlmgren import MarketEnvironment
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def train_sac_comprehensive(episodes=5000, train_interval=1000, eval_episodes=100, max_steps=60, seed=42, 
                           enable_quick_test=True, quick_test_episodes=200):
    """
    Comprehensive SAC training with parallel environment evaluation for better generalization.
    
    Params
    ======
        episodes (int): total number of training episodes
        train_interval (int): train for N episodes then evaluate
        eval_episodes (int): number of evaluation episodes per evaluation phase
        max_steps (int): maximum steps per episode
        seed (int): random seed for reproducibility
    """
    
    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Initialize training environment with GBM price model
    train_env = MarketEnvironment(
        randomSeed=seed,
        price_model='gbm',
        mu=0.0,
        alpha=2.0,
        leftover_penalty=1e-3
    )
    
    # Get state and action dimensions
    state_size = train_env.observation_space_dimension()
    action_size = train_env.action_space_dimension()
    
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    
    # Initialize SAC agent
    agent = SACAgent(state_size, action_size, seed, automatic_entropy_tuning=True)
    
    # Training metrics storage
    training_scores = []
    training_shortfalls = []
    training_captures = []
    training_expected_shortfalls = []
    training_utilities = []
    
    # Evaluation metrics storage
    eval_scores = []
    eval_shortfalls = []
    eval_captures = []
    eval_expected_shortfalls = []
    eval_utilities = []
    eval_checkpoints = []
    
    # Almgren-Chriss theoretical results (using training env as reference)
    ac_expected_shortfall = train_env.get_AC_expected_shortfall(train_env.total_shares)
    ac_variance = train_env.get_AC_variance(train_env.total_shares)
    ac_utility = train_env.compute_AC_utility(train_env.total_shares)
    ac_trade_list = train_env.get_trade_list()
    
    print(f"Almgren-Chriss Theoretical Results:")
    print(f"Expected Shortfall: ${ac_expected_shortfall:.2f}")
    print(f"Variance: {ac_variance:.2f}")
    print(f"Utility: {ac_utility:.2f}")
    
    print(f"\nStarting comprehensive SAC training for {episodes} episodes...")
    print(f"Training interval: {train_interval} episodes")
    print(f"Evaluation episodes: {eval_episodes} per evaluation phase")
    print(f"Logging every 100 episodes")
    print(f"New evaluation environment for each evaluation phase")
    print(f"Evaluation uses noise=0 for accurate measurement")
    print(f"Off-policy learning allows environment switching without harm")
    print(f"Quick test enabled: {enable_quick_test}")
    if enable_quick_test:
        print(f"Quick test episodes: {quick_test_episodes}")
    
    current_episode = 0
    
    while current_episode < episodes:
        # Training phase
        print(f"\n=== Training Phase {len(eval_checkpoints) + 1} ===")
        print(f"Training episodes {current_episode + 1} to {min(current_episode + train_interval, episodes)}")
        
        for episode in range(train_interval):
            if current_episode >= episodes:
                break
                
            current_episode += 1
            
            # Reset training environment
            state = train_env.reset(seed=seed + current_episode)
            train_env.start_transactions()
            
            score = 0
            episode_shortfall = 0
            episode_capture = 0
            episode_expected_shortfall = 0
            episode_utility = 0
            
            for step in range(max_steps):
                # Select action
                action = agent.act(state, add_noise=True)
                
                # Take action in training environment
                next_state, reward, done, info = train_env.step(action)
                
                # Store experience in replay buffer and learn
                agent.step(state, action, reward, next_state, done)
                
                # Update state and score
                state = next_state
                score += reward[0]
                
                if done:
                    episode_shortfall = info.implementation_shortfall
                    episode_capture = train_env.totalCapture
                    episode_expected_shortfall = info.expected_shortfall if hasattr(info, 'expected_shortfall') else 0.0
                    episode_utility = info.utility if hasattr(info, 'utility') else 0.0
                    break
            
            # Store training metrics
            training_scores.append(score)
            training_shortfalls.append(episode_shortfall)
            training_captures.append(episode_capture)
            training_expected_shortfalls.append(episode_expected_shortfall)
            training_utilities.append(episode_utility)
            
            # Log progress every 100 episodes
            if current_episode % 100 == 0:
                avg_score = np.mean(training_scores[-100:])
                avg_shortfall = np.mean(training_shortfalls[-100:])
                print(f'Episode {current_episode}/{episodes} | Score: {score:.4f} | Avg Score (last 100): {avg_score:.4f} | '
                      f'Shortfall: {episode_shortfall:.2f} | Avg Shortfall (last 100): {avg_shortfall:.2f}')
        
        # Evaluation phase (only if we haven't reached the end)
        if current_episode < episodes:
            print(f"\n=== Evaluation Phase {len(eval_checkpoints) + 1} ===")
            print(f"Evaluating after {current_episode} training episodes")
            
            # Create NEW evaluation environment (different from training env)
            eval_env = MarketEnvironment(
                randomSeed=seed + current_episode + 20000 + len(eval_checkpoints) * 1000,  # Different seed for each eval
                price_model='gbm',
                mu=0.0,
                alpha=2.0,
                leftover_penalty=1e-3
            )
            
            eval_score, eval_shortfall, eval_capture, eval_expected_shortfall, eval_utility = evaluate_agent_no_noise(
                agent, eval_env, episodes=eval_episodes, seed=seed + current_episode + 30000
            )
            
            eval_scores.append(eval_score)
            eval_shortfalls.append(eval_shortfall)
            eval_captures.append(eval_capture)
            eval_expected_shortfalls.append(eval_expected_shortfall)
            eval_utilities.append(eval_utility)
            eval_checkpoints.append(current_episode)
            
            print(f"Training Avg Score (last 100): {np.mean(training_scores[-100:]):.4f}")
            print(f"Evaluation Avg Score: {eval_score:.4f}")
            print(f"Training Avg Shortfall (last 100): {np.mean(training_shortfalls[-100:]):.2f}")
            print(f"Evaluation Avg Shortfall: {eval_shortfall:.2f}")
            print(f"AC Theoretical Shortfall: ${ac_expected_shortfall:.2f}")
            print(f"Improvement over AC: {((ac_expected_shortfall - eval_shortfall) / ac_expected_shortfall * 100):.2f}%")
            print(f"Generalization test: Using different environment for evaluation")
    
    print("\nTraining completed!")
    
    # Save the trained agent weights
    print("Saving trained agent weights...")
    
    try:
        # Save model weights as torch state dicts
        torch.save({
            'actor_state_dict': agent.actor.state_dict(),
            'critic1_state_dict': agent.critic1.state_dict(),
            'critic2_state_dict': agent.critic2.state_dict(),
            'critic1_target_state_dict': agent.critic1_target.state_dict(),
            'critic2_target_state_dict': agent.critic2_target.state_dict(),
            'log_alpha': agent.log_alpha.data.clone() if agent.automatic_entropy_tuning else None,
            'state_size': state_size,
            'action_size': action_size,
            'seed': seed,
            'automatic_entropy_tuning': agent.automatic_entropy_tuning
        }, 'trained_sac_comprehensive_weights.pth')
        print("Agent weights saved successfully as .pth")
    except Exception as e:
        print(f"Failed to save weights: {e}")
    
    # Save training results as text files
    try:
        import json
        results = {
            'training_scores': training_scores,
            'training_shortfalls': training_shortfalls,
            'training_captures': training_captures,
            'eval_scores': eval_scores,
            'eval_shortfalls': eval_shortfalls,
            'eval_captures': eval_captures,
            'eval_checkpoints': eval_checkpoints,
            'ac_expected_shortfall': ac_expected_shortfall,
            'ac_variance': ac_variance,
            'ac_utility': ac_utility
        }
        with open('training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("Training results saved as training_results.json")
    except Exception as e:
        print(f"Failed to save results: {e}")
    
    # Create comprehensive plots
    try:
        create_comprehensive_plots(
            training_scores, training_shortfalls, training_captures, training_expected_shortfalls, training_utilities,
            eval_scores, eval_shortfalls, eval_captures, eval_expected_shortfalls, eval_utilities, eval_checkpoints,
            ac_expected_shortfall, ac_variance, ac_utility, ac_trade_list, train_env
        )
        print("Plots created successfully")
    except Exception as e:
        print(f"Failed to create plots: {e}")
        print("Training completed but plotting failed")
    
    return agent, training_scores, eval_scores, eval_checkpoints

def test_trained_agent(agent, episodes=200, seed=42):
    """
    Test the trained agent for correctness verification.
    
    Params
    ======
        agent: trained SAC agent
        episodes (int): number of test episodes
        seed (int): random seed for reproducibility
    """
    
    print(f"\n=== TESTING TRAINED AGENT ===")
    print(f"Testing for {episodes} episodes to verify correctness...")
    
    # Initialize test environment
    test_env = MarketEnvironment(
        randomSeed=seed + 50000,  # Different seed for testing
        price_model='gbm',
        mu=0.0,
        alpha=2.0,
        leftover_penalty=1e-3
    )
    
    test_scores = []
    test_shortfalls = []
    test_captures = []
    test_expected_shortfalls = []
    test_utilities = []
    
    # Get AC theoretical results for comparison
    ac_expected_shortfall = test_env.get_AC_expected_shortfall(test_env.total_shares)
    ac_utility = test_env.compute_AC_utility(test_env.total_shares)
    
    print(f"AC Theoretical Shortfall: ${ac_expected_shortfall:.2f}")
    print(f"AC Theoretical Utility: {ac_utility:.2f}")
    
    for episode in range(episodes):
        # Reset environment
        state = test_env.reset(seed=seed + 50000 + episode)
        test_env.start_transactions()
        
        score = 0
        episode_shortfall = 0
        episode_capture = 0
        episode_expected_shortfall = 0
        episode_utility = 0
        
        for step in range(60):  # Max 60 steps
            # Select action (no noise during testing)
            action = agent.act(state, add_noise=False)
            
            # Take action in environment
            next_state, reward, done, info = test_env.step(action)
            
            # Update state and score
            state = next_state
            score += reward[0]
            
            if done:
                episode_shortfall = info.implementation_shortfall
                episode_capture = test_env.totalCapture
                episode_expected_shortfall = info.expected_shortfall if hasattr(info, 'expected_shortfall') else 0.0
                episode_utility = info.utility if hasattr(info, 'utility') else 0.0
                break
        
        test_scores.append(score)
        test_shortfalls.append(episode_shortfall)
        test_captures.append(episode_capture)
        test_expected_shortfalls.append(episode_expected_shortfall)
        test_utilities.append(episode_utility)
        
        # Print progress every 50 episodes
        if (episode + 1) % 50 == 0:
            avg_score = np.mean(test_scores)
            avg_shortfall = np.mean(test_shortfalls)
            print(f'Test Episode {episode + 1}/{episodes} | Avg Score: {avg_score:.4f} | Avg Shortfall: ${avg_shortfall:.2f}')
    
    # Calculate final statistics
    final_avg_score = np.mean(test_scores)
    final_std_score = np.std(test_scores)
    final_avg_shortfall = np.mean(test_shortfalls)
    final_std_shortfall = np.std(test_shortfalls)
    final_avg_capture = np.mean(test_captures)
    final_avg_utility = np.mean(test_utilities)
    
    improvement_over_ac = ((ac_expected_shortfall - final_avg_shortfall) / ac_expected_shortfall * 100)
    
    print(f"\n=== TEST RESULTS ({episodes} episodes) ===")
    print(f"Average Score: {final_avg_score:.4f} ± {final_std_score:.4f}")
    print(f"Average Shortfall: ${final_avg_shortfall:.2f} ± ${final_std_shortfall:.2f}")
    print(f"Average Capture: ${final_avg_capture:,.0f}")
    print(f"Average Utility: {final_avg_utility:.2f}")
    print(f"AC Theoretical Shortfall: ${ac_expected_shortfall:.2f}")
    print(f"Improvement over AC: {improvement_over_ac:.2f}%")
    
    # Create test results plot
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Test scores distribution
        ax1.hist(test_scores, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(final_avg_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {final_avg_score:.4f}')
        ax1.set_title('Test Scores Distribution')
        ax1.set_xlabel('Score')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True)
        
        # Test shortfalls distribution
        ax2.hist(test_shortfalls, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(final_avg_shortfall, color='red', linestyle='--', linewidth=2, label=f'Mean: ${final_avg_shortfall:.2f}')
        ax2.axvline(ac_expected_shortfall, color='blue', linestyle='--', linewidth=2, label=f'AC: ${ac_expected_shortfall:.2f}')
        ax2.set_title('Test Shortfalls Distribution')
        ax2.set_xlabel('Shortfall ($)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True)
        
        # Test captures
        ax3.plot(test_captures, alpha=0.6, color='purple')
        ax3.axhline(final_avg_capture, color='red', linestyle='--', linewidth=2, label=f'Avg: ${final_avg_capture:,.0f}')
        ax3.axhline(test_env.total_shares * test_env.startingPrice, color='green', linestyle='--', linewidth=2, 
                   label=f'Perfect: ${test_env.total_shares * test_env.startingPrice:,.0f}')
        ax3.set_title('Test Captures')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Capture ($)')
        ax3.legend()
        ax3.grid(True)
        
        # Performance summary
        categories = ['SAC Test', 'AC Theoretical']
        scores_data = [final_avg_score, 0]  # AC doesn't have a score
        shortfalls_data = [final_avg_shortfall, ac_expected_shortfall]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, scores_data, width, label='Score', color='blue', alpha=0.7)
        bars2 = ax4.bar(x + width/2, shortfalls_data, width, label='Shortfall ($)', color='red', alpha=0.7)
        
        ax4.set_xlabel('Method')
        ax4.set_ylabel('Value')
        ax4.set_title('Test Performance Summary')
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories)
        ax4.legend()
        ax4.grid(True, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('sac_test_results.jpg', dpi=300, bbox_inches='tight', format='jpg')
        plt.show()
        
        print("Test results plot saved as sac_test_results.jpg")
        
    except Exception as e:
        print(f"Failed to create test plots: {e}")
    
    return {
        'avg_score': final_avg_score,
        'std_score': final_std_score,
        'avg_shortfall': final_avg_shortfall,
        'std_shortfall': final_std_shortfall,
        'avg_capture': final_avg_capture,
        'avg_utility': final_avg_utility,
        'improvement_over_ac': improvement_over_ac,
        'test_scores': test_scores,
        'test_shortfalls': test_shortfalls,
        'test_captures': test_captures
    }

def evaluate_agent_no_noise(agent, env, episodes=100, seed=42):
    """
    Evaluate agent without noise for accurate performance measurement.
    """
    
    evaluation_scores = []
    evaluation_shortfalls = []
    evaluation_captures = []
    evaluation_expected_shortfalls = []
    evaluation_utilities = []
    
    print(f"  Starting evaluation with {episodes} episodes (noise=0)...")
    
    for episode in range(episodes):
        # Reset environment
        state = env.reset(seed=seed + episode)
        env.start_transactions()
        
        score = 0
        episode_shortfall = 0
        episode_capture = 0
        episode_expected_shortfall = 0
        episode_utility = 0
        
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
                episode_capture = env.totalCapture
                episode_expected_shortfall = info.expected_shortfall if hasattr(info, 'expected_shortfall') else 0.0
                episode_utility = info.utility if hasattr(info, 'utility') else 0.0
                break
        
        evaluation_scores.append(score)
        evaluation_shortfalls.append(episode_shortfall)
        evaluation_captures.append(episode_capture)
        evaluation_expected_shortfalls.append(episode_expected_shortfall)
        evaluation_utilities.append(episode_utility)
    
    # Calculate statistics
    avg_score = np.mean(evaluation_scores)
    avg_shortfall = np.mean(evaluation_shortfalls)
    avg_capture = np.mean(evaluation_captures)
    avg_expected_shortfall = np.mean(evaluation_expected_shortfalls)
    avg_utility = np.mean(evaluation_utilities)
    
    std_score = np.std(evaluation_scores)
    std_shortfall = np.std(evaluation_shortfalls)
    
    print(f"  Evaluation completed:")
    print(f"    Score: {avg_score:.4f} ± {std_score:.4f}")
    print(f"    Shortfall: ${avg_shortfall:.2f} ± ${std_shortfall:.2f}")
    print(f"    Capture: ${avg_capture:,.0f}")
    print(f"    Expected Shortfall: ${avg_expected_shortfall:.2f}")
    print(f"    Utility: {avg_utility:.2f}")
    
    return (avg_score, avg_shortfall, avg_capture, avg_expected_shortfall, avg_utility)

def create_comprehensive_plots(training_scores, training_shortfalls, training_captures, 
                              training_expected_shortfalls, training_utilities,
                              eval_scores, eval_shortfalls, eval_captures, 
                              eval_expected_shortfalls, eval_utilities, eval_checkpoints,
                              ac_expected_shortfall, ac_variance, ac_utility, ac_trade_list, env):
    """Create comprehensive plots including comparison with Almgren-Chriss results."""
    
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Training Scores
    ax1 = plt.subplot(4, 3, 1)
    ax1.plot(training_scores, alpha=0.6, label='Episode Score', color='blue')
    ax1.plot(np.convolve(training_scores, np.ones(100)/100, mode='valid'), 
             label='Moving Average (100)', color='red', linewidth=2)
    ax1.set_title('Training Scores')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Training vs Evaluation Scores
    ax2 = plt.subplot(4, 3, 2)
    ax2.plot(eval_checkpoints, eval_scores, 'o-', label='Evaluation Score', color='green', linewidth=2, markersize=6)
    ax2.axhline(y=np.mean(training_scores[-100:]), color='red', linestyle='--', 
                label=f'Training Avg (last 100): {np.mean(training_scores[-100:]):.4f}')
    ax2.set_title('Training vs Evaluation Scores')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Implementation Shortfalls
    ax3 = plt.subplot(4, 3, 3)
    ax3.plot(training_shortfalls, alpha=0.6, label='Training Shortfall', color='blue')
    ax3.plot(np.convolve(training_shortfalls, np.ones(100)/100, mode='valid'), 
             label='Moving Average (100)', color='red', linewidth=2)
    ax3.axhline(y=ac_expected_shortfall, color='green', linestyle='--', 
                label=f'AC Theoretical: ${ac_expected_shortfall:.2f}')
    ax3.set_title('Implementation Shortfalls')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Shortfall ($)')
    ax3.legend()
    ax3.grid(True)
    
    # 4. Training vs Evaluation Shortfalls
    ax4 = plt.subplot(4, 3, 4)
    ax4.plot(eval_checkpoints, eval_shortfalls, 'o-', label='Evaluation Shortfall', color='green', linewidth=2, markersize=6)
    ax4.axhline(y=np.mean(training_shortfalls[-100:]), color='red', linestyle='--', 
                label=f'Training Avg (last 100): ${np.mean(training_shortfalls[-100:]):.2f}')
    ax4.axhline(y=ac_expected_shortfall, color='blue', linestyle='--', 
                label=f'AC Theoretical: ${ac_expected_shortfall:.2f}')
    ax4.set_title('Shortfall Comparison')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Shortfall ($)')
    ax4.legend()
    ax4.grid(True)
    
    # 5. Trading Capture
    ax5 = plt.subplot(4, 3, 5)
    ax5.plot(training_captures, alpha=0.6, label='Trading Capture', color='blue')
    ax5.plot(np.convolve(training_captures, np.ones(100)/100, mode='valid'), 
             label='Moving Average (100)', color='red', linewidth=2)
    ax5.axhline(y=env.total_shares * env.startingPrice, color='green', linestyle='--', 
                label=f'Perfect Capture: ${env.total_shares * env.startingPrice:,.0f}')
    ax5.set_title('Trading Capture')
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Capture ($)')
    ax5.legend()
    ax5.grid(True)
    
    # 6. Training vs Evaluation Capture
    ax6 = plt.subplot(4, 3, 6)
    ax6.plot(eval_checkpoints, eval_captures, 'o-', label='Evaluation Capture', color='green', linewidth=2, markersize=6)
    ax6.axhline(y=np.mean(training_captures[-100:]), color='red', linestyle='--', 
                label=f'Training Avg (last 100): ${np.mean(training_captures[-100:]):,.0f}')
    ax6.axhline(y=env.total_shares * env.startingPrice, color='blue', linestyle='--', 
                label=f'Perfect Capture: ${env.total_shares * env.startingPrice:,.0f}')
    ax6.set_title('Capture Comparison')
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Capture ($)')
    ax6.legend()
    ax6.grid(True)
    
    # 7. Expected Shortfalls
    ax7 = plt.subplot(4, 3, 7)
    ax7.plot(training_expected_shortfalls, alpha=0.6, label='Training Expected Shortfall', color='blue')
    ax7.plot(np.convolve(training_expected_shortfalls, np.ones(100)/100, mode='valid'), 
             label='Moving Average (100)', color='red', linewidth=2)
    ax7.axhline(y=ac_expected_shortfall, color='green', linestyle='--', 
                label=f'AC Theoretical: ${ac_expected_shortfall:.2f}')
    ax7.set_title('Expected Shortfalls')
    ax7.set_xlabel('Episode')
    ax7.set_ylabel('Expected Shortfall ($)')
    ax7.legend()
    ax7.grid(True)
    
    # 8. Utilities
    ax8 = plt.subplot(4, 3, 8)
    ax8.plot(training_utilities, alpha=0.6, label='Training Utility', color='blue')
    ax8.plot(np.convolve(training_utilities, np.ones(100)/100, mode='valid'), 
             label='Moving Average (100)', color='red', linewidth=2)
    ax8.axhline(y=ac_utility, color='green', linestyle='--', 
                label=f'AC Theoretical: {ac_utility:.2f}')
    ax8.set_title('Utilities')
    ax8.set_xlabel('Episode')
    ax8.set_ylabel('Utility')
    ax8.legend()
    ax8.grid(True)
    
    # 9. Almgren-Chriss Optimal Trade List
    ax9 = plt.subplot(4, 3, 9)
    ax9.plot(ac_trade_list, 'o-', label='AC Optimal Trade List', color='purple', linewidth=2, markersize=4)
    ax9.set_title('Almgren-Chriss Optimal Trade List')
    ax9.set_xlabel('Trade Number')
    ax9.set_ylabel('Shares to Sell')
    ax9.legend()
    ax9.grid(True)
    
    # 10. Performance Distribution
    ax10 = plt.subplot(4, 3, 10)
    ax10.hist(training_shortfalls[-1000:], bins=50, alpha=0.7, label='Training Shortfalls (last 1000)', color='blue')
    ax10.axvline(x=ac_expected_shortfall, color='red', linestyle='--', linewidth=2,
                 label=f'AC Theoretical: ${ac_expected_shortfall:.2f}')
    ax10.axvline(x=np.mean(training_shortfalls[-1000:]), color='green', linestyle='--', linewidth=2,
                 label=f'Mean: ${np.mean(training_shortfalls[-1000:]):.2f}')
    ax10.set_title('Shortfall Distribution (Last 1000 Episodes)')
    ax10.set_xlabel('Shortfall ($)')
    ax10.set_ylabel('Frequency')
    ax10.legend()
    ax10.grid(True)
    
    # 11. Learning Progress
    ax11 = plt.subplot(4, 3, 11)
    window = 500
    moving_avg_scores = np.convolve(training_scores, np.ones(window)/window, mode='valid')
    moving_avg_shortfalls = np.convolve(training_shortfalls, np.ones(window)/window, mode='valid')
    episodes_ma = np.arange(window, len(training_scores) + 1)
    
    ax11_twin = ax11.twinx()
    line1 = ax11.plot(episodes_ma, moving_avg_scores, label='Avg Score', color='blue', linewidth=2)
    line2 = ax11_twin.plot(episodes_ma, moving_avg_shortfalls, label='Avg Shortfall', color='red', linewidth=2)
    
    ax11.set_xlabel('Episode')
    ax11.set_ylabel('Average Score', color='blue')
    ax11_twin.set_ylabel('Average Shortfall ($)', color='red')
    ax11.set_title('Learning Progress')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax11.legend(lines, labels, loc='upper left')
    ax11.grid(True)
    
    # 12. Final Comparison Summary
    ax12 = plt.subplot(4, 3, 12)
    
    # Prepare data for comparison
    categories = ['SAC Training\n(Last 1000)', 'SAC Evaluation\n(Last Eval)', 'AC Theoretical']
    scores_data = [np.mean(training_scores[-1000:]), eval_scores[-1] if eval_scores else 0, 0]
    shortfalls_data = [np.mean(training_shortfalls[-1000:]), eval_shortfalls[-1] if eval_shortfalls else 0, ac_expected_shortfall]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax12.bar(x - width/2, scores_data, width, label='Score', color='blue', alpha=0.7)
    bars2 = ax12.bar(x + width/2, shortfalls_data, width, label='Shortfall ($)', color='red', alpha=0.7)
    
    ax12.set_xlabel('Method')
    ax12.set_ylabel('Value')
    ax12.set_title('Final Performance Comparison')
    ax12.set_xticks(x)
    ax12.set_xticklabels(categories)
    ax12.legend()
    ax12.grid(True, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax12.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('sac_comprehensive_results.jpg', dpi=300, bbox_inches='tight', format='jpg')
    plt.show()
    
    # Create additional summary statistics
    print("\n=== COMPREHENSIVE TRAINING SUMMARY ===")
    print(f"Total Episodes: {len(training_scores)}")
    print(f"Final Training Score (last 100): {np.mean(training_scores[-100:]):.4f}")
    print(f"Final Evaluation Score: {eval_scores[-1] if eval_scores else 'N/A'}")
    print(f"Final Training Shortfall (last 100): ${np.mean(training_shortfalls[-100:]):.2f}")
    print(f"Final Evaluation Shortfall: ${eval_shortfalls[-1] if eval_shortfalls else 'N/A'}")
    print(f"AC Theoretical Shortfall: ${ac_expected_shortfall:.2f}")
    print(f"Improvement over AC: {((ac_expected_shortfall - (eval_shortfalls[-1] if eval_shortfalls else 0)) / ac_expected_shortfall * 100):.2f}%")

if __name__ == "__main__":
    # Train the SAC agent comprehensively
    print("=== COMPREHENSIVE SAC AGENT TRAINING ===")
    print("Environment: Synthetic Chriss-Almgren with GBM price model")
    print("Fee: 1% (commission)")
    print("Training: 2,000 episodes with evaluation every 1,000 episodes")
    print("=" * 60)
    
    agent, training_scores, eval_scores, eval_checkpoints = train_sac_comprehensive(
        episodes=2000,
        train_interval=1000,
        eval_episodes=100,
        max_steps=60,
        seed=42,
        enable_quick_test=True,  # Set to False to disable quick test
        quick_test_episodes=200  # Number of episodes for quick test
    )
    
    print("\nComprehensive training and evaluation completed!")
    print("Files saved:")
    print("- trained_sac_comprehensive_weights.pth")
    print("- training_results.json")
    print("- sac_comprehensive_results.jpg")
    
    # Test the trained agent for correctness verification (if enabled)
    test_results = None
    if enable_quick_test:
        test_results = test_trained_agent(agent, episodes=quick_test_episodes, seed=42)
        
        print("\n=== FINAL SUMMARY ===")
        print(f"Training: {episodes} episodes completed")
        print(f"Testing: {quick_test_episodes} episodes completed")
        print(f"Final Test Score: {test_results['avg_score']:.4f} ± {test_results['std_score']:.4f}")
        print(f"Final Test Shortfall: ${test_results['avg_shortfall']:.2f} ± ${test_results['std_shortfall']:.2f}")
        print(f"Improvement over AC: {test_results['improvement_over_ac']:.2f}%")
        print(f"All results saved as JPG files and JSON data")
    else:
        print("\n=== FINAL SUMMARY ===")
        print(f"Training: {episodes} episodes completed")
        print(f"Quick test: DISABLED")
        print(f"All results saved as JPG files and JSON data") 