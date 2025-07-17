import numpy as np
import torch
from sac_agent import SACAgent
from syntheticChrissAlmgren import MarketEnvironment

def test_sac_agent():
    """Test the SAC agent implementation."""
    
    print("Testing SAC Agent Implementation...")
    print("=" * 50)
    
    # Set random seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Initialize environment with GBM price model
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
    
    print(f"Environment initialized:")
    print(f"- State size: {state_size}")
    print(f"- Action size: {action_size}")
    print(f"- Price model: GBM")
    print(f"- Commission: 1% (0.25 per share)")
    
    # Initialize SAC agent
    agent = SACAgent(state_size, action_size, seed, automatic_entropy_tuning=True)
    
    print(f"\nSAC Agent initialized:")
    print(f"- Actor network: {type(agent.actor).__name__}")
    print(f"- Critic networks: {type(agent.critic1).__name__}")
    print(f"- Replay buffer size: {len(agent.memory)}")
    print(f"- Automatic entropy tuning: {agent.automatic_entropy_tuning}")
    
    # Test a single episode
    print(f"\nTesting single episode...")
    
    # Reset environment
    state = env.reset(seed=seed)
    env.start_transactions()
    
    total_reward = 0
    step_count = 0
    
    for step in range(60):  # Max 60 steps
        # Select action
        action = agent.act(state, add_noise=True)
        
        # Take action in environment
        next_state, reward, done, info = env.step(action)
        
        # Store experience in replay buffer and learn
        agent.step(state, action, reward, next_state, done)
        
        # Update state and metrics
        state = next_state
        total_reward += reward[0]
        step_count += 1
        
        print(f"Step {step + 1}: Action={action[0]:.4f}, Reward={reward[0]:.6f}, Done={done}")
        
        if done:
            print(f"Episode finished after {step + 1} steps")
            print(f"Final implementation shortfall: {info.implementation_shortfall:.2f}")
            if hasattr(info, 'expected_shortfall'):
                print(f"Expected shortfall: {info.expected_shortfall:.2f}")
            else:
                print("Expected shortfall: Not available (all shares sold)")
            break
    
    print(f"\nEpisode Summary:")
    print(f"- Total steps: {step_count}")
    print(f"- Total reward: {total_reward:.6f}")
    print(f"- Replay buffer size: {len(agent.memory)}")
    
    # Test learning (if enough samples)
    if len(agent.memory) > agent.memory.batch_size:
        print(f"\nTesting learning with batch size {agent.memory.batch_size}...")
        experiences = agent.memory.sample()
        agent.learn(experiences, 0.99)
        print("Learning step completed successfully!")
    else:
        print(f"\nNot enough samples for learning yet. Need {agent.memory.batch_size}, have {len(agent.memory)}")
    
    # Test action generation without noise
    print(f"\nTesting action generation without noise...")
    test_state = np.random.randn(state_size)
    action_no_noise = agent.act(test_state, add_noise=False)
    action_with_noise = agent.act(test_state, add_noise=True)
    
    print(f"Test state shape: {test_state.shape}")
    print(f"Action without noise: {action_no_noise}")
    print(f"Action with noise: {action_with_noise}")
    print(f"Action difference: {np.abs(action_no_noise - action_with_noise)}")
    
    print(f"\nSAC Agent test completed successfully!")
    return True

def test_environment_configuration():
    """Test the environment configuration with GBM and 1% fee."""
    
    print("\nTesting Environment Configuration...")
    print("=" * 50)
    
    # Initialize environment
    env = MarketEnvironment(
        randomSeed=42,
        price_model='gbm',
        mu=0.0,
        alpha=2.0,
        leftover_penalty=1e-3
    )
    
    # Check environment parameters
    print(f"Environment parameters:")
    print(f"- Total shares: {env.total_shares:,}")
    print(f"- Starting price: ${env.startingPrice:.2f}")
    print(f"- Liquidation time: {env.liquidation_time} days")
    print(f"- Number of trades: {env.num_n}")
    print(f"- Commission: ${env.epsilon:.2f} per share")
    print(f"- Price impact (eta): {env.eta:.6f}")
    print(f"- Permanent impact (gamma): {env.gamma:.6f}")
    print(f"- Risk aversion (lambda): {env.llambda:.6f}")
    
    # Test price evolution
    print(f"\nTesting GBM price evolution...")
    env.reset(seed=42)
    env.start_transactions()
    
    initial_price = env.prevImpactedPrice
    print(f"Initial price: ${initial_price:.2f}")
    
    # Simulate a few price steps
    for i in range(5):
        # Generate a random action
        action = np.random.uniform(0, 1, 1)
        state, reward, done, info = env.step(action)
        print(f"Step {i+1}: Price=${info.price:.2f}, Exec Price=${info.exec_price:.2f}, "
              f"Shares Sold={info.share_to_sell_now:.0f}")
    
    print(f"Environment configuration test completed!")

if __name__ == "__main__":
    # Test environment configuration
    test_environment_configuration()
    
    # Test SAC agent
    test_sac_agent()
    
    print(f"\nAll tests completed successfully!")
    print(f"The SAC agent is ready for training with the synthetic Chriss-Almgren environment.")
    print(f"Run 'python train_sac_agent.py' to start training.") 