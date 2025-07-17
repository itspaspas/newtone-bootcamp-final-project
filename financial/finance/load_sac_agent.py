import torch
from sac_agent import SACAgent

def load_sac_agent(weights_path='trained_sac_comprehensive_weights.pth'):
    """
    Load a trained SAC agent from saved weights.
    
    Params
    ======
        weights_path (str): path to the saved weights file
    
    Returns
    =======
        agent: loaded SAC agent
    """
    
    # Load the saved weights
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    # Extract parameters
    state_size = checkpoint['state_size']
    action_size = checkpoint['action_size']
    seed = checkpoint['seed']
    automatic_entropy_tuning = checkpoint['automatic_entropy_tuning']
    
    # Create new agent
    agent = SACAgent(state_size, action_size, seed, automatic_entropy_tuning)
    
    # Load the weights
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic1.load_state_dict(checkpoint['critic1_state_dict'])
    agent.critic2.load_state_dict(checkpoint['critic2_state_dict'])
    agent.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
    agent.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
    
    if automatic_entropy_tuning and checkpoint['log_alpha'] is not None:
        agent.log_alpha.data = checkpoint['log_alpha'].clone()
    
    print(f"SAC agent loaded successfully from {weights_path}")
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    print(f"Automatic entropy tuning: {automatic_entropy_tuning}")
    
    return agent

if __name__ == "__main__":
    # Example usage
    try:
        agent = load_sac_agent()
        print("Agent loaded successfully!")
    except Exception as e:
        print(f"Failed to load agent: {e}")
        print("Make sure the weights file exists and is not corrupted.") 