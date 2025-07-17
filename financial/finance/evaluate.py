import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import rewards as rw
from syntheticChrissAlmgren import MarketEnvironment
from td3_agent import TD3
from ddpg_agent import Agent as DDPGAgent
from sac_agent import SACAgent

MODEL_DIR = "trained_models"
TEST_EPISODES = 100

class IdentityAction:
    def compute(self, env, action):
        return float(np.asarray(action).item())

def load_agent(agent_type, path, env):
    if agent_type == 'TD3':
        agent = TD3(env.observation_space_dimension(),
                    env.action_space_dimension(), random_seed=0)
        actor_path = os.path.join(path, "actor.pth")
        critic_path = os.path.join(path, "critic.pth")
        agent.actor_local.load_state_dict(torch.load(actor_path))
        agent.critic_local.load_state_dict(torch.load(critic_path))
    elif agent_type == 'DDPG':
        agent = DDPGAgent(env.observation_space_dimension(),
                          env.action_space_dimension(), random_seed=0)
        torch.load(os.path.join(path, "actor.pth"), map_location='cpu')
        torch.load(os.path.join(path, "critic.pth"), map_location='cpu')
        agent.actor.load_state_dict(torch.load(os.path.join(path, "actor.pth")))
        agent.critic.load_state_dict(torch.load(os.path.join(path, "critic.pth")))
    elif agent_type == 'SAC':
        agent = SACAgent(env.observation_space_dimension(),
                         env.action_space_dimension(), random_seed=0)
        agent.actor.load_state_dict(torch.load(os.path.join(path, "actor.pth")))
        agent.critic1.load_state_dict(torch.load(os.path.join(path, "critic1.pth")))
        agent.critic2.load_state_dict(torch.load(os.path.join(path, "critic2.pth")))
        if os.path.exists(os.path.join(path, "log_alpha.pth")):
            agent.log_alpha = torch.load(os.path.join(path, "log_alpha.pth"))
    else:
        # fallback to pickle
        import pickle
        with open(os.path.join(path, "agent.pkl"), "rb") as f:
            agent = pickle.load(f)
    return agent

def evaluate(agent, env):
    state = env.reset(seed=0)
    env.start_transactions()
    shares_trace = [env.shares_remaining]
    while True:
        a = agent.act(state, add_noise=False)
        state, reward, done, info = env.step(a)
        shares_trace.append(env.shares_remaining)
        if done:
            return np.array(shares_trace), info.implementation_shortfall

def evaluate_all():
    results = {}

    # ——— AC benchmark with U‐based reward ——————————————————————
    ac_reward = rw.ACUtilityReward()
    ac_env = MarketEnvironment(
        reward_function=ac_reward,
        action_type='baseline'      # or whatever, we’ll override
    )
    # force our identity strategy so `step(np.array([s]))` sells exactly s
    ac_env.action_strategy = IdentityAction()
    # bind env into the reward so reset() and calculate() can call compute_AC_utility
    ac_reward.env = ac_env

    # now reset WILL call ac_reward.reset(...) internally
    trade_schedule = ac_env.get_trade_list()

    def simulate_ac(trade_list):
        # fresh copy per sim
        env = MarketEnvironment(reward_function=rw.ACUtilityReward())
        env.action_strategy = IdentityAction()
        env.reward_function.env = env
        env.start_transactions()
        for s in trade_list:
            _, _, done, info = env.step(np.array([s]))
            if done:
                break
        return info.implementation_shortfall

    # build the “trajectory” from the schedule once
    results['Almgren–Chriss (U‐reward)'] = {
        'trajectory': np.concatenate(([ac_env.total_shares],
                                      ac_env.total_shares - np.cumsum(trade_schedule))),
        'shortfalls': [simulate_ac(trade_schedule) for _ in range(TEST_EPISODES)]
    }

    # evaluate each RL agent
    for agent_type in ['SAC']:
        for folder in os.listdir(MODEL_DIR):
            if not folder.startswith(agent_type): continue
            path = os.path.join(MODEL_DIR, folder)
            # parse action_type and reward_name if you need
            env = MarketEnvironment()
            agent = load_agent(agent_type, path, env)

            trajs, sfs = [], []
            for _ in range(TEST_EPISODES):
                t, sf = evaluate(agent, env)
                trajs.append(t)
                sfs.append(sf)
            # average trajectory
            mean_traj = np.mean(np.stack(trajs), axis=0)
            results[folder] = {
                'trajectory': mean_traj,
                'shortfalls': sfs
            }
    return results

def plot_results(results):
    # Plot trajectories
    plt.figure(figsize=(8,5))
    for name, data in results.items():
        plt.plot(data['trajectory'], label=name)
    plt.xlabel("Time step")
    plt.ylabel("Shares remaining")
    plt.title("Execution trajectories")
    plt.legend()
    plt.tight_layout()
    plt.savefig("compare_trajectories.png")

    plt.figure(figsize=(8,5))
    names = list(results.keys())
    sfs = [results[n]['shortfalls'] for n in names]
    plt.boxplot(sfs, labels=names, showfliers=False)
    plt.ylabel("Implementation Shortfall")
    plt.title("Shortfall comparison")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("compare_shortfalls.png")

if __name__=='__main__':
    res = evaluate_all()
    plot_results(res)
    print("Saved plots: compare_trajectories.png, compare_shortfalls.png")