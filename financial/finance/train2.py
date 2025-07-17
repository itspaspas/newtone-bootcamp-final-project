import os
import numpy as np
from collections import deque
import importlib
import torch

import syntheticChrissAlmgren as sca
import price_models
import utils

from td3_agent import TD3
from ddpg_agent import Agent as DDPGAgent
from sac_agent import SACAgent
import rewards as rw
from actions import action_registry

MODEL_DIR = "trained_models"
os.makedirs(MODEL_DIR, exist_ok=True)

def reload_all():
    importlib.reload(price_models)
    importlib.reload(utils)
    importlib.reload(sca)
    importlib.reload(rw)

def run_training(agent_type, action_type, reward_name, reward_fn,
                 episodes=5000, lqt=60, n_trades=60, risk_aversion=1e-6):
    reload_all()
    env = sca.MarketEnvironment(
        reward_function=reward_fn,
        action_type=action_type
    )

    # Instantiate agent
    if agent_type == 'TD3':
        agent = TD3(env.observation_space_dimension(),
                    env.action_space_dimension(), random_seed=0)
    elif agent_type == 'DDPG':
        agent = DDPGAgent(env.observation_space_dimension(),
                          env.action_space_dimension(), random_seed=0)
    elif agent_type == 'SAC':
        agent = SACAgent(env.observation_space_dimension(),
                         env.action_space_dimension(), random_seed=0)
    else:
        raise ValueError(f"Unknown agent type {agent_type}")

    shortfall_deque = deque(maxlen=100)
    all_step_rewards = []
    from utils import ActionTracker
    action_tracker = ActionTracker()

    for ep in range(episodes):
        state = env.reset(seed=ep, liquid_time=lqt,
                          num_trades=n_trades, lamb=risk_aversion)
        env.start_transactions()

        for t in range(n_trades + 1):
            a = agent.act(state, add_noise=True)
            next_state, reward, done, info = env.step(a)

            all_step_rewards.append(float(reward))
            action_tracker.add(float(a))

            agent.step(state, a, reward, next_state, done)
            state = next_state

            if done:
                shortfall_deque.append(info.implementation_shortfall)
                break

        if (ep + 1) % 100 == 0:
            avg_sf = np.mean(shortfall_deque)
            print(f"{agent_type} | {action_type} | {reward_name} "
                  f"Ep {ep+1}/{episodes} avg shortf: ${avg_sf:,.2f}")

    return agent, all_step_rewards, action_tracker.get_all()

def save_agent(agent, agent_type, action_type, reward_name):
    tag = f"{agent_type}__{action_type}__{reward_name}"
    path = os.path.join(MODEL_DIR, tag)
    os.makedirs(path, exist_ok=True)

    # (same saving logic as before...)
    if hasattr(agent, 'actor_local') and hasattr(agent, 'critic_local'):
        torch.save(agent.actor_local.state_dict(), os.path.join(path, "actor.pth"))
        torch.save(agent.critic_local.state_dict(), os.path.join(path, "critic.pth"))
    elif hasattr(agent, 'actor') and hasattr(agent, 'critic1') and hasattr(agent, 'critic2'):
        torch.save(agent.actor.state_dict(), os.path.join(path, "actor.pth"))
        torch.save(agent.critic1.state_dict(), os.path.join(path, "critic1.pth"))
        torch.save(agent.critic2.state_dict(), os.path.join(path, "critic2.pth"))
        if hasattr(agent, 'log_alpha'):
            torch.save(agent.log_alpha, os.path.join(path, "log_alpha.pth"))
    else:
        import pickle
        with open(os.path.join(path, "agent.pkl"), "wb") as f:
            pickle.dump(agent, f)

    return path

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Build reward‑function dictionary
    # ------------------------------------------------------------------
    TOTAL_SHARES = sca.TOTAL_SHARES
    STARTING_PRICE = sca.STARTING_PRICE
    ETA = sca.ETA
    TAU = sca.LIQUIDATION_TIME / sca.NUM_N
    market_env = sca.MarketEnvironment()
    LEFTOVER = market_env.leftover_penalty
    ALPHA = market_env.alpha

    reward_functions = {
        "PnL": rw.PnL(),
        "CjOeCriterion": rw.CjOeCriterion(0.01, 0.0, 2.0, 1.0),
        "CjMmCriterion": rw.CjMmCriterion(0.01, 0.0, 2.0, 1.0),
        "RunningInventoryPenalty": rw.CjCriterion(0.01, 0.0, 2.0),
        "ExponentialUtility": rw.ExponentialUtility(0.1),
        "NormalizedExec": rw.NormalizedExecutionReward(TOTAL_SHARES, STARTING_PRICE),
        "ShortfallPenalties": rw.ExecutionShortfallWithPenaltiesReward(
            STARTING_PRICE, ALPHA, ETA, TAU, LEFTOVER, TOTAL_SHARES
        ),
    }

    agent_name = "TD3"
    actions = [a for a in action_registry.keys() if a != "baseline"]
    EXEC_REWARDS = {"NormalizedExec", "ShortfallPenalties"}
    eligible_rewards = {
        name: rf
        for name, rf in reward_functions.items()
        if name not in EXEC_REWARDS
    }

    for action_name in actions:
        for reward_name, rf in eligible_rewards.items():
            # Train the TD3 agent
            agent, rewards_seq, actions_seq = run_training(
                agent_name, action_name, reward_name, rf
            )

            # Save model files and get folder
            folder = save_agent(agent, agent_name, action_name, reward_name)

            # Save the cumulative‑reward plot
            cum_rew = np.cumsum(np.asarray(rewards_seq))
            utils.plot_rewards(
                cum_rew,
                title=f"Cumulative Reward ({agent_name}-{action_name}-{reward_name})",
                save_path=os.path.join(folder, "cumulative_reward.png"),
            )

            utils.plot_rewards(
                rewards_seq,
                title=f"Step Rewards ({agent_name}-{action_name}-{reward_name})",
                save_path=os.path.join(folder, "step_rewards.png"),
            )

            # Save the action‑trajectory plot
            utils.plot_rewards(
                actions_seq,
                title=f"Actions over Steps ({agent_name}-{action_name}-{reward_name})",
                save_path=os.path.join(folder, "actions.png"),
            )

            print(f" Plots saved → {folder}\n")
