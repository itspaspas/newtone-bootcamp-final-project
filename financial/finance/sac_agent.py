import numpy as np
import random
from collections import deque, namedtuple
from model import Actor, Critic
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)
BATCH_SIZE  = 256
GAMMA       = 0.99
TAU         = 5e-3
LR_ACTOR    = 3e-4
LR_CRITIC   = 3e-4
LR_ALPHA    = 3e-4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        to_tensor = lambda x: torch.from_numpy(np.vstack(x)).float().to(device)
        states      = to_tensor([e.state  for e in experiences])
        actions     = to_tensor([e.action for e in experiences])
        rewards     = to_tensor([e.reward for e in experiences])
        next_states = to_tensor([e.next_state for e in experiences])
        dones       = to_tensor([e.done for e in experiences]).to(device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class GaussianPolicy(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc_mean = nn.Linear(fc2_units, action_size)
        self.fc_log_std = nn.Linear(fc2_units, action_size)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc_mean.weight.data.uniform_(-3e-3, 3e-3)
        self.fc_log_std.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()          # reparameterisation trick
        y_t = torch.tanh(x_t)
        action_env = (y_t + 1.0) / 2.0  # scale tanh [-1,1] → [0,1]

        # Log prob in policy space with tanh correction
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action_env, log_prob, torch.tanh(mean)

class SACAgent:
    def __init__(self, state_size, action_size, random_seed, automatic_entropy_tuning=True):
        self.state_size = state_size
        self.action_size = action_size
        random.seed(random_seed)

        # Actor
        self.actor = GaussianPolicy(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        # Critics (twin)
        self.critic1 = SACCritic(state_size, action_size, random_seed).to(device)
        self.critic2 = SACCritic(state_size, action_size, random_seed).to(device)
        self.critic1_target = SACCritic(state_size, action_size, random_seed).to(device)
        self.critic2_target = SACCritic(state_size, action_size, random_seed).to(device)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=LR_CRITIC)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=LR_CRITIC)

        # Sync targets
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Entropy temperature
        self.automatic_entropy_tuning = automatic_entropy_tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -float(action_size)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ALPHA)
        else:
            self.alpha = 0.2

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        self.learn_step = 0

    def act(self, state, add_noise=True):
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor.eval()
        with torch.no_grad():
            if add_noise:
                action_env, _, _ = self.actor.sample(state_t)
            else:
                mean, _ = self.actor.forward(state_t)
                action_env = torch.tanh(mean)
                action_env = (action_env + 1.0) / 2.0
        self.actor.train()
        return action_env.cpu().numpy().squeeze(0)

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) >= BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Update critics
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor.sample(next_states)
            target_q1_next = self.critic1_target(next_states, next_actions)
            target_q2_next = self.critic2_target(next_states, next_actions)
            target_q_next  = torch.min(target_q1_next, target_q2_next) - self.alpha() * next_log_pi
            q_targets = rewards + (gamma * (1 - dones) * target_q_next)

        # Critic 1 loss
        q1_expected = self.critic1(states, actions)
        critic1_loss = F.mse_loss(q1_expected, q_targets)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        # Critic 2 loss
        q2_expected = self.critic2(states, actions)
        critic2_loss = F.mse_loss(q2_expected, q_targets)
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Update actor
        new_actions, log_pi, _ = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new  = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha() * log_pi - q_new).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Temperature (alpha) update
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # Soft update targets
        self.soft_update(self.critic1, self.critic1_target, TAU)
        self.soft_update(self.critic2, self.critic2_target, TAU)

    def alpha(self):
        if self.automatic_entropy_tuning:
            return self.log_alpha.exp()
        return torch.tensor(self.alpha).to(device)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
