import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # Ensures output is between -1 and 1, scale in select_action
        )

    def forward(self, state):
        return self.network(state)  # Output raw actions

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Output Q-value estimation
        )

    def forward(self, state, action):
        return self.network(torch.cat([state, action], dim=-1))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward, dtype=np.float32), np.array(next_state), np.array(done, dtype=np.uint8)

    def __len__(self):
        return len(self.buffer)

class TD3:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)

        self.critic1 = Critic(state_dim, action_dim)
        self.critic1_target = Critic(state_dim, action_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=0.001)

        self.critic2 = Critic(state_dim, action_dim)
        self.critic2_target = Critic(state_dim, action_dim)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=0.001)

        self.replay_buffer = ReplayBuffer(1000000)  # Ensure this is initialized


        self.batch_size = 100
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_delay = 2
        self.total_it = 0

        self.action_scales = torch.FloatTensor([5, 5, 0.05, 0.1])  # Scale for each action
        self.action_offsets = torch.FloatTensor([0, 0, 0, 0])  # Offset for each action (add if needed)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        raw_actions = self.actor(state).detach()  # Raw actions are in range [-1, 1]
        scaled_actions = raw_actions * self.action_scales + self.action_offsets  # Scale and shift actions
        return scaled_actions.cpu().numpy().flatten()

    def train(self, replay_buffer, environment):
        if len(replay_buffer) < self.batch_size:
            return

        self.total_it += 1

        state, action, reward, next_state, done = replay_buffer.sample(self.batch_size)
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)

        noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)

        target_Q1 = self.critic1_target(next_state, next_action)
        target_Q2 = self.critic2_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + ((1 - done) * self.gamma * target_Q).detach()

        current_Q1 = self.critic1(state, action)
        current_Q2 = self.critic2(state, action)

        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update the target networks
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Correctly unpacking the three outputs from step function
        current_action = self.select_action(state[0].numpy())
        action_to_apply = {
            'adjust_cpu_buffer': current_action[0],
            'adjust_io_buffer': current_action[1],
            'adjust_cpu_weight': current_action[2],
            'adjust_frequency': current_action[3]
        }
        next_state_env, reward_env, done_env = environment.step(action_to_apply)