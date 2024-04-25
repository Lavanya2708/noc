import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from td3 import TD3
from environment import NOCEnv

def plot_metrics(metrics):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(metrics['latency'])
    axs[0, 0].set_title('Latency over Episodes')
    axs[0, 1].plot(metrics['bandwidth'])
    axs[0, 1].set_title('Bandwidth over Episodes')
    axs[1, 0].plot(metrics['buffer_occupancy'])
    axs[1, 0].set_title('Buffer Occupancy over Episodes')
    axs[1, 1].plot(metrics['throttling'])
    axs[1, 1].set_title('Throttling over Episodes')
    plt.show()

def save_model(agent, filename_prefix):
    torch.save(agent.actor.state_dict(), f'{filename_prefix}_actor.pth')
    torch.save(agent.critic1.state_dict(), f'{filename_prefix}_critic1.pth')
    torch.save(agent.critic2.state_dict(), f'{filename_prefix}_critic2.pth')

def load_model(agent, filename_prefix):
    agent.actor.load_state_dict(torch.load(f'{filename_prefix}_actor.pth'))
    agent.critic1.load_state_dict(torch.load(f'{filename_prefix}_critic1.pth'))
    agent.critic2.load_state_dict(torch.load(f'{filename_prefix}_critic2.pth'))
    agent.actor_target.load_state_dict(agent.actor.state_dict())
    agent.critic1_target.load_state_dict(agent.critic1.state_dict())
    agent.critic2_target.load_state_dict(agent.critic2.state_dict())

def main(args):
    env = NOCEnv()
    state_dim = len(env._get_state())
    action_dim = 4  # adjust CPU, IO buffers, CPU weight, and frequency
    agent = TD3(state_dim, action_dim)

    episodes = args.episodes
    episode_length = args.episode_length
    episode_rewards = []
    performance_logs = {'latency': [], 'bandwidth': [], 'buffer_occupancy': [], 'throttling': []}

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        for step in range(episode_length):
            action = agent.select_action(state)
            next_state, reward, done = env.step({
                'adjust_cpu_buffer': action[0],
                'adjust_io_buffer': action[1],
                'adjust_cpu_weight': action[2],
                'adjust_frequency': action[3]
            })
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.train(agent.replay_buffer, env)
            state = next_state
            total_reward += reward

            # Print detailed state information at each step
            print(f"Step {step + 1}:")
            print(f"  Latency: {env.latency}, Bandwidth: {env.bandwidth}, Buffer Occupancy: {env.buffer_occupancy}, Throttling: {env.throttling}")
            print(f"  Reward: {reward}\n")

            if done:
                break

        episode_rewards.append(total_reward)
        performance_logs['latency'].append(env.latency)
        performance_logs['bandwidth'].append(env.bandwidth)
        performance_logs['buffer_occupancy'].append(env.buffer_occupancy)
        performance_logs['throttling'].append(env.throttling)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}\n")

    plot_metrics(performance_logs)
    save_model(agent, 'td3_agent_model')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a TD3 agent for NOC optimization")
    parser.add_argument('--episodes', type=int, default=200, help='Number of episodes to train')
    parser.add_argument('--episode_length', type=int, default=500, help='Number of steps per episode')
    args = parser.parse_args()
    main(args)
