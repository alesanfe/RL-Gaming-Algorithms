import time
from dataclasses import dataclass

import gym
import numpy
import matplotlib.pyplot as plt


@dataclass
class EnvironmentStatistic:
    env: gym.Env

    def reset(self):
        self.episode_data = {'episodes': [], 'total': {'cumulative_rewards': 0, 'episode_lengths': [], 'duration': 0,
                                                       'success_episodes': 0, 'failed_episodes': 0}}
        self.episode_reward = 0
        self.episode_length = 0
        self.time = time.time()


    def reset_episode(self):
        self.episode_reward = 0
        self.episode_length = 0
        self.time = time.time()

    def continue_episode(self, reward):
        self.episode_reward += reward
        self.episode_length += 1


    def add_episode(self, next_state):
        self.episode_data['episodes'].append(
            {'cumulative_reward': self.episode_reward, 'episode_length': self.episode_length, "time": time.time() - self.time})
        self.episode_data['total']['cumulative_rewards'] += self.episode_reward
        self.episode_data['total']['episode_lengths'].append(self.episode_length)
        self.episode_data['total']['duration'] += time.time() - self.time
        if next_state in self.get_terminal_states():
            self.episode_data['total']['success_episodes'] += 1
        else:
            self.episode_data['total']['failed_episodes'] += 1

    def get_terminal_states(self):

        posible_states = range(self.env.observation_space.n)

        # Inicializar una lista para almacenar los estados terminales
        terminal_states = set()

        # Comprobar cada estado si es terminal o no
        if hasattr(self.env, 'target_location'):
            return [self.env.target_location]

        for state in posible_states:
            for action in range(self.env.action_space.n):
                transactions = self.env.P[state][action]
                for transaction in transactions:
                    _, next_state, reward, _ = transaction
                    if reward > 0:
                        terminal_states.add(next_state)
                        break

        return terminal_states

    def calculate_statistics(self):
        num_episodes = len(self.episode_data['episodes'])
        cumulative_rewards = [episode_data['cumulative_reward'] for episode_data in self.episode_data['episodes']]
        episode_lengths = [episode_data['episode_length'] for episode_data in self.episode_data['episodes']]
        episode_time = [episode_data['time'] for episode_data in self.episode_data['episodes']]

        mean_reward = numpy.mean(cumulative_rewards)
        mean_time = numpy.mean(episode_time)
        reward_std = numpy.std(cumulative_rewards)
        mean_length = numpy.mean(episode_lengths)
        length_std = numpy.std(episode_lengths)
        max_reward = numpy.max(cumulative_rewards)
        min_reward = numpy.min(cumulative_rewards)
        num_success_episodes = self.episode_data['total']['success_episodes']
        success_rate = num_success_episodes / num_episodes
        failed_rate = 1 - success_rate
        time = self.episode_data['total']['duration']

        statistics = {
            'mean_reward': mean_reward,
            'reward_std': reward_std,
            'mean_length': mean_length,
            'length_std': length_std,
            'num_episodes': num_episodes,
            'max_reward': max_reward,
            'min_reward': min_reward,
            'num_success_episodes': num_success_episodes,
            'success_rate': success_rate*100,
            'failed_rate': failed_rate*100,
            'time': time,
            'mean_time': mean_time
        }

        return statistics

    def _plot_graph(self, data, ylabel, title):
        plt.plot(data)
        plt.ylabel(ylabel)
        plt.xlabel('Episode')
        plt.title(title)
        plt.show()

    def get_graph_reward(self, title):
        data = [episode['cumulative_reward'] for episode in self.episode_data['episodes']]
        self._plot_graph(data, 'Reward', title)

    def get_graph_length(self, title):
        data = [episode['episode_length'] for episode in self.episode_data['episodes']]
        self._plot_graph(data, 'Length', title)

    def get_graph_time(self, title):
        data = [episode['time'] for episode in self.episode_data['episodes']]
        self._plot_graph(data, 'Duration', title)

    def get_graph_statistics(self, title):
        statistics = self.calculate_statistics()

        # Plotting statistics
        labels = ['mean_reward', 'reward_std', 'mean_length', 'length_std', 'max_reward', 'min_reward']
        values = [statistics[label] for label in labels]
        cumulative_rewards = [episode['cumulative_reward'] for episode in self.episode_data['episodes']]

        plt.bar(labels, values)
        plt.ylabel('Value')
        plt.xlabel('Statistic')
        plt.show()

        # Plotting success and failure rewards
        success_rewards = [reward for reward in cumulative_rewards if reward > 0]
        failed_rewards = [reward for reward in cumulative_rewards if reward <= 0]

        plt.hist(success_rewards, bins=10, alpha=0.5, label='Success Rewards')
        plt.hist(failed_rewards, bins=10, alpha=0.5, label='Failed Rewards')
        plt.legend(loc='upper right')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.show()
