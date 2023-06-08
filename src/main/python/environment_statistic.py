import time

import numpy


class EnvironmentStatistic:
    def reset(self):
        self.episode_data = {'episodes': [], 'total': {'cumulative_rewards': 0, 'episode_lengths': []}}
        self.episode_reward = 0
        self.episode_length = 0
        self.time = time.time()


    def reset_episode(self):
        self.episode_reward = 0
        self.episode_length = 0

    def add_episode(self):
        self.episode_data['episodes'].append(
            {'cumulative_reward': self.episode_reward, 'episode_length': self.episode_length})
        self.episode_data['total']['cumulative_rewards'] += self.episode_reward
        self.episode_data['total']['episode_lengths'].append(self.episode_length)
        self.episode_data['total']['duration'] = time.time() - self.time

    def calculate_statistics(self):
        num_episodes = len(self.episode_data['episodes'])
        cumulative_rewards = [episode_data['cumulative_reward'] for episode_data in self.episode_data['episodes']]
        episode_lengths = [episode_data['episode_length'] for episode_data in self.episode_data['episodes']]

        mean_reward = numpy.mean(cumulative_rewards)
        reward_std = numpy.std(cumulative_rewards)
        mean_length = numpy.mean(episode_lengths)
        length_std = numpy.std(episode_lengths)
        max_reward = numpy.max(cumulative_rewards)
        min_reward = numpy.min(cumulative_rewards)
        num_success_episodes = len([reward for reward in cumulative_rewards if reward > 0])
        success_rate = (num_success_episodes / num_episodes) * 100

        success_rewards = [reward for reward in cumulative_rewards if reward > 0]
        failed_rewards = [reward for reward in cumulative_rewards if reward <= 0]
        mean_success_reward = numpy.mean(success_rewards) if success_rewards != [] else 0
        mean_failed_reward = numpy.mean(failed_rewards) if failed_rewards != [] else 0
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
            'success_rate': success_rate,
            'mean_success_reward': mean_success_reward,
            'mean_failed_reward': mean_failed_reward,
            'time': time
        }

        return statistics
