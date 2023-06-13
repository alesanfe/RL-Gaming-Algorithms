from dataclasses import dataclass
import gym

from src.main.python.games.game import Game
from src.main.python.games.golf.golf_env import GolfEnv


@dataclass
class GameComparator:
    def __init__(self, game: Game):
        self.game = game

    def compare_different_algorithms(self, algorithms=['Montecarlo', 'Q-Learning', 'Sarsa'], epsilon=0.1, alpha=0.1,
                                     gamma=0.9):
        results = {}
        for alg in algorithms:
            self._execute_algorithm(alg, alpha, alg, epsilon, gamma, results)

        data = self._get_data_from_stats(results)
        self._print_data("algoritmos", data)

    def compare_different_cases(self, algorithm='Montecarlo', epsilon=[0.1, 0.2, 0.3, 0.4, 0.5],
                                alpha=[0.1, 0.2, 0.3, 0.4, 0.5], gamma=[0.9, 0.8, 0.7, 0.6, 0.5]):
        results = {}
        for eps in epsilon:
            for alp in alpha:
                for gam in gamma:
                    key = "Epsilon: " + str(eps) + " - Alpha: " + str(alp) + " - Gamma: " + str(gam)
                    self._execute_algorithm(algorithm, alp, key, eps, gam, results)

        data = self._get_data_from_stats(results)
        self._print_data("casos", data)

    def compare_different_environments(self, environments=['FrozenLake-v1', 'Taxi-v3', "Golf-v0"], algorithm='Montecarlo',
                                      epsilon=0.1, alpha=0.1, gamma=0.9):
        results = {}
        for env in environments:
            if env == 'Golf-v0':
                environment = GolfEnv()
            else:
                environment = gym.make(env)
            self.game.environment = environment
            self._execute_algorithm(algorithm, alpha, env, epsilon, gamma, results)

        data = self._get_data_from_stats(results)
        self._print_data("entornos", data)

    def get_graphs(self, title):
        statistics = self.game.agent.statistics
        statistics.get_graph_reward(f"{title} - Reward")
        statistics.get_graph_length(f"{title} - Length")
        statistics.get_graph_time(f"{title} - Time")
        statistics.get_graph_statistics(f"{title} - Statistics")


    def _execute_algorithm(self, algorithm, alpha, env, epsilon, gamma, results):
        if algorithm == 'Montecarlo':
            agent_montecarlo = self.game.resolve_by_montecarlo()
            results[env] = agent_montecarlo.calculate_statistics()
        elif algorithm == 'Q-Learning':
            agent_q_learning = self.game.resolve_by_q_learning(epsilon)
            results[env] = agent_q_learning.calculate_statistics()
        elif algorithm == 'Sarsa':
            agent_sarsa = self.game.resolve_by_sarsa(epsilon, alpha, gamma)
            results[env] = agent_sarsa.calculate_statistics()
        else:
            raise ValueError("Algoritmo no encontrado")

    def _is_some_data(self, key, results):
        return len(set([results[x][key] for x in results])) != 1

    def _get_greater(self, key, results):
        return max(results, key=lambda x: results[x][key]) if self._is_some_data(key, results) else None

    def _get_lower(self, key, results):
        return min(results, key=lambda x: results[x][key]) if self._is_some_data(key, results) else None

    def _into_data(self, key, results, data):
        return data + " (" + str(results[data][key]) + ")" if data is not None else "Todos iguales"

    def _get_data_from_stats(self, results):
        """Obtiene los datos de las estadísticas de un entorno."""
        metrics = ['mean_reward',
            'reward_std',
            'mean_length',
            'length_std',
            'num_episodes',
            'max_reward',
            'min_reward',
            'num_success_episodes',
            'success_rate',
            'failed_rate',
            'time',
            'mean_time']
        data = {}

        for metric in metrics:
            greater = self._get_greater(metric, results)
            lower = self._get_lower(metric, results)
            data["greater_" + metric] = self._into_data(metric, results,greater)
            data["lower_" + metric] = self._into_data(metric, results, lower)

        return data

    def _print_data(self, topic, data):
        without_last_letter = topic[:-1]
        first_upper = without_last_letter[0].upper() + without_last_letter[1:]
        print(f"\nComparación de {topic}:\n")
        print(f"Mejor {without_last_letter} por recompensa media: " + data['greater_mean_reward'])
        print(f"{first_upper} con menor longitud media de episodios: " + data['lower_mean_length'])
        print(f"{first_upper} con mayor porcentaje de recompensas exitosas: " + data['greater_success_rate'])
        print(f"{first_upper} con mayor porcentaje de recompensas fallidas: " + data['greater_failed_rate'])
        print(f"Mejor {without_last_letter} por tiempo de ejecución: " + data['lower_time'])
        print(f"Mejor {without_last_letter} por tiempo de ejecución por episodio: " + data['lower_mean_time'])
        print(f"Mejor {without_last_letter} por máxima recompensa alcanzada: " + data['greater_max_reward'])
        print(f"Mejor {without_last_letter} por mínima recompensa alcanzada: " + data['lower_min_reward'])
