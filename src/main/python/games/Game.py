from dataclasses import dataclass

import time

from gym import Env

from src.main.python.DoubleQLearning import DoubleQLearning
from src.main.python.Sarsa import Sarsa
from src.main.python.aprendizaje_por_refuerzo import Montecarlo_IE, PolíticaEpsilonVoraz, Q_Learning


@dataclass
class Game:
    environment: Env
    discount_factor: float
    learning_factor: float
    iterations: int
    time: float = 0

    def resolve_by_montecarlo(self):
        """Resolución del entorno utilizando Montecarlo con inicios exploratorios."""
        self.time = time.time()
        agent = Montecarlo_IE(self.environment, self.discount_factor)
        agent.entrena(self.iterations)
        self.time = time.time() - self.time
        return agent

    def resolve_by_q_learning(self, epsilon):
        """Resolución del entorno utilizando Q-Learning"""
        self.time = time.time()
        export_policy = PolíticaEpsilonVoraz(epsilon)
        agent = Q_Learning(self.environment, self.discount_factor, self.learning_factor, export_policy)
        agent.entrena(self.iterations)
        self.time = time.time() - self.time
        return agent

    def resolve_by_sarsa(self, epsilon, alpha, gamma):
        """Resolución del entorno utilizando Sarsa."""
        self.time = time.time()
        export_policy = PolíticaEpsilonVoraz(epsilon)
        agent = Sarsa(self.environment, alpha, gamma, export_policy)
        agent.train(self.iterations)
        self.time = time.time()
        return agent

    def resolve_by_double_q_learning(self, epsilon, alpha, gamma):
        """Resolución del entorno utilizando Double Q-Learning."""
        self.time = time.time()
        export_policy = PolíticaEpsilonVoraz(epsilon)
        agent = DoubleQLearning(self.environment, alpha, gamma, export_policy)
        agent.train(self.iterations)
        self.time = time.time() - self.time
        return agent

    def print_stats(self):
        """Imprime las estadísticas del entorno."""
        print("Tiempo de ejecución: " + str(self.time) + " segundos")
        stats = self.environment.get_stats()
        print("Recompensa media: " + str(stats['mean_reward']) + " +/- " + str(stats['reward_std']))
        print("Longitud media de episodios: " + str(stats['mean_length']) + " +/- " + str(stats['length_std']))
        print("Número de episodios: " + str(stats['num_episodes']))
        print("Máxima recompensa alcanzada: " + str(stats['max_reward']))
        print("Mínima recompensa alcanzada: " + str(stats['min_reward']))
        print("Recompensa en el último episodio: " + str(stats['last_episode_reward']))
        print("Longitud del último episodio: " + str(stats['last_episode_length']))
        print("Episodios completados exitosamente: " + str(stats['num_success_episodes']))
        print("Episodios terminados por tiempo límite: " + str(stats['num_time_limit_episodes']))
        print("Episodios terminados por límite de pasos: " + str(stats['num_step_limit_episodes']))
        print("Episodios terminados por límite de tiempo y pasos: " + str(stats['num_time_step_limit_episodes']))
        print("Porcentaje de éxito: " + str(stats['success_rate']) + "%")
        print("Promedio de recompensa de los episodios exitosos: " + str(stats['mean_success_reward']))
        print("Promedio de recompensa de los episodios fallidos: " + str(stats['mean_failed_reward']))