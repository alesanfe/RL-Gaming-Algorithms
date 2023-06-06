from dataclasses import dataclass

import time

import gym
from gym import Env

from src.main.python.sarsa import Sarsa
from src.main.python.aprendizaje_por_refuerzo import Montecarlo_IE, PolíticaEpsilonVoraz, Q_Learning


@dataclass
class Game:
    environment: Env
    discount_factor: float
    learning_factor: float
    iterations: int
    time: float = 0
    agent = None

    def resolve_by_montecarlo(self):
        """Resolución del entorno utilizando Montecarlo con inicios exploratorios."""
        self.time = time.time()
        agent = Montecarlo_IE(self.environment, self.discount_factor)
        agent.entrena(self.iterations)
        self.time = time.time() - self.time
        self.agent = agent
        return agent

    def resolve_by_q_learning(self, epsilon):
        """Resolución del entorno utilizando Q-Learning"""
        self.time = time.time()
        export_policy = PolíticaEpsilonVoraz(epsilon)
        agent = Q_Learning(self.environment, self.discount_factor, self.learning_factor, export_policy)
        agent.entrena(self.iterations)
        self.time = time.time() - self.time
        self.agent = agent
        return agent

    def resolve_by_sarsa(self, epsilon, alpha, gamma):
        """Resolución del entorno utilizando Sarsa."""
        self.time = time.time()
        export_policy = PolíticaEpsilonVoraz(epsilon)
        agent = Sarsa(self.environment, alpha, gamma, export_policy)
        agent.train(self.iterations)
        self.time = time.time()
        self.agent = agent
        return agent

    def print_stats(self):
        """Imprime las estadísticas del entorno."""
        print("Tiempo de ejecución: " + str(self.time) + " segundos")
        stats = self.agent.calculate_statistics()
        print("Recompensa media: " + str(stats['mean_reward']) + " +/- " + str(stats['reward_std']))
        print("Longitud media de episodios: " + str(stats['mean_length']) + " +/- " + str(stats['length_std']))
        print("Número de episodios: " + str(stats['num_episodes']))
        print("Máxima recompensa alcanzada: " + str(stats['max_reward']))
        print("Mínima recompensa alcanzada: " + str(stats['min_reward']))
        print("Episodios completados exitosamente: " + str(stats['num_success_episodes']))
        print("Porcentaje de éxito: " + str(stats['success_rate']) + "%")
        print("Promedio de recompensa de los episodios exitosos: " + str(stats['mean_success_reward']))
        print("Promedio de recompensa de los episodios fallidos: " + str(stats['mean_failed_reward']))
        print("Tiempo de ejecución: " + str(stats['time']) + " segundos")

    def compare_different_algorithms(self, algorithms=['Montecarlo', 'Q-Learning', 'Sarsa'], epsilon=0.1, alpha=0.1, gamma=0.9):
        results = {}
        for alg in algorithms:
            if alg == 'Montecarlo':
                agent_montecarlo = self.resolve_by_montecarlo()
                results[alg] = agent_montecarlo.calculate_statistics()
            elif alg == 'Q-Learning':
                agent_q_learning = self.resolve_by_q_learning(epsilon)
                results[alg] = agent_q_learning.calculate_statistics()
            elif alg == 'Sarsa':
                agent_sarsa = self.resolve_by_sarsa(epsilon, alpha, gamma)
                results[alg] = agent_sarsa.calculate_statistics()
            else:
                raise ValueError("Algoritmo no encontrado")
            print(f"\nAlgoritmo {alg}:")
            self.print_stats()
        # A partir de results, obtener el mejor en cada uno
        print("\nComparación de algoritmos:\n")
        print("Mejor algoritmo por recompensa media: " + max(results, key=lambda x: results[x]['mean_reward']))
        print("Mejor algoritmo por longitud media de episodios: " + min(results, key=lambda x: results[x]['mean_length']))
        print("Mejor algoritmo por porcentaje de éxito: " + max(results, key=lambda x: results[x]['success_rate']))
        print("Mejor algoritmo por promedio de recompensa de los episodios exitosos: " + max(results, key=lambda x: results[x]['mean_success_reward']))
        print("Mejor algoritmo por promedio de recompensa de los episodios fallidos: " + min(results, key=lambda x: results[x]['mean_failed_reward']))
        print("Mejor algoritmo por tiempo de ejecución: " + min(results, key=lambda x: results[x]['time']))
        print("Mejor algoritmo por número de episodios: " + min(results, key=lambda x: results[x]['num_episodes']))
        print("Mejor algoritmo por máxima recompensa alcanzada: " + max(results, key=lambda x: results[x]['max_reward']))
        print("Mejor algoritmo por mínima recompensa alcanzada: " + max(results, key=lambda x: results[x]['min_reward']))
        print("Menor tiempo de ejecución: " + min(results, key=lambda x: results[x]['time']))

    def compare_different_cases(self, algorithm='Montecarlo', epsilon=[0.1, 0.2, 0.3, 0.4, 0.5], alpha=[0.1, 0.2, 0.3, 0.4, 0.5], gamma=[0.9, 0.8, 0.7, 0.6, 0.5]):
        results = {}
        for eps in epsilon:
            for alp in alpha:
                for gam in gamma:
                    print("\n" + algorithm + "\n")
                    key = "Epsilon: " + str(eps) + " - Alpha: " + str(alp) + " - Gamma: " + str(gam)
                    if algorithm == 'Montecarlo':
                        agent_montecarlo = self.resolve_by_montecarlo()
                        results[key] = agent_montecarlo.calculate_statistics()
                    elif algorithm == 'Q-Learning':
                        agent_q_learning = self.resolve_by_q_learning(eps)
                        results[key] = agent_q_learning.calculate_statistics()
                    elif algorithm == 'Sarsa':
                        agent_sarsa = self.resolve_by_sarsa(eps, alp, gam)
                        results[key] = agent_sarsa.calculate_statistics()
                    else:
                        raise ValueError("Algoritmo no encontrado")
        # A partir de results, obtener el mejor en cada uno
        print("\nComparación de casos:\n")
        print("Mejor caso por recompensa media: " + max(results, key=lambda x: results[x]['mean_reward']))
        print("Mejor caso por longitud media de episodios: " + min(results, key=lambda x: results[x]['mean_length']))
        print("Mejor caso por porcentaje de éxito: " + max(results, key=lambda x: results[x]['success_rate']))
        print("Mejor caso por promedio de recompensa de los episodios exitosos: " + max(results, key=lambda x: results[x]['mean_success_reward']))
        print("Mejor caso por promedio de recompensa de los episodios fallidos: " + min(results, key=lambda x: results[x]['mean_failed_reward']))
        print("Mejor caso por tiempo de ejecución: " + min(results, key=lambda x: results[x]['time']))
        print("Mejor caso por número de episodios: " + min(results, key=lambda x: results[x]['num_episodes']))
        print("Mejor caso por máxima recompensa alcanzada: " + max(results, key=lambda x: results[x]['max_reward']))
        print("Mejor caso por mínima recompensa alcanzada: " + max(results, key=lambda x: results[x]['min_reward']))
        print("Menor tiempo de ejecución: " + min(results, key=lambda x: results[x]['time']))



    def compare_diffrent_environments(self, environments, algorithm='Montecarlo', epsilon=0.1, alpha=0.1, gamma=0.9):
        results = {}
        for env in environments:
            print("\n" + env + "\n")
            if env == 'FrozenLake-v0':
                self.environment = gym.make(env, render_mode='human')
            elif env == 'Taxi-v3':
                self.environment = gym.make(env, render_mode='human')
            else:
                raise ValueError("Entorno no encontrado")
            if algorithm == 'Montecarlo':
                agent_montecarlo = self.resolve_by_montecarlo()
                results[env] = agent_montecarlo.calculate_statistics()
            elif algorithm == 'Q-Learning':
                agent_q_learning = self.resolve_by_q_learning(epsilon)
                results[env] = agent_q_learning.calculate_statistics()
            elif algorithm == 'Sarsa':
                agent_sarsa = self.resolve_by_sarsa(epsilon, alpha, gamma)
                results[env] = agent_sarsa.calculate_statistics()
            else:
                raise ValueError("Algoritmo no encontrado")
            print(f"\nEntorno {env}:")
            self.print_stats()
        # A partir de results, obtener el mejor en cada uno
        print("\nComparación de entornos:\n")
        print("Mejor entorno por recompensa media: " + max(results, key=lambda x: results[x]['mean_reward']))
        print("Mejor entorno por longitud media de episodios: " + min(results, key=lambda x: results[x]['mean_length']))
        print("Mejor entorno por porcentaje de éxito: " + max(results, key=lambda x: results[x]['success_rate']))
        print("Mejor entorno por promedio de recompensa de los episodios exitosos: " + max(results, key=lambda x: results[x]['mean_success_reward']))
        print("Mejor entorno por promedio de recompensa de los episodios fallidos: " + min(results, key=lambda x: results[x]['mean_failed_reward']))
        print("Mejor entorno por tiempo de ejecución: " + min(results, key=lambda x: results[x]['time']))
        print("Mejor entorno por número de episodios: " + min(results, key=lambda x: results[x]['num_episodes']))
        print("Mejor entorno por máxima recompensa alcanzada: " + max(results, key=lambda x: results[x]['max_reward']))
        print("Mejor entorno por mínima recompensa alcanzada: " + max(results, key=lambda x: results[x]['min_reward']))
        print("Menor tiempo de ejecución: " + min(results, key=lambda x: results[x]['time']))


