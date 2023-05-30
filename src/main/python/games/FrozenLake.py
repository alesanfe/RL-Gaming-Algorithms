import gym
from gym.wrappers import RecordEpisodeStatistics

from src.main.python.DoubleQLearning import DoubleQLearning
from src.main.python.Sarsa import Sarsa
from src.main.python.aprendizaje_por_refuerzo import Montecarlo_IE, PolíticaEpsilonVoraz, Q_Learning
from src.main.python.games.Game import Game


class FrozenLake(Game):

    def __init__(self, environment='FrozenLake-v1', discount_factor=0.9, learning_factor=0.1, iterations=1000):
        super().__init__(RecordEpisodeStatistics(gym.make(environment, render_mode='human')), discount_factor, learning_factor, iterations)

    # Resolución del entorno Frozen Lake utilizando Montecarlo con inicios exploratorios
    def resolve_frozen_lake_by_montecarlo(self):
        return self.resolve_by_montecarlo()

    # Resolución del entorno Frozen Lake utilizando Q-Learning
    def resolve_frozen_lake_by_q_learning(self, epsilon=0.1):
        return self.resolve_by_q_learning(epsilon)

    # Resolución del entorno Frozen Lake utilizando Sarsa
    def resolve_frozen_lake_by_sarsa(self, epsilon=0.1, alpha=0.1, gamma=0.99):
        return self.resolve_by_sarsa(epsilon, alpha, gamma)

    # Resolución del entorno Frozen Lake utilizando Double Q-Learning
    def resolve_frozen_lake_by_double_q_learning(self, epsilon=0.1, alpha=0.1, gamma=0.99):
        return self.resolve_by_double_q_learning(epsilon, alpha, gamma)
