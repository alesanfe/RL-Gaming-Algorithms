import gym
from gym.wrappers import RecordEpisodeStatistics

from src.main.python.DoubleQLearning import DoubleQLearning
from src.main.python.Sarsa import Sarsa
from src.main.python.aprendizaje_por_refuerzo import Montecarlo_IE, PolíticaEpsilonVoraz, Q_Learning
from src.main.python.games.Game import Game


class FrozenLake(Game):
    """
    El entorno Frozen Lake es un juego en el que el objetivo es cruzar un lago helado desde el punto de partida hasta
    el punto de llegada sin caer en agujeros. Cada estado en el juego se representa como un número entero que indica
    la posición actual en términos de fila y columna.
    """

    def __init__(self, environment='FrozenLake-v1', discount_factor=0.9, learning_factor=0.1, iterations=1000):
        super().__init__(RecordEpisodeStatistics(gym.make(environment, render_mode='human')), discount_factor,
                         learning_factor, iterations)

    def resolve_frozen_lake_by_montecarlo(self):
        """Resolución del entorno Frozen Lake utilizando Montecarlo con inicios exploratorios."""
        return self.resolve_by_montecarlo()

    #
    def resolve_frozen_lake_by_q_learning(self, epsilon=0.1):
        """Resolución del entorno Frozen Lake utilizando Q-Learning."""
        return self.resolve_by_q_learning(epsilon)

    def resolve_frozen_lake_by_sarsa(self, epsilon=0.1, alpha=0.1, gamma=0.99):
        """Resolución del entorno Frozen Lake utilizando Sarsa."""
        return self.resolve_by_sarsa(epsilon, alpha, gamma)

    def resolve_frozen_lake_by_double_q_learning(self, epsilon=0.1, alpha=0.1, gamma=0.99):
        """Resolución del entorno Frozen Lake utilizando Double Q-Learning."""
        return self.resolve_by_double_q_learning(epsilon, alpha, gamma)
