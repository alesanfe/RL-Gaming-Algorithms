from gym.wrappers import RecordEpisodeStatistics

from src.main.python.games.game import Game
from src.main.python.games.golf.golf_env import GolfEnv


class Golf(Game):
    """
    El entorno del juego de golf tiene como objetivo principal llevar la pelota de golf desde una posición inicial hasta
    el hoyo con la menor cantidad de golpes posibles. Hay dos palos disponibles para golpear la pelota,
    cada uno con características distintas.
    """

    def __init__(self, discount_factor=0.9, learning_factor=0.1, iterations=1000):
        super().__init__(RecordEpisodeStatistics(GolfEnv()), discount_factor, learning_factor, iterations)

    def resolve_golf_by_montecarlo(self):
        """Resolución del entorno Golf utilizando Montecarlo con inicios exploratorios."""
        return self.resolve_by_montecarlo()

    def resolve_golf_by_q_learning(self, epsilon=0.1):
        """Resolución del entorno Golf utilizando Q-Learning."""
        return self.resolve_by_q_learning(epsilon)

    def resolve_golf_by_sarsa(self, epsilon=0.1, alpha=0.1, gamma=0.99):
        """Resolución del entorno Golf utilizando Sarsa."""
        return self.resolve_by_sarsa(epsilon, alpha, gamma)
