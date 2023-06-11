import gym
from gym.wrappers import RecordEpisodeStatistics

from src.main.python.games.Game import Game

class Taxi(Game):
    """
    El entorno consiste en un mapa 5x5 que representa la ubicación de un taxi, un pasajero y cuatro destinos señalados
    con colores. El objetivo es que el taxi se mueva desde su posición inicial hasta la posición del pasajero, lo recoja
    y lo lleve al destino deseado lo más rápido posible.
    """

    def __init__(self, environment='Taxi-v3', discount_factor=0.9, learning_factor=0.1, iterations=1000):
        super().__init__(RecordEpisodeStatistics(gym.make(environment, render_mode='human')), discount_factor, learning_factor, iterations)

    def resolve_taxi_by_montecarlo(self):
        """Resolución del entorno Taxi utilizando Montecarlo con inicios exploratorios."""
        return self.resolve_by_montecarlo()

    def resolve_taxi_by_q_learning(self, epsilon=0.1):
        """Resolución del entorno Taxi utilizando Q-Learning."""
        return self.resolve_by_q_learning(epsilon)

    def resolve_taxi_by_sarsa(self, epsilon=0.1, alpha=0.1, gamma=0.99):
        """Resolución del entorno Taxi utilizando Sarsa."""
        return self.resolve_by_sarsa(epsilon, alpha, gamma)

    def resolve_taxi_by_double_q_learning(self, epsilon=0.1, alpha=0.1, gamma=0.99):
        """Resolución del entorno Taxi utilizando Double Q-Learning."""
        return self.resolve_by_double_q_learning(epsilon, alpha, gamma)


