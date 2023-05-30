import gym

from src.main.python.games.Game import Game


class Taxi(Game):

    def __init__(self, environment='Taxi-v3', discount_factor=0.9, learning_factor=0.1, iterations=1000):
        super().__init__(gym.make(environment, render_mode='human'), discount_factor, learning_factor, iterations)

    # Resolución del entorno taxi utilizando Montecarlo IE
    def resolve_taxi_by_montecarlo(self):
        return self.resolve_by_montecarlo()

    # Resolución del entorno Taxi utilizando Q-Learning
    def resolve_taxi_by_q_learning(self, epsilon=0.1):
        return self.resolve_by_q_learning(epsilon)

    # Resolución del entorno Taxi utilizando Sarsa
    def resolve_taxi_by_sarsa(self, epsilon=0.1, alpha=0.1, gamma=0.99):
        return self.resolve_by_sarsa(epsilon, alpha, gamma)

    # Resolución del entorno Taxi utilizando Double Q-Learning
    def resolve_taxi_by_double_q_learning(self, epsilon=0.1, alpha=0.1, gamma=0.99):
        return self.resolve_by_double_q_learning(epsilon, alpha, gamma)


