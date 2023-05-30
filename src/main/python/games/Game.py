import gym

from src.main.python.DoubleQLearning import DoubleQLearning
from src.main.python.Sarsa import Sarsa
from src.main.python.aprendizaje_por_refuerzo import Montecarlo_IE, PolíticaEpsilonVoraz, Q_Learning


class Game:

    def __init__(self, environment, discount_factor, learning_factor, iterations):
        self.environment = environment
        self.discount_factor = discount_factor
        self.learning_factor = learning_factor
        self.iterations = iterations

    # Resolución del entorno utilizando Montecarlo con inicios exploratorios
    def resolve_by_montecarlo(self):
        agent = Montecarlo_IE(self.environment, self.discount_factor)
        agent.entrena(self.iterations)
        return agent

    # Resolución del entorno utilizando Q-Learning
    def resolve_by_q_learning(self, epsilon):
        export_policy = PolíticaEpsilonVoraz(epsilon)
        agent = Q_Learning(self.environment, self.discount_factor, self.learning_factor, export_policy)
        agent.entrena(self.iterations)
        return agent

    # Resolución del entorno utilizando Sarsa
    def resolve_by_sarsa(self, epsilon, alpha, gamma):
        export_policy = PolíticaEpsilonVoraz(epsilon)
        agent = Sarsa(self.environment, alpha, gamma, export_policy)
        agent.train(self.iterations)
        return agent

    # Resolución del entorno utilizando Double Q-Learning
    def resolve_by_double_q_learning(self, epsilon, alpha, gamma):
        export_policy = PolíticaEpsilonVoraz(epsilon)
        agent = DoubleQLearning(self.environment, alpha, gamma, export_policy)
        agent.train(self.iterations)
        return agent