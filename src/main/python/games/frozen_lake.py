import gym
import numpy
from gym.wrappers import RecordEpisodeStatistics

from src.main.python.games.game import Game


class FrozenLake(Game):
    """
    El entorno Frozen Lake es un juego en el que el objetivo es cruzar un lago helado desde el punto de partida hasta
    el punto de llegada sin caer en agujeros. Cada estado en el juego se representa como un número entero que indica
    la posición actual en términos de fila y columna.
    """

    def __init__(self, environment='FrozenLake-v1', discount_factor=0.9, learning_factor=0.1, iterations=1000):
        super().__init__(RecordEpisodeStatistics(gym.make(environment)), discount_factor,
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

    def show_policy(self, agent):
        action_symbols = {
            1: "↓",  # Mover hacia el sur (abajo)
            3: "↑",  # Mover hacia el norte (arriba)
            2: "→",  # Mover hacia el este (derecha)
            0: "←",  # Mover hacia el oeste (izquierda)
        }

        policy = agent.get_policy()
        env = self.environment
        board_shape = env.desc.shape

        # Crear el tablero con la representación gráfica de la política
        policy_board = numpy.empty(board_shape, dtype="<U1")
        size = board_shape[0] * board_shape[1]
        for state in range(size - 1):
            row, col = state // board_shape[0], state % board_shape[0]
            if state in policy.keys():
                action = numpy.argmax(policy[state])
                policy_board[row, col] = action_symbols[action]
            else:
                policy_board[row, col] = "·"

        # Mostrar el tablero con la política
        for row in policy_board:
            print(" ".join(row))
