import gym
import numpy
from gym.wrappers import RecordEpisodeStatistics

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

    def show_policy(self, agent):
        action_symbols = {
            0: "↓",  # Mover hacia el sur (abajo)
            1: "↑",  # Mover hacia el norte (arriba)
            2: "→",  # Mover hacia el este (derecha)
            3: "←",  # Mover hacia el oeste (izquierda)
        }

        policy = agent.get_policy()
        env = self.environment
        board_shape = env.desc.shape

        # Crear el tablero con la representación gráfica de la política
        policy_board = numpy.empty(board_shape, dtype="<U1")
        for row in range(board_shape[0]):
            for col in range(board_shape[1]):
                state = env.desc[row, col]
                if state in [b'S', b'G', b'H']:
                    # Si la celda es un estado inicial, objetivo o agujero, mantener su valor original
                    policy_board[row, col] = state.decode()
                else:
                    # Si la celda es un estado intermedio, asignar el símbolo correspondiente a la acción elegida en la política
                    action = numpy.argmax(policy[state])
                    policy_board[row, col] = action_symbols[action]

        # Mostrar el tablero con la política
        for row in policy_board:
            print(" ".join(row))



