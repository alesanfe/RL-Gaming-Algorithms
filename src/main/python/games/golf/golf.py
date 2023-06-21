import numpy
from gym.wrappers import RecordEpisodeStatistics

from src.main.python.games.game import Game
from src.main.python.games.golf.golf_env import GolfEnv
from src.main.python.games.golf.position_golf_ball import PositionGolfBall


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

    def resolve_golf_by_double_q_learning(self, epsilon, alpha, gamma):
        return self.resolve_by_double_q_learning(epsilon, alpha, gamma)

    def show_policy(self, agent):
        direction_symbols = {
            0: "↓",  # Mover hacia el sur (abajo)
            1: "↑",  # Mover hacia el norte (arriba)
            2: "→",  # Mover hacia el este (derecha)
            3: "←",  # Mover hacia el oeste (izquierda)
        }

        policy = agent.get_policy()
        env = self.environment
        width = env.width
        height = env.height

        for row in range(width):
            for col in range(height):
                state = PositionGolfBall(row, col)
                if state in env.target_location:
                    # Si la celda es una ubicación objetivo, mantener su valor original
                    print("G", end=" ")
                else:
                    # Si la celda no es una ubicación objetivo, obtener la política para el estado actual
                    action = numpy.argmax(policy[state])
                    club_index, direction_index, force_index = self.take_action(action)
                    action_str = f"({club_index}, {self.direction_symbols[direction_index]}, {env.golfs_club[club_index] + force_index})"
                    print(action_str, end=" ")
            print()


