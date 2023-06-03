from dataclasses import dataclass, field
import numpy as np
from gym import Env

@dataclass
class DoubleQLearning:
    """Implementa el algoritmo Double Q-Learning."""

    env: Env
    alpha: float
    gamma: float
    epsilon: float
    q1_table: np.ndarray = field(init=False)
    q2_table: np.ndarray = field(init=False)

    def __post_init__(self):
        self.initialize_q_tables()

    def initialize_q_tables(self):
        """Inicializa las tablas Q con valores aleatorios."""
        # Crea las tablas Q con valores aleatorios en el rango [0, 1]
        self.q1_table = np.random.uniform(low=0, high=1, size=(self.env.observation_space.n, self.env.action_space.n))
        self.q2_table = np.random.uniform(low=0, high=1, size=(self.env.observation_space.n, self.env.action_space.n))
        # Para el estado terminal, las acciones tienen valor 0
        self.q1_table[self.env.observation_space.n - 1] = 0
        self.q2_table[self.env.observation_space.n - 1] = 0

    def choose_action(self, state):
        """Elige una acción para un estado dado."""
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()  # Acción aleatoria para exploración
        else:
            action = self.get_action(state)  # Acción con mayor valor Q para explotación
        return action

    def update_q_tables(self, state, action, reward, next_state):
        """Actualiza las tablas Q utilizando el algoritmo Double Q-Learning."""
        # Elige la siguiente acción y los valores Q correspondientes de las tablas Q
        if np.random.uniform(0, 1) < 0.5:
            next_action = np.argmax(self.q1_table[next_state])
            q_value = self.q1_table[state, action]
            next_q_value = self.q2_table[next_state, next_action]
            # Actualiza el valor Q en la tabla q1_table usando el algoritmo Double Q-Learning
            self.q1_table[state, action] = q_value + self.alpha * (reward + self.gamma * next_q_value - q_value)
        else:
            next_action = np.argmax(self.q2_table[next_state])
            q_value = self.q2_table[state, action]
            next_q_value = self.q1_table[next_state, next_action]
            # Actualiza el valor Q en la tabla q2_table usando el algoritmo Double Q-Learning
            self.q2_table[state, action] = q_value + self.alpha * (reward + self.gamma * next_q_value - q_value)

    def get_action(self, state):
        """Obtiene la acción óptima para un estado dado según las tablas Q aprendidas."""
        # Retorna la acción con el mayor valor Q en la suma de las tablas q1_table y q2_table para el estado dado
        return np.argmax(self.q1_table[state] + self.q2_table[state])

    def train(self, num_episodes):
        """Entrena el agente durante un número de episodios."""
        for _ in range(num_episodes):
            state = self.env.reset()

            while True:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)

                self.update_q_tables(state, action, reward, next_state)

                state = next_state

                if done:
                    break

        print("Training finished.")

    def test(self, num_episodes):
        """Prueba el agente durante un número de episodios."""
        total_reward = 0

        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0

            while True:
                action = np.argmax(self.q1_table[state] + self.q2_table[state])
                state, reward, done, _ = self.env.step(action)

                episode_reward += reward

                if done:
                    break

            total_reward += episode_reward

        average_reward = total_reward / num_episodes
        print(f"Average reward over {num_episodes} episodes: {average_reward}")