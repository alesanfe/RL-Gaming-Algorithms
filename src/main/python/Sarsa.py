from dataclasses import dataclass, field
import gym
import numpy as np

@dataclass
class Sarsa:
    """Implementa el algoritmo Sarsa."""

    env: gym.Env
    alpha: float
    gamma: float
    epsilon: float
    q_table: np.ndarray = field(init=False)

    def __post_init__(self):
        """Inicializa la tabla Q con valores aleatorios."""
        self.initialize_q_table()

    def initialize_q_table(self):
        """Inicializa la tabla Q con valores aleatorios."""
        self.q_table = np.random.uniform(low=0, high=1, size=(self.env.observation_space.n, self.env.action_space.n))
        # Para el estado terminal, las acciones tienen valor 0
        self.q_table[self.env.observation_space.n - 1] = 0

    def choose_action(self, state):
        """Elige una acción para un estado dado."""
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()  # Acción aleatoria para exploración
        else:
            action = self.get_action(state)  # Acción con mayor valor Q para explotación
        return action

    def update_q_table(self, state, action, reward, next_state, next_action):
        """Actualiza la tabla Q utilizando el algoritmo Sarsa."""
        q_value = self.q_table[state, action]
        next_q_value = self.q_table[next_state, next_action]
        new_q_value = q_value + self.alpha * (reward + self.gamma * next_q_value - q_value)
        self.q_table[state, action] = new_q_value

    def get_action(self, state):
        """Obtiene la acción óptima para un estado dado según la tabla Q aprendida."""
        return np.argmax(self.q_table[state])

    def train(self, num_episodes):
        """Entrena el agente durante un número de episodios."""
        for _ in range(num_episodes):
            state = self.env.reset()
            action = self.choose_action(state)

            while True:
                next_state, reward, done, _ = self.env.step(action)
                next_action = self.choose_action(next_state)

                self.update_q_table(state, action, reward, next_state, next_action)

                state = next_state
                action = next_action

                if done:
                    break

        print("Training finished.")

    def test(self, num_episodes):
        """Evalúa el agente durante un número de episodios."""
        total_reward = 0

        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0

            while True:
                action = np.argmax(self.q_table[state])
                state, reward, done, _ = self.env.step(action)

                episode_reward += reward

                if done:
                    break

            total_reward += episode_reward

        average_reward = total_reward / num_episodes
        print(f"Average reward over {num_episodes} episodes: {average_reward}")