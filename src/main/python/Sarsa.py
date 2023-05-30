import gym
import numpy as np

class Sarsa:
    """Implementa el algoritmo Sarsa."""

    def __init__(self, env, alpha, gamma, epsilon):
        """Crea una instancia del algoritmo.

                Argumentos:
                entorno -- un entorno implementado mediante la API de Gymnasium
                           (se asume que tanto el espacio de estados como el de
                            acciones son de tipo Discrete)
                alpha -- un número real entre 0 y 1
                gamma -- un número real mayor que 0 y menor o igual que 1
                epsilon -- un número real entre 0 y 1
                """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.initialize_q_table()

    def initialize_q_table(self):
        """Inicializa la tabla Q con valores aleatorios."""
        self.q_table = np.random.uniform(low=0, high=1, size=(self.env.observation_space.n, self.env.action_space.n))

    def choose_action(self, state):
        """Elige una acción para un estado dado."""
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()  # Acción aleatoria
        else:
            action = np.argmax(self.q_table[state])  # Acción con mayor valor Q
        return action

    def update_q_table(self, state, action, reward, next_state, next_action):
        """Actualiza la tabla Q."""
        q_value = self.q_table[state, action]
        next_q_value = self.q_table[next_state, next_action]
        new_q_value = q_value + self.alpha * (reward + self.gamma * next_q_value - q_value)
        self.q_table[state, action] = new_q_value

    def train(self, num_episodes):
        """Entrena el agente durante un número de episodios."""
        for episode in range(num_episodes):
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

        for episode in range(num_episodes):
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