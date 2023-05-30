import gym
import numpy as np

class DoubleQLearning:
    """Implementa el algoritmo Double Q-Learning."""

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
        self.initialize_q_tables()

    def initialize_q_tables(self):
        """Inicializa las tablas Q con valores aleatorios."""
        self.q1_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.q2_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))


    def choose_action(self, state):
        """Elige una acción para un estado dado."""
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()  # Acción aleatoria
        else:
            q_values = self.q1_table[state] + self.q2_table[state]
            action = np.argmax(q_values)  # Acción con mayor valor Q
        return action

    def update_q_tables(self, state, action, reward, next_state):
        """Actualiza las tablas Q."""
        if np.random.uniform(0, 1) < 0.5:
            max_q_next = np.max(self.q1_table[next_state])
            self.q1_table[state, action] += self.alpha * (reward + self.gamma * max_q_next - self.q1_table[state, action])
        else:
            max_q_next = np.max(self.q2_table[next_state])
            self.q2_table[state, action] += self.alpha * (reward + self.gamma * max_q_next - self.q2_table[state, action])

    def train(self, num_episodes):
        """Entrena el agente durante un número de episodios."""
        for episode in range(num_episodes):
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

        for episode in range(num_episodes):
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