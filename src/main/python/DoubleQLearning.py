import gym
import numpy as np

class DoubleQLearning:
    def __init__(self, env, alpha, gamma, epsilon):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q1_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.q2_table = np.zeros((env.observation_space.n, env.action_space.n))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()  # Acción aleatoria
        else:
            q_values = self.q1_table[state] + self.q2_table[state]
            action = np.argmax(q_values)  # Acción con mayor valor Q
        return action

    def update_q_tables(self, state, action, reward, next_state):
        if np.random.uniform(0, 1) < 0.5:
            max_q_next = np.max(self.q1_table[next_state])
            self.q1_table[state, action] += self.alpha * (reward + self.gamma * max_q_next - self.q1_table[state, action])
        else:
            max_q_next = np.max(self.q2_table[next_state])
            self.q2_table[state, action] += self.alpha * (reward + self.gamma * max_q_next - self.q2_table[state, action])

    def train(self, num_episodes):
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


# Crear el entorno FrozenLake
env = gym.make('FrozenLake-v0')

# Crear una instancia de DoubleQLearning
double_q_learning_agent = DoubleQLearning(env, alpha=0.1, gamma=0.99, epsilon=0.1)

# Entrenar el agente
double_q_learning_agent.train(num_episodes=10000)

# Probar el agente entrenado
double_q_learning_agent.test(num_episodes=100)