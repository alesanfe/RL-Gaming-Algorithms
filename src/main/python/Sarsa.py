import gym
import numpy as np

class Sarsa:
    def __init__(self, env, alpha, gamma, epsilon):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()  # Acción aleatoria
        else:
            action = np.argmax(self.q_table[state])  # Acción con mayor valor Q
        return action

    def update_q_table(self, state, action, reward, next_state, next_action):
        q_value = self.q_table[state, action]
        next_q_value = self.q_table[next_state, next_action]
        new_q_value = q_value + self.alpha * (reward + self.gamma * next_q_value - q_value)
        self.q_table[state, action] = new_q_value

    def train(self, num_episodes):
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


# Crear el entorno FrozenLake
env = gym.make('FrozenLake-v0')

# Crear una instancia de Sarsa
sarsa_agent = Sarsa(env, alpha=0.1, gamma=0.99, epsilon=0.1)

# Entrenar el agente
sarsa_agent.train(num_episodes=10000)

# Probar el agente entrenado
sarsa_agent.test(num_episodes=100)