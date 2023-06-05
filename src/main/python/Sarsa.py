import numpy


class Sarsa:

    def __init__(
            self,
            env,
            discount_factor,
            learning_factor,
            export_policy
    ):
        """
        Crea una instacia del algoritmo.

        Argumentos:
        env -- Entorno en el que se ejecuta el algoritmo.
        discount_factor -- Factor de descuento.
        learning_factor -- Factor de aprendizaje.
        export_policy -- Política de exportación.
        """
        self.env = env
        self.discount_factor = discount_factor
        self.learning_factor = learning_factor
        self.export_policy = export_policy
        self.initialize_q_table()

    def initialize_q_table(self):
        """Inicializa la tabla Q con valores aleatorios."""
        self.q_table = numpy.random.uniform(low=0, high=1, size=(self.env.observation_space.n, self.env.action_space.n))
        # Para el estado terminal, las acciones tienen valor 0
        self.q_table[self.env.observation_space.n - 1] = 0

    def update_q_table(self, state, action, reward, next_state, next_action):
        """Actualiza el valor de una acción para un estado.

        Argumentos:
        state -- Estado.
        action -- Acción.
        reward -- Recompensa.
        next_state -- Siguiente estado.
        next_action -- Siguiente acción.
        """
        q_value = self.q_table[state, action]
        next_q_value = self.q_table[next_state, next_action]
        new_q_value = q_value + self.learning_factor * (reward + self.discount_factor * next_q_value - q_value)
        self.q_table[state, action] = new_q_value

    def choose_action(self, state, info):
        """Elige una acción para un estado dado.

        Argumentos:
        state -- Estado.
        info -- Información adicional.
        """
        action = self.export_policy.elige_acción(
            state,
            self.env.action_space,
            self.q_table
        )
        return action

    def execute_episode(self):
        """Ejecuta un episodio para el entorno.

        El episodio comenzará en el estado proporcionado por el método reset
        del entorno y recorrerá los estados proporcionados por el método step
        del entorno hasta que se obtenga la señal de terminación o de truncado.
        """
        current_state, info = self.env.reset()
        self.env.render()
        episode_reward = 0
        episode_length = 0

        while True:
            action = self.choose_action(current_state, info)
            next_state, reward, truncated, done, info = self.env.step(action)
            self.env.render()
            next_action = self.choose_action(next_state, info)

            self.update_q_table(current_state, action, reward, next_state, next_action)

            current_state = next_state

            episode_reward += reward
            episode_length += 1

            if done or truncated:
                self.episode_data['episodes'].append(
                    {'cumulative_reward': reward, 'episode_length': episode_length})
                self.episode_data['total']['cumulative_rewards'] += episode_reward
                self.episode_data['total']['episode_lengths'].append(episode_length)
                break

    def train(self, num_episodes):
        """Ejecuta el algoritmo durante un cierto número de episodios.

        Argumentos:
        num_episodes -- entero no negativo que establece el número de episodios a entrenar
        """
        self.episode_data = {'episodes': [], 'total': {'cumulative_rewards': 0, 'episode_lengths': []}}
        for _ in range(num_episodes):
            self.execute_episode()

    def calculate_statistics(self):
        """Calcula las estadísticas a partir de los datos de los episodios."""
        num_episodes = len(self.episode_data['episodes'])
        cumulative_rewards = [episode_data['cumulative_reward'] for episode_data in self.episode_data['episodes']]
        episode_lengths = [episode_data['episode_length'] for episode_data in self.episode_data['episodes']]

        mean_reward = numpy.mean(cumulative_rewards)
        reward_std = numpy.std(cumulative_rewards)
        mean_length = numpy.mean(episode_lengths)
        length_std = numpy.std(episode_lengths)
        max_reward = numpy.max(cumulative_rewards)
        min_reward = numpy.min(cumulative_rewards)
        num_success_episodes = len([reward for reward in cumulative_rewards if reward > 0])
        success_rate = (num_success_episodes / num_episodes) * 100

        success_rewards = [reward for reward in cumulative_rewards if reward > 0]
        failed_rewards = [reward for reward in cumulative_rewards if reward <= 0]
        mean_success_reward = numpy.mean(success_rewards)
        mean_failed_reward = numpy.mean(failed_rewards)

        statistics = {
            'mean_reward': mean_reward,
            'reward_std': reward_std,
            'mean_length': mean_length,
            'length_std': length_std,
            'num_episodes': num_episodes,
            'max_reward': max_reward,
            'min_reward': min_reward,
            'num_success_episodes': num_success_episodes,
            'success_rate': success_rate,
            'mean_success_reward': mean_success_reward,
            'mean_failed_reward': mean_failed_reward,
            'time': self.env.time
        }

        return statistics

