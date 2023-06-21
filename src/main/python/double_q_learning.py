from collections import defaultdict

import numpy
import numpy as np

from src.main.python.environment_statistic import EnvironmentStatistic


class DoubleQLearning:

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
        self.q1_table = None
        self.env = env
        self.discount_factor = discount_factor
        self.learning_factor = learning_factor
        self.export_policy = export_policy
        self.statistics = EnvironmentStatistic(env)
        self.q1_table = self._initialize_q_table()
        self.q2_table = self._initialize_q_table()

    def _initialize_q_table(self):
        # Crea las tablas Q con valores aleatorios en el rango [0, 1]
        cantidad_acciones = self.env.action_space.n
        q_table = defaultdict(lambda: numpy.random.uniform(0, 1, cantidad_acciones))
        # Para el estado terminal, las acciones tienen valor 0
        for estado in self.statistics.get_terminal_states():
            q_table[estado] = numpy.zeros(cantidad_acciones)
        return q_table

    def choose_action(self, state, info):
        """Elige una acción para un estado dado.

                Argumentos:
                state -- Estado.
                info -- Información adicional.
                """
        combined_q_values = {}
        combined_q_values.update(self.q1_table[state])
        combined_q_values.update(self.q2_table[state])

        action = self.export_policy.elige_acción(
            state,
            self.env.action_space,
            self.get_q_table()
        )
        return action

    def get_q_table(self):
        keys = set(self.q1_table.keys()).union(set(self.q2_table.keys()))
        q_table = {}
        for key in keys:
            action_q1_table = self.q1_table[key]
            action_q2_table = self.q2_table[key]
            action_q_table = [q1 + q2 for q1, q2 in zip(action_q1_table, action_q2_table)]
            q_table[key] = action_q_table
        return q_table

    def update_q_tables(self, state, action, reward, next_state, next_action):
        """Actualiza las tablas Q utilizando el algoritmo Double Q-Learning."""
        # Elige la siguiente acción y los valores Q correspondientes de las tablas Q

        if np.random.uniform(0, 1) < 0.5:
            self.q1_table = self.update_q_table(self.q1_table, action, next_state, reward, state, next_action)
        else:
            self.q2_table = self.update_q_table(self.q2_table, action, next_state, reward, state, next_action)

    def update_q_table(self, q_table, action, next_state, reward, state, next_action):
        next_action = np.argmax(q_table[next_state])
        q_value = q_table[state][action]
        next_q_value = q_table[next_state][next_action]
        # Actualiza el valor Q en la tabla q_table usando el algoritmo Double Q-Learning
        q_table[state][action] = q_value + self.learning_factor * (
                    reward + self.discount_factor * next_q_value - q_value)
        return q_table

    def execute_episode(self):
        """Ejecuta un episodio para el entorno.

        El episodio comenzará en el estado proporcionado por el método reset
        del entorno y recorrerá los estados proporcionados por el método step
        del entorno hasta que se obtenga la señal de terminación o de truncado.
        """
        current_state, info = self.env.reset()
        self.statistics.reset_episode()

        while True:
            action = self.choose_action(current_state, info)

            next_state, reward, truncated, done, info = self.env.step(action)

            next_action = self.choose_action(next_state, info)

            self.update_q_tables(current_state, action, reward, next_state, next_action)

            self.statistics.continue_episode(reward)

            if done or truncated:
                self.statistics.add_episode(next_state)
                break

            current_state = next_state

    def train(self, num_episodes):
        """Ejecuta el algoritmo durante un cierto número de episodios.

        Argumentos:
        num_episodes -- entero no negativo que establece el número de episodios a entrenar
        """
        self.statistics.reset()
        for _ in range(num_episodes):
            self.execute_episode()

    def calculate_statistics(self):
        """Calcula las estadísticas a partir de los datos de los episodios."""
        return self.statistics.calculate_statistics()

    def get_policy_for_state(self, state):
        """Devuelve la política elegida para un estado.

        Argumentos:
        state -- Estado para el que se desea obtener la política.
        """
        q_values = self.get_q_table()[state]
        max_q_value = numpy.max(q_values)
        best_actions = numpy.where(q_values == max_q_value)[0]
        policy = numpy.zeros_like(q_values)
        policy[best_actions] = 1.0 / len(best_actions)
        return policy

    def get_policy(self):
        """Devuelve la política elegida para todos los estados."""
        policy = {}
        for state in self.get_q_table().keys():
            policy[state] = self.get_policy_for_state(state)
        return policy
