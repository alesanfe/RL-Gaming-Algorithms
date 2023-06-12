from collections import defaultdict

import numpy

from src.main.python.environment_statistic import EnvironmentStatistic


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
        self.statistics = EnvironmentStatistic(env)
        self.initialize_q_table()

    def initialize_q_table(self):
        """Inicializa la tabla Q con valores aleatorios."""
        cantidad_acciones = self.env.action_space.n
        self.q_table = defaultdict(lambda: numpy.random.uniform(0, 1, cantidad_acciones))
        # Para el estado terminal, las acciones tienen valor 0
        for estado in self.statistics.get_terminal_states():
            self.q_table[estado] = numpy.zeros(cantidad_acciones)

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
        self.statistics.reset_episode()

        while True:
            action = self.choose_action(current_state, info)
            next_state, reward, truncated, done, info = self.env.step(action)

            next_action = self.choose_action(next_state, info)

            self.update_q_table(current_state, action, reward, next_state, next_action)

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

