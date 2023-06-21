import random
import numpy
from collections import defaultdict

from src.main.python.environment_statistic import EnvironmentStatistic


class PolíticaVoraz:
    """Implementa una política voraz sobre el valor de pares estado-acción."""

    def elige_acción(self, estado, espacio_de_acciones, tabla_q, máscara=None):
        """Elige una acción voraz que aplicar a un estado.

        Argumentos:
        estado -- un número entero representando el estado
        espacio_de_acciones -- el espacio de posibles acciones
                               (se asume de tipo Discrete)
        tabla_q -- un diccionario que asocia a cada estado un array con el
                   valor de cada acción para el estado
        máscara -- un array binario que indica las acciones elegibles
                   (el valor por defecto, None, representa que todas las
                    acciones son elegibles)

        Se elige aleatoriamente entre todas las acciones maximalmente
        valoradas para el estado.
        """
        if máscara is None:
            máscara = numpy.full(espacio_de_acciones.n, 1, dtype=numpy.int8)
        valores_acciones = tabla_q[estado]
        máscara_mejores_acciones = (
                valores_acciones == max(valores_acciones)
        ).astype(numpy.int8)
        acción_elegida = espacio_de_acciones.sample(
            máscara * máscara_mejores_acciones
        )
        return acción_elegida


class PolíticaEpsilonVoraz:
    """Implementa una política ε-voraz sobre el valor de pares estado-acción."""

    def __init__(self, epsilon):
        """Requiere el valor del parámetro ε de la política."""
        self.epsilon = epsilon

    def elige_acción(self, estado, espacio_de_acciones, tabla_q, máscara=None):
        """Elige una acción voraz que aplicar a un estado.

        Argumentos:
        estado -- un número entero representando el estado
        espacio_de_acciones -- el espacio de posibles acciones
                               (se asume de tipo Discrete)
        tabla_q -- un diccionario que asocia a cada estado un array con el
                   valor de cada acción para el estado
        máscara -- un array binario que indica las acciones elegibles
                   (el valor por defecto, None, representa que todas las
                    acciones son elegibles)

        En su caso, se elige aleatoriamente entre todas las acciones
        maximalmente valoradas para el estado.
        """
        if máscara is None:
            máscara = numpy.full(espacio_de_acciones.n, 1, dtype=numpy.int8)
        if random.random() < self.epsilon:
            acción_elegida = espacio_de_acciones.sample(máscara)
            return acción_elegida
        else:
            valores_acciones = tabla_q[estado]
            máscara_mejores_acciones = (
                    valores_acciones == max(valores_acciones)
            ).astype(numpy.int8)
            acción_elegida = espacio_de_acciones.sample(
                máscara * máscara_mejores_acciones
            )
            return acción_elegida


class Montecarlo_IE:
    """Implementa el algoritmo de Montecarlo con inicios exploratorios."""

    def __init__(
            self,
            entorno,
            factor_de_descuento,
            primera_visita=False
    ):
        """Crea una instancia del algoritmo.

        Argumentos:
        entorno -- un entorno implementado mediante la API de Gymnasium
                   (se asume que tanto el espacio de estados como el de
                    acciones son de tipo Discrete)
        factor_de_descuento -- un número real entre 0 y 1
        primera_visita -- un valor lógico que indica si el algoritmo es de
                          primera_visita o de cada visita (por defecto)
        """
        self.entorno = entorno
        self.factor_de_descuento = factor_de_descuento
        self.primera_visita = primera_visita
        self.política_exploratoria = PolíticaVoraz()
        self.statistics = EnvironmentStatistic(entorno)
        self.inicializa_tablas_q_y_r()

    def inicializa_tablas_q_y_r(self):
        """Inicializa las tablas usadas por el algoritmo.

        El atributo tabla_q es un diccionario que asocia a cada estado un
        array con el valor (inicialmente −∞) de cada acción para el estado.

        El atributo tabla_r es un diccionario que asocia a cada estado una
        lista (inicialmente vacía) con todas las recompensas acumuladas a
        partir del estado que se han observado.
        """
        cantidad_acciones = self.entorno.action_space.n
        self.tabla_q = defaultdict(
            lambda: numpy.full(cantidad_acciones, -numpy.inf)
        )
        self.tabla_r = defaultdict(
            lambda: tuple([] for _ in range(cantidad_acciones))
        )

    def elige_acción(self, estado, info):
        """Elige una acción a aplicar a un estado.

        Argumentos:
        estado -- un número entero representando el estado
        info -- información proporcionada por los métodos reset y step del
                entorno (no usada en esta implementación)
        """
        acción = self.política_exploratoria.elige_acción(
            estado,
            self.entorno.action_space,
            self.tabla_q
        )
        return acción

    def ejecuta_episodio(self):
        """Ejecuta un episodio para el entorno.

        El episodio comenzará en el estado proporcionado por el método reset
        del entorno y recorrerá los estados proporcionados por el método step
        del entorno hasta que se obtenga la señal de terminación o de truncado.
        """
        pares_estado_acción = []
        recompensas = []

        self.statistics.reset_episode()

        # El estado inicial es aleatorio
        estado_actual, info = self.entorno.reset()

        # La acción inicial es aleatoria
        acción = self.entorno.action_space.sample()
        while True:
            pares_estado_acción.append((estado_actual, acción))
            estado_siguiente, recompensa, terminado, truncado, info = (
                self.entorno.step(acción)
            )
            recompensas.append(recompensa)

            self.statistics.continue_episode(recompensa)

            if terminado or truncado:
                self.statistics.add_episode(estado_siguiente)
                break
            estado_actual = estado_siguiente
            acción = self.elige_acción(estado_actual, info)
        U = 0
        while pares_estado_acción:
            estado, acción = pares_estado_acción.pop()
            recompensa = recompensas.pop()
            U = self.factor_de_descuento * U + recompensa
            if (not self.primera_visita or
                    not (estado, acción) in pares_estado_acción):
                self.tabla_r[estado][acción].append(U)
                self.tabla_q[estado][acción] = numpy.mean(
                    self.tabla_r[estado][acción]
                )

    def entrena(self, número_episodios):
        """Ejecuta el algoritmo durante un cierto número de episodios.

        Argumentos:
        número_episodios -- entero no negativo que establece el número de
                            episodios a entrenar
        """
        self.statistics.reset()
        for _ in range(número_episodios):
            self.ejecuta_episodio()

    def calculate_statistics(self):
        """Calcula las estadísticas a partir de los datos de los episodios."""
        return self.statistics.calculate_statistics()

    def get_policy_for_state(self, state):
        """Devuelve la política elegida para un estado.

        Argumentos:
        state -- Estado para el que se desea obtener la política.
        """
        q_values = self.q_table[state]
        max_q_value = numpy.max(q_values)
        best_actions = numpy.where(q_values == max_q_value)[0]
        policy = numpy.zeros_like(q_values)
        policy[best_actions] = 1.0 / len(best_actions)
        return policy

    def get_policy(self):
        """Devuelve la política elegida para todos los estados."""
        policy = {}
        for state in self.q_table.keys():
            policy[state] = self.get_policy_for_state(state)
        return policy


class Q_Learning:
    """Implementa el algoritmo Q-learning."""

    def __init__(
            self,
            entorno,
            factor_de_descuento,
            tasa_de_aprendizaje,
            política_exploratoria
    ):
        """Crea una instancia del algoritmo.

        Argumentos:
        entorno -- un entorno implementado mediante la API de Gymnasium
                   (se asume que tanto el espacio de estados como el de
                    acciones son de tipo Discrete)
        factor_de_descuento -- un número real entre 0 y 1
        tasa_de_aprendizaje -- un número real mayor que 0 y menor o igual que 1
        política_exploratoria -- una instancia de PolíticaEpsilonVoraz
        """
        self.entorno = entorno
        self.tasa_de_aprendizaje = tasa_de_aprendizaje
        self.factor_de_descuento = factor_de_descuento
        self.política_exploratoria = política_exploratoria
        self.statistics = EnvironmentStatistic(entorno)
        self.inicializa_tabla_q()

    def inicializa_tabla_q(self):
        """Inicializa la tabla usada por el algoritmo.

        El atributo tabla_q es un diccionario que asocia a cada estado un
        array con el valor (inicialmente 0) de cada acción para el estado.
        """
        cantidad_acciones = self.entorno.action_space.n
        self.tabla_q = defaultdict(lambda: numpy.zeros(cantidad_acciones))

    def actualiza_tabla_q(
            self,
            estado_actual,
            acción,
            recompensa,
            estado_siguiente
    ):
        """Actualiza el valor de una acción para un estado.

        Argumentos:
        estado_actual -- un número entero representando el estado actual
        acción -- un número entero representando la acción aplicada
        recompensa -- un número real representando la recompensa observada
        estado_siguiente -- un número entero representando el nuevo estado
                            observado
        """
        máximo_valor_q = max(self.tabla_q[estado_siguiente])
        error_DT = (
                recompensa +
                self.factor_de_descuento * máximo_valor_q -
                self.tabla_q[estado_actual][acción]
        )
        self.tabla_q[estado_actual][acción] += (
                self.tasa_de_aprendizaje * error_DT
        )

    def elige_acción(self, estado, info):
        """Elige una acción a aplicar a un estado.

        Argumentos:
        estado -- un número entero representando el estado
        info -- información proporcionada por los métodos reset y step del
                entorno (no usada en esta implementación)
        """
        acción = self.política_exploratoria.elige_acción(
            estado,
            self.entorno.action_space,
            self.tabla_q
        )
        return acción

    def ejecuta_episodio(self):
        """Ejecuta un episodio para el entorno.

        El episodio comenzará en el estado proporcionado por el método reset
        del entorno y recorrerá los estados proporcionados por el método step
        del entorno hasta que se obtenga la señal de terminación o de truncado.
        """
        estado_actual, info = self.entorno.reset()
        self.statistics.reset_episode()

        while True:
            acción = self.elige_acción(estado_actual, info)
            estado_siguiente, recompensa, terminado, truncado, info = (
                self.entorno.step(acción)
            )
            self.actualiza_tabla_q(
                estado_actual, acción, recompensa, estado_siguiente
            )

            self.statistics.continue_episode(recompensa)

            if terminado or truncado:
                self.statistics.add_episode(estado_siguiente)
                break

            estado_actual = estado_siguiente

    def entrena(self, número_episodios):
        """Ejecuta el algoritmo durante un cierto número de episodios.

        Argumentos:
        número_episodios -- entero no negativo que establece el número de
                            episodios a entrenar
        """
        self.statistics.reset()
        for _ in range(número_episodios):
            self.ejecuta_episodio()

    def calculate_statistics(self):
        """Calcula las estadísticas a partir de los datos de los episodios."""
        return self.statistics.calculate_statistics()

    def get_policy_for_state(self, state):
        """Devuelve la política elegida para un estado.

        Argumentos:
        state -- Estado para el que se desea obtener la política.
        """
        q_values = self.q_table[state]
        max_q_value = numpy.max(q_values)
        best_actions = numpy.where(q_values == max_q_value)[0]
        policy = numpy.zeros_like(q_values)
        policy[best_actions] = 1.0 / len(best_actions)
        return policy

    def get_policy(self):
        """Devuelve la política elegida para todos los estados."""
        policy = {}
        for state in self.q_table.keys():
            policy[state] = self.get_policy_for_state(state)
        return policy