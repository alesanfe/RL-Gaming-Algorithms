import gym
from gym import spaces
import numpy as np

class GolfEnv(gym.Env):
    def __init__(self):
        self.field_width = 10  # Ancho del campo de golf
        self.field_height = 10  # Alto del campo de golf
        self.min_force = 1  # Fuerza mínima del golpe
        self.max_force = 8  # Fuerza máxima del golpe
        self.max_inaccuracy = 3  # Máxima imprecisión del golpe
        self.action_space = spaces.Discrete(16)  # 16 acciones posibles (2 palos x 8 direcciones)
        self.observation_space = spaces.Box(low=0, high=self.field_width, shape=(2,), dtype=np.float32)  # Posición de la pelota (x, y)
        self.ball_position = np.zeros(2)
        self.hole_position = np.array([self.field_width - 1, self.field_height - 1])
        self.steps = 0
        self.max_steps = 100

    def reset(self):
        self.ball_position = np.random.uniform(low=0, high=self.field_width, size=(2,))
        self.steps = 0
        return self.ball_position

    def step(self, action):
        assert self.action_space.contains(action), "Acción inválida"

        # Obtener la fuerza y dirección del golpe según la acción seleccionada
        if action < 8:
            club = 'big'  # Palo que permite golpear con gran fuerza
            direction = self._action_to_direction(action)
            force = np.random.uniform(low=self.min_force, high=self.max_force)
            inaccuracy = np.random.randint(low=-self.max_inaccuracy, high=self.max_inaccuracy+1)
        else:
            club = 'small'  # Palo que permite golpear con poca fuerza
            direction = self._action_to_direction(action - 8)
            force = np.random.uniform(low=self.min_force, high=self.min_force + 3)
            inaccuracy = np.random.randint(low=-1, high=2)

        # Calcular la nueva posición de la pelota después del golpe
        new_position = self.ball_position + (force + inaccuracy) * direction

        # Comprobar si la pelota llega al hoyo
        done = False
        if np.array_equal(new_position, self.hole_position):
            done = True
            reward = -self.steps  # Recompensa negativa basada en el número de pasos
        else:
            # Limitar la posición de la pelota dentro del campo de golf
            new_position = np.clip(new_position, [0, 0], [self.field_width - 1, self.field_height - 1])

            # Calcular la recompensa negativa basada en la distancia a la bandera
            distance_to_hole = np.linalg.norm(new_position - self.hole_position)
            reward = -distance_to_hole

            # Comprobar si la pelota sale del campo de juego
            if not np.array_equal(new_position, self.ball_position):
                reward -= 1  # Penalizar por salir del campo de juego

        self.ball_position = new_position
        self.steps += 1

        return self.ball_position, reward, done, {}

    def render(self):
        # Crear una representación visual del campo de golf y la posición de la pelota
        field = np.zeros((self.field_height, self.field_width))
        field[int(self.hole_position[1]), int(self.hole_position[0])] = 2  # Bandera
        field[int(self.ball_position[1]), int(self.ball_position[0])] = 1  # Pelota

        print(field)

    def _action_to_direction(self, action):
        # Mapear las acciones a las direcciones correspondientes
        directions = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
        return directions[action]