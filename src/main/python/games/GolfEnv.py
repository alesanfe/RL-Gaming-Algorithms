class GolfEnv(gym.Env):
    def __init__(self):
        self.max_steps = 100  # Número máximo de pasos por episodio
        self.start_position = [0, 0]  # Posición de salida de la pelota
        self.hole_position = [5, 5]  # Posición del hoyo
        self.action_space = spaces.Discrete(16)  # Espacio de acciones (dos palos x ocho direcciones)
        self.observation_space = spaces.Box(low=0, high=5, shape=(2,), dtype=np.float32)  # Espacio de observación (posición x, posición y)
        self.current_position = self.start_position
        self.current_step = 0

    def step(self, action):
        dx, dy, f, i = self._decode_action(action)
        new_x = self.current_position[0] + (f + i) * dx
        new_y = self.current_position[1] + (f + i) * dy
        self.current_position = [new_x, new_y]
        self.current_step += 1

        if self.current_position == self.hole_position:
            reward = 1.0  # Recompensa positiva si la pelota llega al hoyo
            done = True
        elif self.current_position[0] < 0 or self.current_position[0] > 10 or self.current_position[1] < 0 or self.current_position[1] > 10:
            reward = -1.0  # Recompensa negativa si la pelota sale fuera del campo de juego
            done = True
        elif self.current_step >= self.max_steps:
            reward = 0.0  # Recompensa neutral si se alcanza el número máximo de pasos sin llegar al hoyo
            done = True
        else:
            reward = 0.0  # Recompensa neutral para el resto de los casos
            done = False

        return np.array(self.current_position), reward, done, {}

    def reset(self):
        self.current_position = self.start_position
        self.current_step = 0
        return np.array(self.current_position)

    def _decode_action(self, action):
        dx = (action // 4) - 1
        dy = (action % 4) - 1
        if action // 8 == 0:  # Primer palo
            f = np.random.randint(5, 9)
            i = np.random.randint(-3, 4)
        else:  # Segundo palo
            f = np.random.randint(1, 4)
            i = np.random.randint(-1, 2)
        return dx, dy, f, i