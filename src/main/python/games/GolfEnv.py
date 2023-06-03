import random

import gym
from gym import spaces
import numpy as np
import pygame
from pygame.surfarray import make_surface


import random
import pygame
from pygame.surfarray import make_surface
from dataclasses import dataclass
from typing import Dict, List, DefaultDict
import gym
from gym import spaces
import numpy as np

@dataclass
class GolfEnv(gym.Env):
    width: int = 10
    height: int = 15
    palo1_min_force: int = 5
    palo1_max_force: int = 8
    palo2_min_force: int = 1
    palo2_max_force: int = 3
    field: np.array = np.zeros((width, height), dtype=int)
    target_location: np.array = np.array([2, height - 1])
    agent_location: np.array = None
    terminated: bool = False
    origin: np.array = np.array([width, 0, 1, 2]).reshape(-1, 2)
    window_size: int = 512
    render_mode: str = "human"
    window = None
    clock = None
    obstacles = np.array([
        [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
        [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5],
        [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5],
        [3, 0], [3, 1], [3, 2], [3, 3], [3, 4],
        [4, 0], [4, 1], [4, 2], [4, 3],
        [5, 0], [5, 1], [5, 2],
        [9, 7], [9, 8], [9, 9], [9, 10], [9, 11], [9, 12], [9, 13], [9, 14],
        [8, 7], [8, 8], [8, 9], [8, 10], [8, 11], [8, 12], [8, 13], [8, 14],
        [7, 7], [7, 8], [7, 9], [7, 10], [7, 11], [7, 12], [7, 13], [7, 14],
        [6, 7], [6, 8], [6, 9], [6, 10], [6, 11], [6, 12], [6, 13],
        [5, 7], [5, 8], [5, 9], [5, 10], [5, 11], [5, 12],
        [4, 7], [4, 8], [4, 9], [4, 10], [4, 11],
    ])

    def __post_init__(self):
        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
        self.directions = [np.array([dx, dy]) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 and dy != 0]

        assert self.render_mode is None or self.render_mode in self.metadata["render_modes"]

        # Definición del espacio de observación
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, np.array([self.width, self.height]) - 1, dtype=int),
            "target": spaces.Box(0, np.array([self.width, self.height]) - 1, dtype=int),
        })

        # Definición del espacio de acción
        self.action_space = spaces.Tuple((
            spaces.Discrete(2), # Selección del palo (0 o 1)
            spaces.Discrete(len(self.directions)) # Selección de la dirección (0 a 7)
        ))

    def reset(self):
        self.field = np.zeros((self.size, self.size), dtype=int)
        self.agent_location = self._get_random_location()
        self.terminated = False
        return self._get_observation()

    def step(self, action):
        palo, direction = action

        if palo == 0:
            min_force, max_force = self.palo1_min_force, self.palo1_max_force
        else:
            min_force, max_force = self.palo2_min_force, self.palo2_max_force

        force = np.random.randint(min_force, max_force + 1)
        direction_vector = self.directions[direction]

        new_location = self.agent_location + force * direction_vector
        new_location = self._clip_location(new_location)

        if np.array_equal(new_location, self.target_location):
            self.terminated = True
            reward = 1  # Recompensa por llegar al hoyo
        elif self.field[tuple(new_location)] != 0 or (
                self.obstacles[tuple(new_location)] != 0 and self.field[tuple(new_location)] == 0):
            self.terminated = True
            reward = -1  # Penalización por golpear obstáculo
        else:
            self.field[tuple(new_location)] = 1
            reward = 0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_observation(), reward, self.terminated, self._get_info()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _get_observation(self):
        return {
            "agent": self.agent_location,
            "target": self.target_location
        }

    def _get_info(self):
        return {"distance": np.linalg.norm(self.agent_location - self.target_location, ord=1)}

    def _get_random_location(self):
        return self.origin[random.randint(0, 2)]

    def _clip_location(self, location):
        return np.clip(location, 0, self.size - 1)

    def _render_frame(self):
        # Verificar si es necesario inicializar la ventana y el reloj en modo humano
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        # Crear el lienzo de dibujo
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_width = self.window_size / self.width
        pix_square_height = self.window_size / self.height

        # Dibujar el objetivo
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_width * self.target_location[0],
                pix_square_height * self.target_location[1],
                pix_square_width,
                pix_square_height,
            ),
        )

        # Dibujar el agente
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            ((self.agent_location[0] + 0.5) * pix_square_width, (self.agent_location[1] + 0.5) * pix_square_height),
            min(pix_square_width, pix_square_height) / 3,
        )

        # Dibujar las líneas verticales del campo de golf
        for x in range(self.width + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_height * x),
                (self.window_size, pix_square_height * x),
                width=3,
            )

        # Dibujar las líneas horizontales del campo de golf
        for y in range(self.height + 1):
            pygame.draw.line(
                canvas,
                0,
                (pix_square_width * y, 0),
                (pix_square_width * y, self.window_size),
                width=3,
            )

        # Dibujar los obstáculos
        obstacle_color = (0, 0, 0)
        for obstacle in self.obstacles:
            pygame.draw.rect(
                canvas,
                obstacle_color,
                pygame.Rect(
                    pix_square_width * obstacle[0],
                    pix_square_height * obstacle[1],
                    pix_square_width,
                    pix_square_height,
                ),
            )

        # Actualizar la ventana y controlar la frecuencia de actualización en modo humano
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            # Devolver el lienzo en formato numpy si no es modo humano
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()