import random
from dataclasses import dataclass
from typing import Optional

import gym
import numpy as np
import pygame
from gym import spaces

from src.main.python.games.golf.golf_club import GolfClub
from src.main.python.games.golf.position_golf_ball import PositionGolfBall


@dataclass
class GolfEnv(gym.Env):
    width: int = 10
    height: int = 15
    golfs_club = [
        GolfClub(5,8),
        GolfClub(1,3)
    ]
    target_location: np.array = PositionGolfBall(2, height - 1)
    agent_location: np.array = None
    terminated: bool = False
    origin: np.array = np.array([
        PositionGolfBall(0, 9), PositionGolfBall(1, 9), PositionGolfBall(2, 9)
    ])
    window_size: int = 512
    render_mode: str = "human"
    window = None
    clock = None
    obstacles = np.array([
        PositionGolfBall(0, 0), PositionGolfBall(0, 1), PositionGolfBall(0, 2), PositionGolfBall(0, 3), PositionGolfBall(0, 4), PositionGolfBall(0, 5),
        PositionGolfBall(1, 0), PositionGolfBall(1, 1), PositionGolfBall(1, 2), PositionGolfBall(1, 3), PositionGolfBall(1, 4), PositionGolfBall(1, 5),
        PositionGolfBall(2, 0), PositionGolfBall(2, 1), PositionGolfBall(2, 2), PositionGolfBall(2, 3), PositionGolfBall(2, 4), PositionGolfBall(2, 5),
        PositionGolfBall(3, 0), PositionGolfBall(3, 1), PositionGolfBall(3, 2), PositionGolfBall(3, 3), PositionGolfBall(3, 4),
        PositionGolfBall(4, 0), PositionGolfBall(4, 1), PositionGolfBall(4, 2), PositionGolfBall(4, 3),
        PositionGolfBall(5, 0), PositionGolfBall(5, 1), PositionGolfBall(5, 2),
        PositionGolfBall(9, 7), PositionGolfBall(9, 8), PositionGolfBall(9, 9), PositionGolfBall(9, 10), PositionGolfBall(9, 11), PositionGolfBall(9, 12), PositionGolfBall(9, 13), PositionGolfBall(9, 14),
        PositionGolfBall(8, 7), PositionGolfBall(8, 8), PositionGolfBall(8, 9), PositionGolfBall(8, 10), PositionGolfBall(8, 11), PositionGolfBall(8, 12), PositionGolfBall(8, 13), PositionGolfBall(8, 14),
        PositionGolfBall(7, 7), PositionGolfBall(7, 8), PositionGolfBall(7, 9), PositionGolfBall(7, 10), PositionGolfBall(7, 11), PositionGolfBall(7, 12), PositionGolfBall(7, 13), PositionGolfBall(7, 14),
        PositionGolfBall(6, 7), PositionGolfBall(6, 8), PositionGolfBall(6, 9), PositionGolfBall(6, 10), PositionGolfBall(6, 11), PositionGolfBall(6, 12), PositionGolfBall(6, 13),
        PositionGolfBall(5, 7), PositionGolfBall(5, 8), PositionGolfBall(5, 9), PositionGolfBall(5, 10), PositionGolfBall(5, 11), PositionGolfBall(5, 12),
        PositionGolfBall(4, 7), PositionGolfBall(4, 8), PositionGolfBall(4, 9), PositionGolfBall(4, 10), PositionGolfBall(4, 11)
    ])


    def __post_init__(self):
        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
        self.directions = [np.array([dx, dy]) for dx in range(-1, 2) for dy in range(-1, 2) if not (dx == 0 and dy == 0)]
        self.field = np.array([PositionGolfBall(x, y) for x in range(self.width) for y in range(self.height) if PositionGolfBall(x, y) not in self.obstacles])
        print(self.directions)

        assert self.render_mode is None or self.render_mode in self.metadata["render_modes"]

        # Definición del espacio de observación
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, np.array([self.width, self.height]) - 1, dtype=int),
            "target": spaces.Box(0, np.array([self.width, self.height]) - 1, dtype=int),
        })

        # self.observation_space = spaces.Dict({i: 0 for i in self.field})

        # Definición del espacio de acción
        self.action_space = spaces.Discrete(len(self.golfs_club) * len(self.directions))

    def reset(self):
        self.agent_location = self._get_random_location()
        self.terminated = False
        self.truncated = False
        return self._get_observation()

    def step(self, action):
        print(action)

        direction = action % len(self.directions)
        print(direction)
        palo = action // len(self.directions)
        print(palo)

        min_force = self.golfs_club[palo].min_force
        max_force = self.golfs_club[palo].max_force

        force = np.random.randint(min_force, max_force + 1)
        direction_vector = self.directions[direction]

        new_location = self.agent_location.new_position(force, direction_vector)
        print(new_location)
        if new_location == self.target_location:
            self.terminated = True
            reward = 1  # Recompensa por llegar al hoyo
        elif new_location in self.obstacles or new_location not in self.field:
            self.truncated = True
            reward = -1  # Penalización por golpear obstáculo
        else:
            self.agent_location = new_location
            reward = 0

        if self.render_mode == "human":
            self._render_frame()

        return new_location, reward, self.terminated, self.truncated, self._get_info()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _get_observation(self):
        return {
            "agent": self.agent_location,
            "target": self.target_location
        }

    def _get_info(self):
        return {"distance": self.agent_location.calculate_distance(self.target_location)}

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
                pix_square_width * self.target_location.x,
                pix_square_height * self.target_location.y,
                pix_square_width,
                pix_square_height,
            ),
        )

        # Dibujar el agente
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            ((self.agent_location.x + 0.5) * pix_square_width, (self.agent_location.y + 0.5) * pix_square_height),
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
                    pix_square_width * obstacle.x,
                    pix_square_height * obstacle.y,
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

    def _get_random_location(self):
        # TODO: Cambiar para que devuelva una de las tres localizaciones válidas
        return self.origin[np.random.randint(0, len(self.origin))]


