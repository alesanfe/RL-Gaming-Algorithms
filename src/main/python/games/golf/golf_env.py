import os
from dataclasses import dataclass

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
        GolfClub(5, 8, -3, 3),
        GolfClub(1, 3, -1, 1),
        GolfClub(4, 9, -2, 2),
        GolfClub(6, 7, 0, 4),
        GolfClub(3, 10, -1, 1)
    ]
    target_location: np.array = PositionGolfBall(2, height - 1)
    agent_location: np.array = None
    terminated: bool = False
    window_size: int = 512
    render_mode: str = None
    window = None
    clock = None

    def __post_init__(self):
        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
        self.directions = [np.array([dx, dy]) for dx in range(-1, 2) for dy in range(-1, 2) if
                           not (dx == 0 and dy == 0)]

        self._read_camp()

        # Comprobación de los parámetros
        assert self.render_mode is None or self.render_mode in self.metadata["render_modes"]

        # Definición del espacio de observación
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, np.array([self.width, self.height]) - 1, dtype=int),
            "target": spaces.Box(0, np.array([self.width, self.height]) - 1, dtype=int),
        })

        # Definición del espacio de acción
        actions = 0
        for golf_club in self.golfs_club:
            actions += len(self.directions) * (golf_club.max_force - golf_club.min_force + 1)
        self.action_space = spaces.Discrete(actions)

    def reset(self):
        self.agent_location = self._get_random_location()
        self.terminated = False
        self.truncated = False
        return self._get_observation()

    def step(self, action):

        club_index, direction_index, force_index = self.take_action(action)

        golf_club = self.golfs_club[club_index]
        direction_vector = self.directions[direction_index]

        if self.render_mode == "human":
            self._render_frame()

        self.agent_location = golf_club.hit(self.agent_location, direction_vector, force_index)

        if self.agent_location == self.target_location:
            self.terminated = True
            reward = 100  # Recompensa por llegar al hoyo
        elif self.agent_location in self.lake or (
                self.agent_location not in self.field and self.agent_location not in self.sands):
            self.truncated = True
            reward = -100  # Penalización por caer en el lago o fuera del campo
        elif self.agent_location in self.sands:
            reward = -10  # Penalización por caer en arena
            golf_club.modify = 0.5  # Modificar la fuerza del golpe
        else:
            reward = -5  # Penalización por caer en el césped

        return self.agent_location, reward, self.terminated, self.truncated, self._get_info()

    def take_action(self, action):
        num_clubs = len(self.golfs_club)
        num_directions = len(self.directions)

        # Obtener el índice de la dirección
        direction_index = action % num_directions

        # Calcular el índice del palo
        club_index = (action // num_directions) % num_clubs

        # Calcular el índice de la fuerza
        force_index = (action // (num_directions * num_clubs))
        return club_index, direction_index, force_index

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _get_observation(self):
        agent_position = self.agent_location
        target_position = self.target_location

        observation = {
            "agent": agent_position,
            "target": target_position,
            "distance_to_target": np.linalg.norm(agent_position - target_position),
            "club_information": []
        }

        for golf_club in self.golfs_club:
            club_info = {
                "min_force": golf_club.min_force,
                "max_force": golf_club.max_force
            }
            observation["club_information"].append(club_info)

        return observation

    def _get_info(self):
        return {"distance": self.agent_location.calculate_distance(self.target_location)}

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            # Verificar si es necesario inicializar la ventana y el reloj en modo humano
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        # Crear el lienzo de dibujo
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_width = self.window_size / self.width
        pix_square_height = self.window_size / self.height

        # Dibujar el césped
        self._draw_grass(canvas, pix_square_height, pix_square_width)

        # Dibujar el hoyo
        self._draw_hole(canvas, pix_square_height, pix_square_width)

        # Dibujar el agua
        self._draw_water(canvas, pix_square_height, pix_square_width)

        # Dibujar la arena
        self._draw_sand(canvas, pix_square_height, pix_square_width)

        # Dibujar los rectángulos
        self._draw_rectangles(canvas, pix_square_height, pix_square_width)

        # Dibujar la bola
        self._draw_ball(canvas, pix_square_height, pix_square_width)

        if self.render_mode == "human":
            # Mostrar un mensaje en medio de la pantalla
            if (self.terminated):
                self._send_message(canvas, "Has llegado al hoyo")
            elif (self.truncated):
                self._send_message(canvas, "No has llegado al hoyo")

            # Actualizar la ventana y controlar la frecuencia de actualización en modo humano
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            # Devolver el lienzo en formato numpy si no es modo humano
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def _send_message(self, canvas, message):
        path = os.path.dirname(os.path.abspath(__file__))
        if "/" in path:
            font_path = os.path.dirname(
                os.path.abspath(__file__.replace("python/games/golf", "resources/fonts/8-bit Arcade In.ttf")))
        else:
            font_path = os.path.dirname(
                os.path.abspath(__file__.replace("python\games\golf", "resources\\fonts\8-bit Arcade In.ttf")))
        font = pygame.font.Font(font_path, 36)
        text = font.render(message, True, (255, 255, 255))
        text_rect = text.get_rect(center=(self.window_size // 2, self.window_size // 2))
        canvas.blit(text, text_rect)

    def _draw_grass(self, canvas, pix_square_height, pix_square_width):
        # Dentro del bucle de dibujo:
        for x in range(self.height + 1):
            for y in range(self.width + 1):
                # Calcular las coordenadas del rectángulo
                rect_x = pix_square_width * y
                rect_y = pix_square_height * x

                # Dibujar el rectángulo
                pygame.draw.rect(canvas, (255, 255, 255),
                                 pygame.Rect(rect_x, rect_y, pix_square_width, pix_square_height))

                # Cargar la imagen y redimensionarla al tamaño del rectángulo
                image_path = self._get_image("grass.jpg")
                resized_image = pygame.transform.scale(image_path, (pix_square_width, pix_square_height))

                # Superponer la imagen en el rectángulo
                canvas.blit(resized_image, (rect_x, rect_y))

    def _draw_hole(self, canvas, pix_square_height, pix_square_width):
        # Redimensionar la imagen al tamaño del cuadrado de píxeles
        target_image = pygame.transform.scale(self._get_image("hole.png"),
                                              (int(pix_square_width), int(pix_square_height)))
        # Dibujar la imagen del objetivo en el lienzo de dibujo
        canvas.blit(target_image,
                    (pix_square_width * self.target_location.x, pix_square_height * self.target_location.y))

    def _draw_water(self, canvas, pix_square_height, pix_square_width):
        water_image = self._get_image("water.jpg")
        for water in self.lake:
            # Obtener las coordenadas de píxel del obstáculo
            water_x = pix_square_width * water.x
            water_y = pix_square_height * water.y

            # Redimensionar la imagen al tamaño del rectángulo
            resized_image = pygame.transform.scale(water_image, (pix_square_width, pix_square_height))

            # Dibujar la imagen redimensionada en el lienzo con la posición adecuada
            canvas.blit(resized_image, (water_x, water_y))

    def _draw_sand(self, canvas, pix_square_height, pix_square_width):
        sand_image = self._get_image("sand.jpg")
        for sand in self.sands:
            # Obtener las coordenadas de píxel del obstáculo
            sand_x = pix_square_width * sand.x
            sand_y = pix_square_height * sand.y

            # Redimensionar la imagen al tamaño del rectángulo
            resized_image = pygame.transform.scale(sand_image, (pix_square_width, pix_square_height))

            # Dibujar la imagen redimensionada en el lienzo con la posición adecuada
            canvas.blit(resized_image, (sand_x, sand_y))

    def _draw_rectangles(self, canvas, pix_square_height, pix_square_width):
        # Dibujar las líneas verticales del campo de golf
        for x in range(self.height + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_height * x),
                (self.window_size, pix_square_height * x),
                width=3,
            )
        # Dibujar las líneas horizontales del campo de golf
        for y in range(self.width + 1):
            pygame.draw.line(
                canvas,
                0,
                (pix_square_width * y, 0),
                (pix_square_width * y, self.window_size),
                width=3,
            )

    def _draw_ball(self, canvas, pix_square_height, pix_square_width):
        agent_image = pygame.transform.scale(self._get_image("golf.png"),
                                             (int(pix_square_width), int(pix_square_height)))
        canvas.blit(agent_image,
                    (pix_square_width * self.agent_location.x, pix_square_height * self.agent_location.y))

    def _get_image(self, name):
        path = os.path.dirname(os.path.abspath(__file__))
        if "/" in path:
            image_path = os.path.dirname(
                os.path.abspath(__file__.replace("python/games/golf", "resources/images/" + name)))
        else:
            image_path = os.path.dirname(
                os.path.abspath(__file__.replace("python\games\golf", "resources\images\\" + name)))
        target_image = pygame.image.load(image_path)
        return target_image

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _get_random_location(self):
        return self.origin[np.random.randint(0, len(self.origin))]

    def _read_camp(self):
        # Obtener la ruta del archivo del campo de golf
        path = os.path.dirname(os.path.abspath(__file__))
        if "/" in path:
            camp_path = os.path.dirname(
                os.path.abspath(__file__.replace("python/games/golf", "resources/golf_camp.txt")))
        else:
            camp_path = os.path.dirname(
                os.path.abspath(__file__.replace("python\games\golf", "resources\golf_camp.txt")))

        # Abrir el archivo en modo lectura
        with open(camp_path, 'r') as f:
            # Leer todas las líneas del archivo y almacenarlas en una lista
            lines = f.readlines()

            # Obtener el número de filas y columnas del campo de golf
            self.height = len(lines) - 1
            self.width = len(lines[0]) - 1

            # Inicializar listas para almacenar las ubicaciones de los obstáculos y el campo
            self.lake = []
            self.sands = []
            self.origin = []
            self.field = []

            # Recorrer cada línea del archivo
            for i in range(self.height):
                line = lines[i]

                # Recorrer cada carácter de la línea
                for j in range(self.width):
                    char = line[j]
                    position = PositionGolfBall(j, i)

                    if char == '#':  # Por si se quiere añadir algún comentario en el archivo
                        break
                    elif char == 'W':  # Obstáculo de agua (lake)
                        self.lake = np.append(self.lake, [position])
                    elif char == 'S':  # Obstáculo de arena (sands)
                        self.sands = np.append(self.sands, [position])
                    elif char == 'O':  # Posición de origen (origin) y campo (field)
                        self.field = np.append(self.field, [position])
                        self.origin = np.append(self.origin, [position])
                    elif char == 'E':  # Posición objetivo (target_location) y campo (field)
                        self.field = np.append(self.field, [position])
                        self.target_location = position
                    elif char == 'G':  # Campo (field)
                        self.field = np.append(self.field, [position])
