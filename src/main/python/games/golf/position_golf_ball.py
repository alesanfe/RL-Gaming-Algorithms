from dataclasses import dataclass


@dataclass
class PositionGolfBall:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def new_position(self, force, direction_vector):
        """Debe devolver la nueva dirección, que será la antigua desplazada"""
        return PositionGolfBall(self.x + force * direction_vector[0], self.y + force * direction_vector[1])