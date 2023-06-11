import random
from dataclasses import dataclass

from src.main.python.games.golf.position_golf_ball import PositionGolfBall


@dataclass
class GolfClub:
    min_force: int
    max_force: int
    min_precision: int
    max_precision: int
    modify: float = 1

    def hit(self, origin, direction):
        force = random.randint(int(self.min_force*self.modify), int(self.max_force*self.modify))
        precision = random.randint(int(self.min_precision/self.modify), int(self.max_precision/self.modify))

        x = origin.x + direction[0] * (force + precision)
        y = origin.y + direction[1] * (force + precision)
        return PositionGolfBall(x, y)
