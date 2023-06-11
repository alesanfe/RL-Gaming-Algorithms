from dataclasses import dataclass


@dataclass
class PositionGolfBall:
    x: float
    y: float

    def calculate_distance(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def rotate(self):
        return PositionGolfBall(self.y, self.x)

    def __sub__(self, other):
        if isinstance(other, PositionGolfBall):
            return PositionGolfBall(self.x - other.x, self.y - other.y)
        else:
            raise TypeError(
                "Unsupported operand type(s) for -: 'PositionGolfBall' and '{}'".format(type(other).__name__))

    def __getitem__(self, item):
        if item == 0:
            return self.x
        elif item == 1:
            return self.y
        else:
            raise IndexError("Invalid index '{}' for PositionGolfBall".format(item))

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        if isinstance(other, PositionGolfBall):
            return self.x == other.x and self.y == other.y
        return False
