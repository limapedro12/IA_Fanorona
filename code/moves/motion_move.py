from moves.move import Move
from player import Player

class MotionMove(Move):
    """
    Move that leads to movement of a piece.
    """
    def __init__(self, row_origin, col_origin, row_destination, col_destination):
        super().__init__()
        self.row_origin = row_origin
        self.col_origin = col_origin
        self.row_destination = row_destination
        self.col_destination = col_destination

    def __str__(self):
        return f"({self.row_origin}, {self.col_origin}) -> ({self.row_destination}, {self.col_destination})"

    def __eq__(self, other):
        if isinstance(other, MotionMove):
            return (self.row_origin == other.row_origin and
                    self.col_origin == other.col_origin and
                    self.row_destination == other.row_destination and
                    self.col_destination == other.col_destination)
        return False

    def get_destination(self):
        return (self.row_destination, self.col_destination)

    def get_origin(self):
        return (self.row_origin, self.col_origin)

    def execute_decorator(func):
        return super().execute_decorator(func)

    def execute(self, state):
        state.get_board_matrix()[self.row_destination][self.col_destination] = state.get_board_matrix()[self.row_origin][self.col_origin]
        state.get_board_matrix()[self.row_origin][self.col_origin] = Player.EMPTY
        return state
