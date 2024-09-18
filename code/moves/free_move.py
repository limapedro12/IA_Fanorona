from moves.move import Move
from moves.motion_move import MotionMove

class FreeMove(MotionMove):
    """
    Move that does not capture any piece.

    The piece only moves 1 tile.    
    """
    def __init__(self, row_origin, col_origin, row_destination, col_destination):
        super().__init__(row_origin, col_origin, row_destination, col_destination)
    
    def __str__(self):
        return super().__str__() + " Free"

    def __eq__(self, other: object):
        if isinstance(other, FreeMove):
            return super().__eq__(other)
        return False

    def allows_multiple_moves(self) -> bool:
        return False

    @Move.execute_decorator
    def execute(self, state):
        state.count += 1
        return super().execute(state)
