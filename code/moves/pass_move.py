from moves.move import Move

class PassMove(Move):
    """
    A move that represents a player passing their turn. Does not modify the board.
    """
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Pass"

    def __eq__(self, other: object):
        return isinstance(other, PassMove)

    def allows_multiple_moves(self) -> bool:
        return False

    @Move.execute_decorator
    def execute(self, state):
        state.count += 1
        return state
