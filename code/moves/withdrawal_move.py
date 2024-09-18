from moves.move import Move
from moves.motion_move import MotionMove
from player import Player

class WithdrawalMove(MotionMove):
    def __init__(self, row_origin, col_origin, row_destination, col_destination):
        super().__init__(row_origin, col_origin, row_destination, col_destination)

    def __str__(self):
        return super().__str__() + " Withdrawal"

    def __eq__(self, other: object):
        if isinstance(other, WithdrawalMove):
            return super().__eq__(other)
        return False

    def allows_multiple_moves(self) -> bool:
        return True

    def get_first_to_kill(self) -> tuple[int, int]:
        """Returns the first piece that this move will capture."""
        rdir, cdir = self.get_directions()
        return self.row_origin + rdir, self.col_origin + cdir

    def get_directions(self):
        """Returns the direction of the move."""
        return (self.row_origin - self.row_destination, self.col_origin - self.col_destination)

    @Move.execute_decorator
    def execute(self, state):   
        state = super().execute(state)
        rdir, cdir = self.get_directions()

        opponent = Player.opponent_player(state.player)

        row_to_kill = self.row_origin
        col_to_kill = self.col_origin
        while True:
            row_to_kill += rdir
            col_to_kill += cdir
            if state.board.inside_board(row_to_kill, col_to_kill) and state.get_board_matrix()[row_to_kill][col_to_kill] == opponent:
                state.get_board_matrix()[row_to_kill][col_to_kill] = Player.EMPTY   # captures the enemy piece
                state.decrement_pieces_count(opponent)
            else:
                break   # no more enemy pieces in a straight line
        state.count = 0 # just captured a piece, so the count is reset (count since last capture)
        return state
