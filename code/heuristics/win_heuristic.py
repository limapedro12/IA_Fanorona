if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heuristics.heuristic import Heuristic
from state import State
from player import Player

class WinHeuristic(Heuristic):
    """
    Heuristic that only values a state for a given player if he has won or lost.
    """

    def evaluate_board(self, state: State, player_to_win: Player):
        if state.check_winner() != Player.EMPTY:
            return 1 if state.check_winner() == player_to_win else -1
        return 0


def test_win_heuristic():
    h = WinHeuristic()
    s = State()

    s.get_board_matrix().fill(Player.EMPTY) # clear the board
    s.get_board_matrix()[0][0] = Player.WHITE
    s.white_pieces_count = 1
    s.black_pieces_count = 0
    print(s.get_board_matrix())

    white_eval = h.evaluate_board(s, Player.WHITE)
    black_eval = h.evaluate_board(s, Player.BLACK)
    print(white_eval, black_eval)
    assert white_eval > 0, f"Expected white won, but got {white_eval}"
    assert black_eval < 0, f"Expected black lost, but got {black_eval}"

if __name__ == "__main__":
    test_win_heuristic()
