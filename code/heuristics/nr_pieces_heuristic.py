if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heuristics.heuristic import Heuristic
from player import Player
from state import State
from board import Player

class NrPiecesHeuristic(Heuristic):
    """
    Heuristic that values the amount of pieces each player has.

    This heuristic gives an higher evaluation for the player with more pieces.
    """

    def evaluate_board(self, state: State, player_to_win: Player) -> float:
        adversary = Player.opponent_player(player_to_win)
        my_num_pieces = state.get_num_pieces(player_to_win)
        enemy_num_pieces = state.get_num_pieces(adversary)

        max_num_pieces = max(my_num_pieces, enemy_num_pieces)
        min_num_pieces = min(my_num_pieces, enemy_num_pieces)

        if min_num_pieces == 0:
            return (my_num_pieces - enemy_num_pieces) * 1000000

        return (my_num_pieces - enemy_num_pieces) * max_num_pieces / min_num_pieces

def test_nr_pieces_heuristic():
    s = State()
    h = NrPiecesHeuristic()

    s.get_board_matrix().fill(Player.EMPTY) # clear the board

    s.get_board_matrix()[0][0:5] = Player.WHITE
    s.get_board_matrix()[1][0:4] = Player.BLACK
    s.white_pieces_count = 5
    s.black_pieces_count = 4

    print(s.get_board_matrix())
    white_eval = h.evaluate_board(s, Player.WHITE)
    black_eval = h.evaluate_board(s, Player.BLACK)
    print(white_eval, black_eval)
    assert white_eval > 0, f"Expected white advantage, but got {white_eval}"
    assert black_eval < 0, f"Expected black disadvantage, but got {black_eval}"

if __name__ == "__main__":
    test_nr_pieces_heuristic()
