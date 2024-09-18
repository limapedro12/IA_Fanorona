if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from heuristics.heuristic import Heuristic
from state import State
from board import Player


class CenterControlHeuristic(Heuristic):
    """
    Heuristic that values proximity to the center.
    
    A state where the player's pieces are closer to the center and the adversary's are farther is desirable by this heuristic.
    """

    def evaluate_board(self, state: State, player_to_win: Player):
        rowMiddle = state.get_board_matrix().shape[0] // 2
        colMiddle = state.get_board_matrix().shape[1] // 2
        enemy = Player.opponent_player(player_to_win)

        player_pieces = state.board.get_pieces(player_to_win)        
        enemy_pieces = state.board.get_pieces(enemy)

        my_distances = np.abs(player_pieces[:,0] - rowMiddle) + np.abs(player_pieces[:,1] - colMiddle)
        enemy_distances = np.abs(enemy_pieces[:,0] - rowMiddle) + np.abs(enemy_pieces[:,1] - colMiddle)

        if len(enemy_pieces) == 0 or len(player_pieces) == 0:
            return 0

        # better if my pieces are closer to the center
        return np.sum(enemy_distances) / len(enemy_pieces) - np.sum(my_distances) / len(player_pieces)

def test_center_control_heuristic():
    h = CenterControlHeuristic()
    s = State()

    s.get_board_matrix().fill(Player.BLACK) # fill with black pieces

    row_center = s.get_board_matrix().shape[0] // 2
    col_center = s.get_board_matrix().shape[1] // 2
    s.get_board_matrix()[row_center][col_center] = Player.WHITE
    print(s.get_board_matrix())

    white_eval = h.evaluate_board(s, Player.WHITE)
    black_eval = h.evaluate_board(s, Player.BLACK)
    print(white_eval, black_eval)
    assert white_eval > 0, f"Expected white advantage, but got {white_eval}"
    assert black_eval < 0, f"Expected black disadvantage, but got {black_eval}"

def test_center_control_heuristic_few_pieces():
    h = CenterControlHeuristic()
    s = State()

    s.get_board_matrix().fill(Player.EMPTY) # fill with white pieces

    row_center = s.get_board_matrix().shape[0] // 2
    col_center = s.get_board_matrix().shape[1] // 2
    s.get_board_matrix()[row_center][col_center] = Player.WHITE
    s.get_board_matrix()[0][0] = Player.BLACK
    print(s.get_board_matrix())

    white_eval = h.evaluate_board(s, Player.WHITE)
    black_eval = h.evaluate_board(s, Player.BLACK)
    print(white_eval, black_eval)
    assert white_eval > 0, f"Expected white advantage, but got {white_eval}"
    assert black_eval < 0, f"Expected black disadvantage, but got {black_eval}"

if __name__ == "__main__":
    test_center_control_heuristic()
    test_center_control_heuristic_few_pieces()
