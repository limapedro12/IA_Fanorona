if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state import State, Player
from player import Player
import numpy as np

from heuristics.heuristic import Heuristic
from heuristics.groups_heuristic import GroupsHeuristic, test_groups_heuristic
from heuristics.center_control_heuristic import CenterControlHeuristic, test_center_control_heuristic
from heuristics.win_heuristic import WinHeuristic, test_win_heuristic
from heuristics.adjacent_pieces_heuristic import AdjacentPiecesHeuristic, test_adjacent_pieces_heuristic
from heuristics.nr_pieces_heuristic import NrPiecesHeuristic, test_nr_pieces_heuristic
from heuristics.approximate_enemy_heuristic import ApproximateEnemyHeuristic, test_approximate_enemy_heuristic

class HeuristicsList(Heuristic):
    """
    A list of heuristics with given weights.
    
    It evaluates the state with the given heuristics, multiplying the value from each one by the corresponding weights.
    """

    def __init__(self, heuristics, weights):
        self.heuristics : list[Heuristic]   = heuristics
        self.weights    : list[int]         = weights

    def evaluate_board(self, state: State, player_to_win: Player) -> float:
        map_func = lambda h, w: h.evaluate_board(state, player_to_win) * w
        map_list = np.vectorize(map_func)(self.heuristics, self.weights)
        return sum(map_list)


def test_heuristic_list():
    s = State()
    h = HeuristicsList(
        heuristics=np.array([
            WinHeuristic(),
            NrPiecesHeuristic(),
            AdjacentPiecesHeuristic(),
            GroupsHeuristic(),
            CenterControlHeuristic(),
        ]),
        weights=np.array([100000, 50, 25, 10, 5]),
    )

    s.get_board_matrix().fill(Player.EMPTY)  # clear the board

    s.get_board_matrix()[0,0:5] = Player.WHITE
    s.get_board_matrix()[1,0:4] = Player.BLACK
    s.white_pieces_count = 5
    s.black_pieces_count = 4

    print(s.get_board_matrix())
    white_eval = h.evaluate_board(s, Player.WHITE)
    black_eval = h.evaluate_board(s, Player.BLACK)
    print(white_eval, black_eval)
    assert white_eval > 0, f"Expected white advantage, but got {white_eval}"
    assert black_eval < 0, f"Expected black disadvantage, but got {black_eval}"

def test_pieces_approx_enemy():
    s = State()
    h = HeuristicsList(
        heuristics=np.array([
            NrPiecesHeuristic(),
            ApproximateEnemyHeuristic(),
        ]),
        weights=np.array([10, 2]),
    )
    s.get_board_matrix().fill(Player.EMPTY)  # clear the board
    s.get_board_matrix()[0:5, 3] = Player.WHITE
    s.get_board_matrix()[1, 4] = Player.WHITE
    s.white_pieces_count = 6
    
    s.get_board_matrix()[2, 6] = Player.BLACK
    s.get_board_matrix()[1, 7] = Player.BLACK
    s.black_pieces_count = 2

    print(s.get_board_matrix())
    white_eval = h.evaluate_board(s, Player.WHITE)
    black_eval = h.evaluate_board(s, Player.BLACK)
    print(white_eval, black_eval)

    assert white_eval > 0, f"Expected white advantage, but got {white_eval}"
    assert black_eval < 0, f"Expected white advantage, but got {black_eval}"

if __name__ == "__main__":
    test_win_heuristic()
    test_nr_pieces_heuristic()
    test_adjacent_pieces_heuristic()
    test_groups_heuristic()
    test_center_control_heuristic()
    test_approximate_enemy_heuristic()
    test_heuristic_list()
    test_pieces_approx_enemy()
