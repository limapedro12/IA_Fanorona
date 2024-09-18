import numpy as np

if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heuristics.heuristic import Heuristic
from player import Player
from state import State
from board import Player


class ApproximateEnemyHeuristic(Heuristic):
    """
    This heuristic evaluates the board by reducing the distance from my pieces to the enemy pieces if I have more pieces than the enemy.
    The heuristic returns the sum of the minimum distances for each of my pieces. Negative if I have fewer pieces than the enemy.

    It should never be used alone, but as part of a list of heuristics. And it should always have a relatively lower weight.
    """

    def evaluate_board(self, state: State, player_to_win: Player):
        adversary = Player.opponent_player(player_to_win)
        my_num_pieces = state.get_num_pieces(player_to_win)
        enemy_num_pieces = state.get_num_pieces(adversary)
        if (
            my_num_pieces == 0
            or enemy_num_pieces == 0
            or my_num_pieces == enemy_num_pieces
        ):
            return 0

        max_distance = state.board.width + state.board.height

        my_pieces = state.board.get_pieces(player_to_win)
        enemy_pieces = state.board.get_pieces(adversary)

        distances = np.linalg.norm(my_pieces[:, None] - enemy_pieces, axis=2)
        min_distances = np.min(distances, axis=1)

        result = np.sum(min_distances) / len(my_pieces)
        if my_num_pieces > enemy_num_pieces:  # more pieces, better to be closer
            return max_distance - result
        return result - max_distance  # less pieces, better to be further


def test_approximate_enemy_heuristic():
    s = State()
    h = ApproximateEnemyHeuristic()

    s.get_board_matrix().fill(Player.EMPTY)  # clear the board

    s.get_board_matrix()[0][0:5] = Player.WHITE
    s.white_pieces_count = 5

    s.get_board_matrix()[3][0:4] = Player.BLACK
    s.black_pieces_count = 4

    print(s.get_board_matrix())
    white_eval = h.evaluate_board(s, Player.WHITE)
    black_eval = h.evaluate_board(s, Player.BLACK)
    print(white_eval, black_eval)
    assert white_eval > 0, f"Expected white advantage, but got {white_eval}"
    assert black_eval < 0, f"Expected black disadvantage, but got {black_eval}"


if __name__ == "__main__":
    test_approximate_enemy_heuristic()
