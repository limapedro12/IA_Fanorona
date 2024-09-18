if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heuristics.heuristic import Heuristic
from board import Board
from player import Player
from state import State
from adjacent_positions import ADJACENT_ALL
import numpy as np


class GroupsHeuristic(Heuristic):
    """
    Heuristic that values separating pieces from each other.

    A state where all the opponent's pieces are in a "group" and the player's pieces are all separated is desirable by this heuristic.
    
    A group is defined as a set of adjacent pieces (horizontally, vertically or diagonally)
    """

    def flood_fill(self, board: Board, row, col, player: Player, visited: np.ndarray):
        if (
            not board.inside_board(row, col)
            or visited[row][col]
            or board.board[row][col] != player
        ):
            return
        visited[row][col] = True
        for r, c in ADJACENT_ALL:
            self.flood_fill(board, row + r, col + c, player, visited)

    def count_groups(self, board: Board, player: Player, visited: np.ndarray):
        pieces = board.get_pieces(player)
        groups_count = 0

        for row, col in pieces:
            if not visited[row][col]:
                self.flood_fill(board, row, col, player, visited)
                groups_count += 1
        return groups_count

    def evaluate_board(self, state, player_to_win):
        """
        perform a flood fill algorithm to find the groups of pieces
        count the number of groups of pieces for each player
        return the difference of the number of groups of pieces for each player
        """
        adversary_player = Player.opponent_player(player_to_win)

        visited = np.zeros(state.get_board_matrix().shape, dtype=bool)
        my_groups = self.count_groups(state.board, player_to_win, visited)
        adv_groups = self.count_groups(state.board, adversary_player, visited)

        return my_groups - adv_groups  # better (> 0) if my pieces have more groups


def test_groups_heuristic_1():
    h = GroupsHeuristic()
    s = State()

    s.get_board_matrix().fill(Player.EMPTY)
    s.get_board_matrix()[0, 0:2] = Player.WHITE
    s.get_board_matrix()[1, 0] = Player.WHITE

    s.get_board_matrix()[0, 6] = Player.BLACK

    s.get_board_matrix()[2, 2:4] = Player.BLACK

    s.get_board_matrix()[4, 4] = Player.BLACK
    s.get_board_matrix()[3, 5] = Player.BLACK
    print(s.get_board_matrix())

    white_eval = h.evaluate_board(s, Player.WHITE)
    black_eval = h.evaluate_board(s, Player.BLACK)
    print(white_eval, black_eval)
    assert white_eval < 0, f"Expected white disadvantage, but got {white_eval}"
    assert black_eval > 0, f"Expected black advantage, but got {black_eval}"


def test_groups_heuristic_2():
    h = GroupsHeuristic()
    s = State()

    s.get_board_matrix().fill(Player.EMPTY)
    s.get_board_matrix()[0, 0:5] = Player.WHITE
    s.get_board_matrix()[1, 0:4] = Player.BLACK

    print(s.get_board_matrix())

    white_eval = h.evaluate_board(s, Player.WHITE)
    black_eval = h.evaluate_board(s, Player.BLACK)
    print(white_eval, black_eval)
    assert (
        white_eval == 0
    ), f"Expected no advantage or disadvantage, but got {white_eval}"
    assert (
        black_eval == 0
    ), f"Expected no advantage or disadvantage, but got {black_eval}"


def test_groups_heuristic():
    test_groups_heuristic_1()
    test_groups_heuristic_2()


if __name__ == "__main__":
    test_groups_heuristic()
