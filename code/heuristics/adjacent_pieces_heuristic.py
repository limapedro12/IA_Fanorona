if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heuristics.heuristic import Heuristic
from adjacent_positions import ADJACENT_RIGHT
from state import State
from player import Player
import numpy as np



class AdjacentPiecesHeuristic(Heuristic):
    """
    Heuristic that calculates the difference between the maximum number of 
    adjacent pieces of the player and the adversary.

    The more adjacent pieces the adversary has, the better for us.
    This is because we may be able to capture them all in a single move.
    """

    def calculate_max_adjacent_pieces(self, state, player):
        pieces = state.board.get_pieces(player)
        visited = np.zeros(state.get_board_matrix().shape, dtype=bool)

        max_adjacent = 0

        for r, c in ADJACENT_RIGHT:
            adjacent = 0
            for row, col in pieces:
                if visited[row][col]:
                    continue
                adjacent = 0
                current_row = row + r
                current_col = col + c
                while (
                    state.board.inside_board(current_row, current_col)
                    and state.get_board_matrix()[current_row][current_col] == player
                ):
                    visited[current_row][current_col] = True
                    adjacent += 1
                    current_row += r
                    current_col += c
            max_adjacent = max(max_adjacent, adjacent)
        return max_adjacent

    def evaluate_board(self, state, player_to_win):
        adversary = Player.opponent_player(player_to_win)
        max_adj_me = self.calculate_max_adjacent_pieces(state, player_to_win)
        max_adj_adv = self.calculate_max_adjacent_pieces(state, adversary)
        return (
            max_adj_adv - max_adj_me
        )  # the more adjacent pieces the adversary has, the better for us


def test_adjacent_pieces_heuristic():
    s = State(5, 3)
    h = AdjacentPiecesHeuristic()

    s.get_board_matrix().fill(Player.EMPTY)  # clear the board

    s.get_board_matrix()[0][0:3] = Player.BLACK
    s.get_board_matrix()[1][0:2] = Player.WHITE

    print(s.get_board_matrix())
    white_eval = h.evaluate_board(s, Player.WHITE)
    black_eval = h.evaluate_board(s, Player.BLACK)
    print(white_eval, black_eval)
    assert white_eval > 0, f"Expected white advantage, but got {white_eval}"
    assert black_eval < 0, f"Expected black disadvantage, but got {black_eval}"


if __name__ == "__main__":
    test_adjacent_pieces_heuristic()
