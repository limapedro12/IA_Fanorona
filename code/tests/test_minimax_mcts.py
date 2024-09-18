import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np

from tests.test import test
from state import State, DRAW_COUNTER_THRESHOLD
from player import Player
from game import Game
from heuristics.heuristic import Heuristic
from heuristics.nr_pieces_heuristic import NrPiecesHeuristic
from heuristics.adjacent_pieces_heuristic import AdjacentPiecesHeuristic
from heuristics.heuristics_list import HeuristicsList
from heuristics.win_heuristic import WinHeuristic
from heuristics.groups_heuristic import GroupsHeuristic
from heuristics.center_control_heuristic import CenterControlHeuristic
from heuristics.approximate_enemy_heuristic import ApproximateEnemyHeuristic

from minimax import get_minimax_move
from monte_carlo_tree_search.tree import MonteCarloTree
from monte_carlo_tree_search.tree_heuristics import MonteCarloTreeHeuristic


def test_mcts_vs_minimax(
    minimax,
    mcts: MonteCarloTree,
    width: int,
    height: int,
    mcts_limit: int,
    mcts_is_white: bool,
    nr: int,
    log_states=False,
):
    wins = {}
    time_minimax = 0
    time_mcts = 0

    def ai_white(state: State) -> State:
        if mcts_is_white:
            mcts.train_until(mcts_limit)
            move = mcts.get_best_move()
            mcts.update_move(move)
            return state.execute_move(move)
        move = minimax(state)
        mcts.update_move(move)
        return state.execute_move(move)

    def ai_black(state: State) -> State:
        if mcts_is_white:
            move = minimax(state)
            mcts.update_move(move)
            return state.execute_move(move)

        mcts.train_until(mcts_limit)
        move = mcts.get_best_move()
        mcts.update_move(move)
        return state.execute_move(move)

    for _ in range(nr):
        state = State(width, height)
        winner, time_white, time_black = state.run_ais(ai_white, ai_black, log_states)

        wins[winner] = wins.get(winner, 0) + 1
        if mcts_is_white:
            time_mcts += time_white
            time_minimax += time_black
        else:
            time_minimax += time_white
            time_mcts += time_black
        mcts.reset_game()

    return wins, time_minimax / nr, time_mcts / nr


def minimax_vs_mcts_wrapper(minimax, nr: int, mcts_iters: int):
    width, height = 9, 5
    mcts = MonteCarloTree.from_player(width, height, Player.WHITE)

    wins, time_minimax, time_mcts = test_mcts_vs_minimax(
        minimax,
        mcts,
        nr=nr,
        mcts_is_white=True,
        mcts_limit=mcts_iters,
        width=width,
        height=height,
    )
    print("White: mcts, Black: minimax")
    print(f"{wins}, minimax: {time_minimax}, mcts: {time_mcts}")

    mcts = MonteCarloTree.from_player(width, height, Player.BLACK)

    wins, time_minimax, time_mcts = test_mcts_vs_minimax(
        minimax,
        mcts,
        nr=nr,
        mcts_is_white=False,
        mcts_limit=mcts_iters,
        width=width,
        height=height,
    )
    print("White: minimax, Black: mcts")
    print(f"{wins}, minimax: {time_minimax}, mcts: {time_mcts}")


@test
def test_nr_pieces_vs_mcts(nr: int, mcts_iters: int, minimax_depth: int):
    minimax = get_minimax_move(NrPiecesHeuristic().evaluate_board, minimax_depth)
    minimax_vs_mcts_wrapper(minimax, nr, mcts_iters)


@test
def test_nr_pieces_approx_enemy_vs_mcts(nr: int, mcts_iters: int, minimax_depth: int):
    minimax = get_minimax_move(
        HeuristicsList(
            np.array([NrPiecesHeuristic(), ApproximateEnemyHeuristic()]),
            np.array([2, 1]),
        ).evaluate_board,
        minimax_depth,
    )
    minimax_vs_mcts_wrapper(minimax, nr, mcts_iters)


@test
def test_pieces_adjacent_groups_vs_mcts(nr: int, mcts_iters: int, minimax_depth: int):
    minimax = get_minimax_move(
        HeuristicsList(
            np.array(
                [NrPiecesHeuristic(), AdjacentPiecesHeuristic(), GroupsHeuristic()]
            ),
            np.array([10, 2, 1]),
        ).evaluate_board,
        minimax_depth,
    )
    minimax_vs_mcts_wrapper(minimax, nr, mcts_iters)


@test
def test_pieces_groups_center_vs_mcts(nr: int, mcts_iters: int, minimax_depth: int):
    minimax = get_minimax_move(
        HeuristicsList(
            np.array(
                [NrPiecesHeuristic(), GroupsHeuristic(), CenterControlHeuristic()]
            ),
            np.array([10, 1, 1]),
        ).evaluate_board,
        minimax_depth,
    )
    minimax_vs_mcts_wrapper(minimax, nr, mcts_iters)


if __name__ == "__main__":
    nr = 20
    mcts_quick_iters, mcts_better_iters = 100, 1000
    minimax_lower_depth, minimax_higher_depth = 2, 4

    test_nr_pieces_vs_mcts(nr, mcts_quick_iters, minimax_lower_depth)
    test_nr_pieces_vs_mcts(nr, mcts_better_iters, minimax_higher_depth)

    test_nr_pieces_approx_enemy_vs_mcts(nr, mcts_quick_iters, minimax_lower_depth)
    test_nr_pieces_approx_enemy_vs_mcts(nr, mcts_better_iters, minimax_higher_depth)

    test_pieces_adjacent_groups_vs_mcts(nr, mcts_quick_iters, minimax_lower_depth)
    test_pieces_adjacent_groups_vs_mcts(nr, mcts_better_iters, minimax_higher_depth)

    test_pieces_groups_center_vs_mcts(nr, mcts_quick_iters, minimax_lower_depth)
    test_pieces_groups_center_vs_mcts(nr, mcts_better_iters, minimax_higher_depth)
