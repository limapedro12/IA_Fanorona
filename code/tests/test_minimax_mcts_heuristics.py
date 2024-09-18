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

from minimax import execute_minimax_move, get_minimax_move
from monte_carlo_tree_search.tree import MonteCarloTree
from monte_carlo_tree_search.tree_heuristics import MonteCarloTreeHeuristic
import time

heuristicList = [
   NrPiecesHeuristic(),
   AdjacentPiecesHeuristic(),
   WinHeuristic(),
   GroupsHeuristic(),
   CenterControlHeuristic(),
   ApproximateEnemyHeuristic(),
    HeuristicsList(
        heuristics=np.array([
            WinHeuristic(),
            NrPiecesHeuristic(),
            AdjacentPiecesHeuristic(),
            GroupsHeuristic(),
            CenterControlHeuristic(),
        ]),
        weights=np.array([100000, 50, 25, 10, 5]),
    )
]

def test_mcts_heuristic_minimax(nr, heuristic, boardWidth=9, boardHeight=5):
    mcts = MonteCarloTreeHeuristic.from_player(
            heuristic, boardWidth, boardHeight, Player.WHITE
        )
    minimax = get_minimax_move(heuristic.evaluate_board, 6)
    state = State(boardWidth, boardHeight)

    def ai_white(state: State) -> State:
        mcts.train_until(2000)
        move = mcts.get_best_move()
        mcts.update_move(move)
        return state.execute_move(move)
    def ai_black(state: State) -> State:
        move = minimax(state)
        mcts.update_move(move)
        return state.execute_move(move)
    
    wins = {}
    time1_total = 0
    time2_total = 0
    for _ in range(nr):
        state = State(boardWidth, boardHeight)
        mcts.reset_game()

        print("Starting game")
        winner, time1, time2 = state.run_ais(ai_white, ai_black, False, True)

        wins[winner] = wins.get(winner, 0) + 1
        time1_total += time1
        time2_total += time2

    return (wins, time1_total, time2_total)

@test
def all_test_mcts_heuristic_minimax(nr: int):
    for heuristic in heuristicList:
        print(f"Playing {heuristic}")
        wins, time1, time2 = test_mcts_heuristic_minimax(nr, heuristic)
        print(wins)
        print(f"Time for MCTS: {time1}")
        print(f"Time for MiniMax: {time2}")
    

if __name__ == "__main__":
    all_test_mcts_heuristic_minimax(20)
