import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
from copy import deepcopy

from tests.test import test
from state import State
from player import Player
from heuristics.nr_pieces_heuristic import NrPiecesHeuristic
from heuristics.adjacent_pieces_heuristic import AdjacentPiecesHeuristic
from heuristics.heuristics_list import HeuristicsList
from heuristics.win_heuristic import WinHeuristic
from heuristics.groups_heuristic import GroupsHeuristic
from heuristics.center_control_heuristic import CenterControlHeuristic

from monte_carlo_tree_search.tree import MonteCarloTree
from monte_carlo_tree_search.tree_heuristics import MonteCarloTreeHeuristic

def play_one_game(state: State, mcts1: MonteCarloTree, no_rollouts1, mcts2: MonteCarloTree, no_rollouts2):
    state = deepcopy(state)
    def ai_white(state: State) -> State:
        mcts1.train_until(no_rollouts1)
        move_to_exe = mcts1.get_best_move()
        if(move_to_exe not in state.get_available_moves()):
            raise Exception("Invalid move: ", move_to_exe, " in ", state.get_available_moves())
        mcts1.update_move(move_to_exe)
        mcts2.update_move(move_to_exe)
        return state.execute_move(move_to_exe)
    
    def ai_black(state: State) -> State:
        mcts2.train_until(no_rollouts2)
        move_to_exe = mcts2.get_best_move()
        if(move_to_exe not in state.get_available_moves()):
            raise Exception("Invalid move: ", move_to_exe, " in ", state.get_available_moves())
        mcts1.update_move(move_to_exe)
        mcts2.update_move(move_to_exe)
        return state.execute_move(move_to_exe)
    
    print("Starting game")
    
    return state.run_ais(ai_white, ai_black)

def play_one_game_random(state: State, mcts: MonteCarloTree, no_rollouts):
    state = deepcopy(state)
    def ai_white(state: State) -> State:
        mcts.train_until(no_rollouts)
        move_to_exe = mcts.get_best_move()
        if(move_to_exe not in state.get_available_moves()):
            raise Exception("Invalid move: ", move_to_exe, " in ", state.get_available_moves())
        mcts.update_move(move_to_exe)
        return state.execute_move(move_to_exe)
    
    def ai_black(state: State) -> State:
        move_to_exe = np.random.choice(state.get_available_moves())
        if(move_to_exe not in state.get_available_moves()):
            raise Exception("Invalid move: ", move_to_exe, " in ", state.get_available_moves())
        mcts.update_move(move_to_exe)
        return state.execute_move(move_to_exe)
    
    print("Starting game")
    
    return state.run_ais(ai_white, ai_black)

def play_n_games(boardWidth, boardHeight, mcts1: MonteCarloTree, no_rollouts1, mcts2: MonteCarloTree, no_rollouts2, nr: int):
    wins = {}
    time1_total = 0
    time2_total = 0
    for _ in range(nr):
        state = State(boardWidth, boardHeight)
        mcts1_copy = deepcopy(mcts1)
        mcts2_copy = deepcopy(mcts2)

        winner, time1, time2 = play_one_game(state, mcts1_copy, no_rollouts1, mcts2_copy, no_rollouts2)

        wins[winner] = wins.get(winner, 0) + 1
        time1_total += time1
        time2_total += time2

    return (wins, time1_total, time2_total)


@test
def test_quick_vs_better(nr: int):
    print("Quick vs Better")
    boardWidth = 9
    boardHeight = 5
    mcts1 = MonteCarloTree.from_player(boardWidth, boardHeight, Player.WHITE)
    mcts2 = MonteCarloTree.from_player(boardWidth, boardHeight, Player.BLACK)
    wins, time1, time2 = play_n_games(boardWidth, boardHeight, mcts1, 100, mcts2, 1000, nr)
    print(wins)
    print(f"Time for quick: {time1}")
    print(f"Time for better: {time2}")

@test
def test_quick_vs_heuristic(nr: int):
    print("Quick vs Heuristic")
    boardWidth = 9
    boardHeight = 5
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
    mcts1 = MonteCarloTree.from_player(boardWidth, boardHeight, Player.BLACK)
    mcts2 = MonteCarloTreeHeuristic.from_player(h, boardWidth, boardHeight, Player.WHITE)
    wins, time1, time2 = play_n_games(boardWidth, boardHeight, mcts1, 100, mcts2, 2000, nr)
    print(wins)
    print(f"Time for quick: {time1}")
    print(f"Time for better: {time2}")

@test
def test_better_vs_heuristic(nr: int):
    print("Better vs Heuristic")
    boardWidth = 9
    boardHeight = 5
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
    mcts1 = MonteCarloTree.from_player(boardWidth, boardHeight, Player.BLACK)
    mcts2 = MonteCarloTreeHeuristic.from_player(h, boardWidth, boardHeight, Player.WHITE)
    wins, time1, time2 = play_n_games(boardWidth, boardHeight, mcts1, 1000, mcts2, 2000, nr)
    print(wins)
    print(f"Time for quick: {time1}")
    print(f"Time for better: {time2}")

def test_quick_vs_random(nr: int):
    print("MCTS Quick vs Random")
    boardWidth = 9
    boardHeight = 5
    mcts = MonteCarloTree.from_player(boardWidth, boardHeight, Player.WHITE)
    wins = {}
    time1_total = 0
    time2_total = 0
    for _ in range(nr):
        state = State(boardWidth, boardHeight)
        mcts_copy = deepcopy(mcts)

        winner, time1, time2 = play_one_game_random(state, mcts_copy, 100)

        wins[winner] = wins.get(winner, 0) + 1
        time1_total += time1
        time2_total += time2

    print(wins)
    print(f"Time for quick: {time1_total}")
    print(f"Time for random: {time2_total}")

def test_better_vs_random(nr: int):
    print("MCTS Better vs Random")
    boardWidth = 9
    boardHeight = 5
    mcts = MonteCarloTree.from_player(boardWidth, boardHeight, Player.WHITE)
    wins = {}
    time1_total = 0
    time2_total = 0
    for _ in range(nr):
        state = State(boardWidth, boardHeight)
        mcts_copy = deepcopy(mcts)

        winner, time1, time2 = play_one_game_random(state, mcts_copy, 1000)

        wins[winner] = wins.get(winner, 0) + 1
        time1_total += time1
        time2_total += time2

    print(wins)
    print(f"Time for better: {time1_total}")
    print(f"Time for random: {time2_total}")

def test_heuristic_vs_random(nr: int):
    print("MCTS Heuristic vs Random")
    boardWidth = 9
    boardHeight = 5
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
    mcts = MonteCarloTreeHeuristic.from_player(h, boardWidth, boardHeight, Player.WHITE)
    wins = {}
    time1_total = 0
    time2_total = 0
    for _ in range(nr):
        state = State(boardWidth, boardHeight)
        mcts_copy = deepcopy(mcts)

        winner, time1, time2 = play_one_game_random(state, mcts_copy, 2000)

        wins[winner] = wins.get(winner, 0) + 1
        time1_total += time1
        time2_total += time2
    
    print(wins)
    print(f"Time for better: {time1_total}")
    print(f"Time for random: {time2_total}")


if __name__ == "__main__":
    nr = 20  
    test_quick_vs_heuristic(nr)
    test_better_vs_heuristic(nr)
    test_heuristic_vs_random(nr)
    

