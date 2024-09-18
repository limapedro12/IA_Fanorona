from monte_carlo_tree_search.node import MonteCarloNode
from player import Player
from heuristics.heuristic import Heuristic
from state import State
from copy import deepcopy

import random
import numpy as np

class MonteCarloNodeHeuristic(MonteCarloNode):
    """
    Subclass of MonteCarloNode that uses an heuristic to evaluate the nodes, instead of rollouts.
    """
    def __init__(self, heuristic: Heuristic, parentNode, state: State, cWhite: int, cBlack: int):
        super().__init__(parentNode, state, cWhite, cBlack)
        self.heuristic = heuristic

    def expand(self):
        """
        Expands the node, creating children for all possible moves.
        """
        self.expanded = True
        moves = self.state.get_available_moves()
        self.children = np.empty(len(moves), dtype=object)

        for i in range(len(moves)):
            move = moves[i]
            new_state = self.state.execute_move(move)
            self.children[i] = (MonteCarloNodeHeuristic(self.heuristic, self, new_state, self.cWhite, self.cBlack), move)

        return random.choice(self.children)[0]

    def evaluate(self):
        """
        Evaluates the board using the heuristic.
        """
        valWhite = self.heuristic.evaluate_board(self.state, Player.WHITE)
        valBlack = self.heuristic.evaluate_board(self.state, Player.BLACK)
        return (valWhite, valBlack)
    
    def backpropagate(self, valWhite: int, valBlack: int):
        """
        Backpropagates the result of the heuristic evaluation up the tree.
        """
        self.visits += 1
        if self.player == Player.WHITE:
            self.total += valWhite - valBlack
        elif self.player == Player.BLACK:
            self.total += valBlack - valWhite
        
        if self.parentNode is not None:
            self.parentNode.backpropagate(valWhite, valBlack)

    def one_training_iteration(self):
        """
        Executes one iteration of the Monte Carlo Tree Search algorithm, including
        selection, expansion, evaluation and backpropagation.
        """
        node = self

        while node.children.size != 0 and not node.game_finished:
            node, move = node.select_child()
        
        if not node.expanded and not node.game_finished:
            new_node = node.expand()
            node.delete_state()
            valWhite, valBlack = new_node.evaluate()
            new_node.backpropagate(valWhite, valBlack)
        else:
            valWhite, valBlack = node.evaluate()
            node.backpropagate(valWhite, valBlack)
