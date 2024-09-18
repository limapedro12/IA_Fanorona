if __name__ == '__main__':
    from import_from_parent import import_from_parent
    import_from_parent()

from monte_carlo_tree_search.tree import MonteCarloTree
from monte_carlo_tree_search.node_heuristics import MonteCarloNodeHeuristic
from state import State
from player import Player
import time

from heuristics.heuristic import Heuristic
import numpy as np

class MonteCarloTreeHeuristic(MonteCarloTree):
    """
    Subclass of MonteCarloTree that uses an heuristic to evaluate the nodes, instead of rollouts.
    """
    def __init__(self, heuristic: Heuristic, boardWidth: int, boardHeight: int, cWhite=2, cBlack=2):
        super().__init__(boardWidth, boardHeight, cWhite, cBlack)
        self.root = MonteCarloNodeHeuristic(heuristic, None, State(boardWidth, boardHeight), cWhite, cBlack)
        self.currNode = self.root
        self.heuristic = heuristic
    
    @classmethod
    def from_player(self, heuristic: Heuristic, boardWidth: int, boardHeight: int, player: Player):
        """
        Constructs a MonteCarloTree with the appropriate constants for the player.
        """
        if player == Player.WHITE:
            return MonteCarloTreeHeuristic(heuristic, boardWidth, boardHeight, 2, 10)
        else:
            return MonteCarloTreeHeuristic(heuristic, boardWidth, boardHeight, 10, 2)

    def reset_game(self):
        """
        Resets the game to the initial state.
        """
        self.state = State(self.boardWidth, self.boardHeight)
        self.root = MonteCarloNodeHeuristic(self.heuristic, None, State(self.boardWidth, self.boardHeight), self.cWhite, self.cBlack)
        self.currNode = self.root



