if __name__ == '__main__':
    from import_from_parent import import_from_parent
    import_from_parent()

from monte_carlo_tree_search.node import MonteCarloNode
from state import State
from player import Player
from moves.move import Move
import pickle
import time
import random

class MonteCarloTree:
    """
    Implementation of the traditional Monte Carlo Tree Search algorithm.
    """
    def __init__(self, boardWidth: int, boardHeight: int, cWhite=2, cBlack=2):
        self.boardWidth = boardWidth
        self.boardHeight = boardHeight
        self.state = State(boardWidth, boardHeight)
        self.root = MonteCarloNode(None, State(boardWidth, boardHeight), cWhite, cBlack)
        self.currNode = self.root
        self.cWhite = cWhite
        self.cBlack = cBlack

    @classmethod
    def from_player(self, boardWidth: int, boardHeight: int, player: Player):
        """
        Constructs a MonteCarloTree with the appropriate constants for the player.
        """
        if player == Player.WHITE:
            return MonteCarloTree(boardWidth, boardHeight, 2, 10)
        else:
            return MonteCarloTree(boardWidth, boardHeight, 10, 2)
    
    def save_to_disk(self, path: str):
        """
        Saves the tree to a file.
        """
        with open(path, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_from_disk(path: str):
        """
        Loads a tree from a file.
        """
        with open(path,  "rb") as f:
            return pickle.load(f)

    def print_tree(self):
        """
        Prints the tree to the console.
        """
        self.print_tree_aux(self.root, 0)
    
    def print_tree_aux(self, node: MonteCarloNode, depth: int):
        """
        Auxiliary function to print the tree.
        """
        print(" " * 4*depth, "t:", node.total, "v:", node.visits)
        for child, move in node.children:
            print(" " * depth)
            self.print_tree_aux(child, depth + 1)

    def train(self, iterations: int):
        """
        Exectutes a number of rollouts on the tree.
        """
        for _ in range(iterations):
            self.currNode.one_training_iteration()

    def train_time(self, timeout: float):
        """
        Executes rollouts until a certain amount of time has passed.
        Timout is expressed in seconds.
        """
        start = time.time()
        while time.time() - start < timeout or not all([child.visits != 0 for child, _ in self.currNode.children]):
            self.currNode.one_training_iteration()

    def train_until(self, total_iterations: int):
        """
        Executes rollouts until the root node has been visited a certain amount of times.
        """
        while self.currNode.visits < total_iterations:
            self.currNode.one_training_iteration()
        
    def get_best_move(self):
        """
        Based on the current tree, returns the best move.
        """
        if not self.currNode.expanded:
            print("Warning: Tree not trained enough to get best move. Returning random move.")
            return random.choice(self.state.get_available_moves())
        
        if(self.currNode.children[0][0].player == self.currNode.player):
            return max(self.currNode.children, key=lambda x: x[0].total/x[0].visits if x[0].visits != 0 else float("-inf"))[1]
        else:
            return min(self.currNode.children, key=lambda x: x[0].total/x[0].visits if x[0].visits != 0 else float("inf"))[1]
    
    def update_move(self, move_to_exe: Move):
        """
        Updates the tree with the move that was executed, changing the node to the
        child that corresponds to the move, and deleting the parent and siblings
        of selected node.
        """
        if not self.currNode.expanded:
            self.currNode.expand()
            
        for child, move in self.currNode.children:
            if move == move_to_exe:
                self.currNode = child
                self.currNode.parentNode = None
                self.state = self.state.execute_move(move_to_exe)
                return
            
        print("Player inside mcts: ", self.state.player)
        print("Children: ", self.currNode.children)
        print("State inside mcts:")
        raise Exception("Move not found: ", move_to_exe, " in children: ", self.currNode.children)
    
    def reset_game(self):
        """
        Resets the the tree to the initial state.
        """
        self.state = State(self.boardWidth, self.boardHeight)
        self.root = MonteCarloNode(None, State(self.boardWidth, self.boardHeight), self.cWhite, self.cBlack)
        self.currNode = self.root
    
