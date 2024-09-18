from abc import ABC, abstractmethod

class Heuristic(ABC):
    @abstractmethod
    def evaluate_board(self, state, player_to_win) -> float:
        """
        Evaluates the state for the given player.
        """
        pass
