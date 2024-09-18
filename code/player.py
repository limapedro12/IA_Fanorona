from enum import Enum


class Player(Enum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2

    def __str__(self):
        match self:
            case Player.EMPTY:
                return "EMPTY"
            case Player.BLACK:
                return "\033[94m" + "BLACK" + "\033[0m"
            case Player.WHITE:
                return "\033[93m" + "WHITE" + "\033[0m"

    def __repr__(self):
        match self:
            case Player.EMPTY:
                return " "
            case Player.BLACK:
                return "B"
            case Player.WHITE:
                return "W"

    @staticmethod
    def opponent_player(player):
        """
        Returns the opponent of the given player.

        Should not be called with Player.EMPTY.
        """
        if player == Player.BLACK:
            return Player.WHITE
        return Player.BLACK
