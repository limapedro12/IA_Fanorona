from typing import Callable
import time

from board import Board
from player import Player
from moves.move import Move
from moves.pass_move import PassMove

DRAW_COUNTER_THRESHOLD = 20

class State:
    """
    State representation of the game.
    
    The state is mainly composed of a player, the board and the move log.
    """

    def __init__(self, width: int = 9, height: int = 5):
        self.board = Board(width, height)
        self.player = Player.WHITE
        self.move_log = []
        self.count = 0                                  # count of moves without captures (used to draw games in simulations)
        self.white_pieces_count = self.board.num_pieces # number of white pieces (for efficiency)
        self.black_pieces_count = self.board.num_pieces # number of black pieces (for efficiency)

    def decrement_pieces_count(self, player: Player):
        """Decrements the count of the pieces of the given player."""
        if player == Player.WHITE:
            self.white_pieces_count -= 1
        else:
            self.black_pieces_count -= 1
    def get_num_pieces(self, player: Player):
        """Gets the count of the pieces of the given player"""
        if player == Player.WHITE:
            return self.white_pieces_count
        return self.black_pieces_count

    def get_board_matrix(self):
        """Returns the matrix corresponding to the board with all of the pieces."""
        return self.board.board

    def change_player(self):
        """Changes the current player to the opponent player."""
        self.player = Player.opponent_player(self.player)

    def finish_turn(self):
        """Finishes the turn of the current player."""
        self.change_player()
        self.move_log.clear()

    def add_to_log(self, move):
        """Adds a move to the move log."""
        self.move_log.append(move)

    def is_white_turn(self):
        """Returns whether it is the white player's turn."""
        return self.player == Player.WHITE

    def in_move_log(self, move: Move):
        """
        Checks if a move's destination is common with any of the positions of moves in the move_log.

        This function should be used to check if consecutive moves (only captures) result in repeating positions.
        """
        if self.move_log == []:
            return False
        if not move.allows_multiple_moves(): # no need to check the log (does not allow multiple moves)
            return False

        # Do not allow movement to tiles from moves in the log
        return move.get_destination() == self.move_log[0].get_origin() or any(
            map(lambda x: x.get_destination() == move.get_destination(), self.move_log)
        )

    def get_available_moves(self):
        """
        Returns the available moves for the current state (for the current player in the current board).

        Depending on the move log, will return normal moves or consecutive moves (only captures and pass).
        """
        if self.move_log == []:
            return self.board.get_all_moves(self.player)

        all_tile_moves = self.board.get_tile_moves(self.move_log[-1].row_destination, self.move_log[-1].col_destination)
        result = list(filter(lambda x: not self.in_move_log(x), all_tile_moves))
        if result != []:    # if there are available consecutive moves (captures), then allow the player to pass
            result.append(PassMove())
        return result

    def execute_move(self, move: Move | None) -> "State":
        """
        Executes a move and returns the new state.

        Finishes the player's turn if the move does not allow multiple moves or if there are no more available moves.
        """
        if move is None:
            self.finish_turn()
            return self

        nstate: State = move.execute(self)
        nstate.add_to_log(move)
        # if no possible next move (no more captures), change player
        if not move.allows_multiple_moves() or nstate.get_available_moves() == []:
            nstate.finish_turn()
        return nstate

    def check_winner(self) -> Player:
        """
        Returns the Player that won the game. A player has won if the adversary has 0 pieces.
        If the game is not over, returns Player.EMPTY.
        """
        if self.get_num_pieces(Player.WHITE) == 0:
            return Player.BLACK
        if self.get_num_pieces(Player.BLACK) == 0:
            return Player.WHITE
        return Player.EMPTY

    def game_over(self) -> bool:
        """
        Checks if the game is over.

        Considers that after DRAW_COUNTER_THRESHOLD moves without captures, the game is a draw.
        """

        return self.check_winner() != Player.EMPTY or self.count >= DRAW_COUNTER_THRESHOLD

    def draw(self) -> None:
        """Prints in text format the current state of the game."""

        print("Next Player: ", self.player)
        self.board.draw()
        print("Move Log: ", self.move_log)

    def run_ais(
        self,
        ai_white: Callable[["State"], bool],
        ai_black: Callable[["State"], bool],
        log_states: bool = False,
        log_number_of_moves: bool = False,
    ) -> tuple[Player, float, float]:
        """
        Run a "simulation" with 2 AI players until it is over.

        Parameters:
        - ai_white: A function that represents the AI player for the white side. It takes a "State" object as input and returns the new state after executing its move.
        - ai_black: A function that represents the AI player for the black side. It takes a "State" object as input and returns the new state after executing its move.
        - log_states: A boolean flag indicating whether to log the game states during the AI player's moves. Default is False.

        Returns:
        - The winner of the game.
        """
        time_white, time_black = 0, 0
        number_of_moves = 0
        while not self.game_over():
            number_of_moves += 1
            if log_states:
                print()
                print(self.get_available_moves())
                self.draw() # print state
            if self.is_white_turn():
                start = time.time()
                self = ai_white(self)
                end = time.time()
                time_white += end - start   # measure white time
            else:
                start = time.time()
                self = ai_black(self)
                end = time.time()
                time_black += end - start   # measure black time

        if self.count == DRAW_COUNTER_THRESHOLD and log_states:
            # if X moves have passed without a capture, the game is a draw (in our simulations)
            print(f"Draw by {DRAW_COUNTER_THRESHOLD} moves rule (no captures)")

        if log_states:
            self.draw() # print state

        winner = self.check_winner()

        if log_states:
            print(f"Winner: {winner}")

        if log_number_of_moves:
            print(f"Number of moves: {number_of_moves}")

        return winner, time_white, time_black
