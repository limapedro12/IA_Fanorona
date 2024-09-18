import random
import math
from moves.move import Move
from player import Player
from state import State
from typing import Callable

EPS = 1e-9


def get_random_move(state: State) -> Move | None:
    """
    Returns a random move from the available moves of the state.
    """
    moves = state.get_available_moves()
    if moves == []:  # No moves available
        return None
    return random.choice(moves)


def execute_random_move(state: State) -> State:
    """
    Returns a new state after executing a random move.
    """
    move = get_random_move(state)
    state = state.execute_move(move)
    return state


def minimax(
    state: State,
    depth,
    alpha,
    beta,
    player_to_win,
    evaluate_func: Callable[[State, Player], float],
):
    """
    Minimax algorithm with alpha-beta pruning.

    Evaluates the state using the evaluate_func and returns the best evaluation.

    Args:
    - state: The state of the game.
    - depth: The depth of the search tree.
    - alpha: The alpha value for alpha-beta pruning.
    - beta: The beta value for alpha-beta pruning.
    - player_to_win: The player that should win.
    - evaluate_func: The evaluation function to evaluate the state.
    """
    if depth == 0 or state.check_winner() != Player.EMPTY:  # leaf node
        return evaluate_func(state, player_to_win)

    if player_to_win == state.player:  # MAX
        max_eval = -math.inf
        for mv in state.get_available_moves():
            nstate = state.execute_move(mv)
            eval = minimax(nstate, depth - 1, alpha, beta, player_to_win, evaluate_func)
            max_eval = max(max_eval, eval)

            alpha = max(alpha, eval)
            if beta <= alpha:  # cut
                break
        return max_eval

    else:  # MIN
        min_eval = math.inf
        for mv in state.get_available_moves():
            nstate = state.execute_move(mv)
            eval = minimax(nstate, depth - 1, alpha, beta, player_to_win, evaluate_func)
            min_eval = min(min_eval, eval)

            beta = min(beta, eval)
            if beta <= alpha:  # cut
                break
        return min_eval


def get_minimax_move(evaluate_func, depth: int) -> Move | None:
    """
    Returns a function that returns the best move for a given state using the minimax algorithm with alpha-beta pruning,
    according to the evaluation function and using the depth given.
    """

    def get_minimax_move_aux(state: State) -> Move:
        """
        Auxiliar function to get the best move using the minimax algorithm for the given state,
        returning the resulting state.
        """
        best_moves = []

        alpha = best_eval = -math.inf
        beta = math.inf
        for mv in state.get_available_moves():
            nstate = state.execute_move(mv)
            nstate_eval = minimax(
                nstate, depth - 1, alpha, beta, state.player, evaluate_func
            )

            if nstate_eval > best_eval:
                alpha = best_eval = nstate_eval
                best_moves.clear()  # clear previous best moves, since this is a better move
                best_moves.append(mv)

            if best_eval - EPS <= nstate_eval <= best_eval + EPS:  # float equality
                best_moves.append(mv)  # move is as good as the best move

        if best_moves != []:
            return random.choice(best_moves)  # return a random move from the best moves
        return None  # no moves available

    return get_minimax_move_aux


def execute_minimax_move(evaluate_func, depth: int) -> State:
    """
    Returns a function that executes the best move for a given state and returns the resulting state
    using the minimax algorithm with alpha-beta pruning,
    according to the evaluation function and using the depth given.
    """

    def execute_minimax_move_aux(state: State):
        """
        Auxiliar function that executes the best move using the minimax algorithm for the given state,
        returning the resulting state.
        """
        move = get_minimax_move(evaluate_func, depth)(state)
        return state.execute_move(move)

    return execute_minimax_move_aux
