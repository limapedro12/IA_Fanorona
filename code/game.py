import pygame
import numpy as np
from enum import Enum

from state import State
from player import Player
import minimax
from minimax import get_random_move
from heuristics.nr_pieces_heuristic import NrPiecesHeuristic
from heuristics.adjacent_pieces_heuristic import AdjacentPiecesHeuristic
from heuristics.heuristics_list import HeuristicsList
from heuristics.win_heuristic import WinHeuristic
from heuristics.groups_heuristic import GroupsHeuristic
from heuristics.center_control_heuristic import CenterControlHeuristic
from heuristics.approximate_enemy_heuristic import ApproximateEnemyHeuristic

from monte_carlo_tree_search.tree import MonteCarloTree
from monte_carlo_tree_search.tree_heuristics import MonteCarloTreeHeuristic
from moves.move import Move
from moves.motion_move import MotionMove
from moves.pass_move import PassMove

from button import Button


WHITE_COLOR = (255, 255, 255)
BLACK_COLOR = (0, 0, 0)
GREEN_COLOR = (0, 255, 0)
GRAY_COLOR_WHITE = (128, 128, 128)
GRAY_COLOR_BLACK = (64, 64, 64)
YELLOW_COLOR = (235, 235, 52)
STRONG_RED_COLOR = (255, 0, 0)
BG_RED_COLOR = (184, 59, 50)

MIN_SIZE, MAX_SIZE = 5, 10


class WindowState(Enum):
    """Enum to represent the current state of the window (program)"""

    BOARD_SIZE_SEL = 0
    WHITE_MODE_SEL = 1
    BLACK_MODE_SEL = 2
    PLAYING = 3
    GAME_OVER = 4


class PlayerModes(Enum):
    """Enum to represent the different playing modes of a team"""

    HUMAN = 0
    RANDOM = 1
    MINIMAX_VERY_EASY = 2
    MINIMAX_EASY = 3
    MINIMAX_DEFENSIVE_EASY = 4
    MINIMAX_DEFENSIVE_HARD = 5
    MINIMAX_AGRESSIVE_EASY = 6
    MINIMAX_AGRESSIVE_HARD = 7
    MCTS_QUICK = 8
    MCTS_BETTER = 9
    MCTS_HEURISTICS = 10

    def __str__(self):
        arr = [
            "Humano",
            "Aleatório",
            "Minimax (Muito Fácil)",
            "Minimax (Fácil)",
            "Minimax (Defensivo Fácil)",
            "Minimax (Defensivo Difícil)",
            "Minimax (Agressivo Fácil)",
            "Minimax (Agressivo Difícil)",
            "MCTS (Rápido)",
            "MCTS (Melhor)",
            "MCTS (Heurísticas)",
        ]
        return arr[self.value]


class Game:
    """Class to represent the game and the window"""

    DEFAULT_WIDTH = 7
    DEFAULT_HEIGHT = 7

    @staticmethod
    def get_canvas_width(width):
        return width * 70

    @staticmethod
    def get_canvas_height(height):
        return height * 70 + 40

    def get_default_canvas_width(self):
        return self.get_canvas_width(self.DEFAULT_WIDTH)

    def get_default_canvas_height(self):
        return self.get_canvas_height(self.DEFAULT_HEIGHT)

    def __init__(self):
        # Set up the window
        pygame.init()
        self.change_canvas_size(self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)
        self.canvas.fill(BG_RED_COLOR)  # Background
        self.font = pygame.font.Font("freesansbold.ttf", 15)
        pygame.display.set_caption("Fanorona")
        pygame.display.update()

        self.frame_rate = 30
        self.frame_time_counter = 0

        self.window_state = WindowState.BOARD_SIZE_SEL
        self.selected_piece = None

        self.width = None
        self.height = None

        self.game_state: State = None
        self.winner: Player = Player.EMPTY
        self.white_mode = None
        self.black_mode = None
        self.available_moves = None

        self.white_alg = None
        self.black_alg = None

        self.size_sel_title = self.font.render(
            "Selecione o tamanho do tabuleiro", True, WHITE_COLOR
        )
        self.size_sel_title_rect = self.size_sel_title.get_rect(
            center=(self.get_default_canvas_width() // 2, 15)
        )

        self.size_sel_buttons: list[Button] = []
        for col in range(MIN_SIZE, MAX_SIZE + 1):
            for row in range(MIN_SIZE, MAX_SIZE + 1):
                self.size_sel_buttons.append(
                    Button(
                        (col - 4) * self.DEFAULT_WIDTH * 10,
                        (row - 4) * self.DEFAULT_HEIGHT * 10,
                        f"{col}*{row}",
                        self.canvas,
                        self.font,
                        BLACK_COLOR,
                        WHITE_COLOR,
                        self.size_sel_button_action(col, row),
                    )
                )
        self.mode_sel_buttons = [
            Button(
                x=self.get_default_canvas_width() // 2,
                y=(i + 2) * 40,
                text=str(mode),
                canvas=self.canvas,
                font=self.font,
                textColor=BLACK_COLOR,
                bgColor=WHITE_COLOR,
                action=self.mode_sel_button_action(mode),
                width=self.get_default_canvas_width() // 2,
                height=None,
            )
            for i, mode in enumerate(PlayerModes)
        ]

        self.widthStr = None
        self.whiteTypeStr = None

        self.back_buttons = [
            self.create_back_button(self.back_to_board_size_sel(from_game_over=False)),
            self.create_back_button(self.back_to_white_mode_sel),
            self.create_back_button(self.back_to_black_mode_sel),
            self.create_back_button(self.back_to_board_size_sel(from_game_over=True)),
        ]
        self.selected_moves = []

    def back_to_white_mode_sel(self):
        """Go back to the white team's mode selection screen"""

        self.white_mode = None
        self.window_state = WindowState.WHITE_MODE_SEL

    def back_to_black_mode_sel(self):
        """Go back to the black team's mode selection screen"""

        self.black_mode = None
        self.change_canvas_size(self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)
        self.window_state = WindowState.BLACK_MODE_SEL

    def back_to_board_size_sel(self, from_game_over):
        """Go back to the board size selection screen"""

        def func():
            if from_game_over:
                self.winner = Player.EMPTY
                self.selected_piece = None
                self.back_to_black_mode_sel()
                self.back_to_white_mode_sel()
                self.back_to_board_size_sel(from_game_over=False)()
            self.width = None
            self.height = None
            self.window_state = WindowState.BOARD_SIZE_SEL

        return func

    def create_back_button(self, action) -> Button:
        return Button(
            30,
            15,
            "Voltar",
            self.canvas,
            self.font,
            BLACK_COLOR,
            WHITE_COLOR,
            action=action,
        )

    def get_back_button(self) -> Button | None:
        match self.window_state:
            case WindowState.WHITE_MODE_SEL:
                return self.back_buttons[0]
            case WindowState.BLACK_MODE_SEL:
                return self.back_buttons[1]
            case WindowState.PLAYING:
                return self.back_buttons[2]
            case WindowState.GAME_OVER:
                return self.back_buttons[3]
        return None

    def mode_sel_button_action(self, mode):
        def func():
            if self.window_state == WindowState.WHITE_MODE_SEL:
                self.white_mode = PlayerModes(mode)
                self.whiteTypeStr = str(mode)
            elif self.window_state == WindowState.BLACK_MODE_SEL:
                self.black_mode = PlayerModes(mode)
                self.game_state = State(self.width, self.height)
                self.available_moves = self.game_state.get_available_moves()
                self.selected_piece = None

        return func

    def change_canvas_size(self, width, height):
        """Changes pygame's window (canvas) size"""
        self.canvas = pygame.display.set_mode(
            (self.get_canvas_width(width), self.get_canvas_height(height))
        )

    def size_sel_button_action(self, col, row):
        def func():
            self.width = col
            self.height = row
            self.window_state = WindowState.WHITE_MODE_SEL
            self.widthStr = f"{self.width}*{self.height}"

        return func

    def check_exit_event(self, event):
        """Checks if a pygame event is an exit event and exits the program if so"""
        if event.type == pygame.QUIT or (
            event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
        ):
            pygame.quit()
            exit()

    def size_sel(self):
        """Draws the board size selection screen and handles its logic"""

        self.canvas.fill(BG_RED_COLOR)  # Background
        self.canvas.blit(self.size_sel_title, self.size_sel_title_rect)

        back_button = self.get_back_button()
        if back_button is not None:
            back_button.draw()

        for button in self.size_sel_buttons:
            button.draw()

        for event in pygame.event.get():
            self.check_exit_event(event)

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                for button in self.size_sel_buttons:
                    if button.mouse_collision(x, y):
                        button.action()
                        return

                if back_button is not None and back_button.mouse_collision(x, y):
                    back_button.action()
                    return

        pygame.display.update()

    def draw_mode_sel_text(self):
        textList = []

        size_str = f"Tamanho: {self.widthStr}"
        if self.window_state == WindowState.WHITE_MODE_SEL:
            textList += [
                self.font.render(size_str, True, WHITE_COLOR),
                self.font.render(
                    "Selecione o modo da equipa branca", True, WHITE_COLOR
                ),
            ]
        elif self.window_state == WindowState.BLACK_MODE_SEL:
            textList += [
                self.font.render(
                    size_str + f"  Branco: {self.whiteTypeStr}", True, BLACK_COLOR
                ),
                self.font.render("Selecione o modo da equipa preta", True, BLACK_COLOR),
            ]
        for i, text in enumerate(textList):
            textRect = text.get_rect(
                center=(self.get_default_canvas_width() // 2, 15 + 15 * (i + 1))
            )
            self.canvas.blit(text, textRect)

    def choose_alg(self, mode: PlayerModes, player: Player):
        """Returns the algorithm, correctly initialized, to be used by a team according to the mode selected"""

        match mode:
            case PlayerModes.RANDOM:
                return get_random_move
            case PlayerModes.MINIMAX_VERY_EASY:
                return minimax.get_minimax_move(
                    HeuristicsList(
                        np.array([WinHeuristic(), ApproximateEnemyHeuristic()]),
                        np.array([1e6, 1]),
                    ).evaluate_board,
                    2,
                )
            case PlayerModes.MINIMAX_EASY:
                return minimax.get_minimax_move(
                    HeuristicsList(
                        np.array([WinHeuristic(), NrPiecesHeuristic()]),
                        np.array([1e6, 1]),
                    ).evaluate_board,
                    2,
                )
            case PlayerModes.MINIMAX_DEFENSIVE_EASY:
                return minimax.get_minimax_move(
                    HeuristicsList(
                        np.array(
                            [
                                WinHeuristic(),
                                NrPiecesHeuristic(),
                                GroupsHeuristic(),
                                AdjacentPiecesHeuristic(),
                                CenterControlHeuristic(),
                            ]
                        ),
                        np.array([1e6, 10, 2, 1, 1]),
                    ).evaluate_board,
                    2,
                )
            case PlayerModes.MINIMAX_DEFENSIVE_HARD:
                return minimax.get_minimax_move(
                    HeuristicsList(
                        np.array(
                            [
                                WinHeuristic(),
                                NrPiecesHeuristic(),
                                GroupsHeuristic(),
                                CenterControlHeuristic(),
                            ]
                        ),
                        np.array([1e6, 10, 1, 1]),
                    ).evaluate_board,
                    4,
                )
            case PlayerModes.MINIMAX_AGRESSIVE_EASY:
                return minimax.get_minimax_move(
                    HeuristicsList(
                        np.array([WinHeuristic(), NrPiecesHeuristic(), ApproximateEnemyHeuristic()]),
                        np.array([1e6, 10, 2]),
                    ).evaluate_board,
                    2,
                )
            case PlayerModes.MINIMAX_AGRESSIVE_HARD:
                return minimax.get_minimax_move(
                    HeuristicsList(
                        np.array([WinHeuristic(), NrPiecesHeuristic(), ApproximateEnemyHeuristic()]),
                        np.array([1e6, 2, 1]),
                    ).evaluate_board,
                    4,
                )
            case PlayerModes.MCTS_QUICK | PlayerModes.MCTS_BETTER:
                return MonteCarloTree.from_player(self.width, self.height, player)
            case PlayerModes.MCTS_HEURISTICS:
                return MonteCarloTreeHeuristic.from_player(
                    HeuristicsList(
                        heuristics=np.array(
                            [
                                WinHeuristic(),
                                NrPiecesHeuristic(),
                                GroupsHeuristic(),
                                CenterControlHeuristic(),
                            ]
                        ),
                        weights=np.array([100000, 50, 10, 5]),
                    ),
                    self.width,
                    self.height,
                    player,
                )

    def mode_sel(self):
        """Draws the mode selection screen and handles the mode selection logic"""

        self.canvas.fill(BG_RED_COLOR)  # Background
        self.draw_mode_sel_text()

        back_button = self.get_back_button()
        if back_button is not None:
            back_button.draw()
        for button in self.mode_sel_buttons:
            button.draw()

        pygame.display.update()

        for event in pygame.event.get():
            self.check_exit_event(event)
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                for button in self.mode_sel_buttons:
                    if button.mouse_collision(x, y):
                        button.action()

                if back_button is not None and back_button.mouse_collision(x, y):
                    back_button.action()
                    return

        if (
            self.white_mode is not None
            and self.window_state == WindowState.WHITE_MODE_SEL
        ):
            self.white_alg = self.choose_alg(self.white_mode, Player.WHITE)
            self.window_state = WindowState.BLACK_MODE_SEL
        elif (
            self.black_mode is not None
            and self.window_state == WindowState.BLACK_MODE_SEL
        ):
            self.black_alg = self.choose_alg(self.black_mode, Player.BLACK)
            self.window_state = WindowState.PLAYING
            self.change_canvas_size(self.width, self.height)

        return

    def draw_static_board_elements(self):
        for i in range(self.width):
            pygame.draw.line(
                self.canvas,
                WHITE_COLOR,
                (70 * i + 35, 35),
                (70 * i + 35, 70 * self.height - 35),
                2,
            )
        for i in range(self.height):
            pygame.draw.line(
                self.canvas,
                WHITE_COLOR,
                (35, 70 * i + 35),
                (70 * self.width - 35, 70 * i + 35),
                2,
            )
        for i in range(self.width - 1):
            for j in range(self.height - 1):
                if (i + j) % 2 == 0:
                    pygame.draw.line(
                        self.canvas,
                        WHITE_COLOR,
                        (70 * i + 35, 70 * j + 35),
                        (70 * i + 105, 70 * j + 105),
                        2,
                    )
                else:
                    pygame.draw.line(
                        self.canvas,
                        WHITE_COLOR,
                        (70 * i + 105, 70 * j + 35),
                        (70 * i + 35, 70 * j + 105),
                        2,
                    )

    def execute_human_move(self, move: Move):
        if self.game_state.player == Player.BLACK and self.is_mcts(self.white_mode):
            self.white_alg.update_move(move)
        elif self.game_state.player == Player.WHITE and self.is_mcts(self.black_mode):
            self.black_alg.update_move(move)

        self.selected_moves = []
        self.game_state = self.game_state.execute_move(move)  # the order matters here
        self.selected_piece = None

    def draw_pieces(self):
        for row in range(self.height):
            for col in range(self.width):
                if self.game_state.get_board_matrix()[row][col] == Player.WHITE:
                    pygame.draw.circle(
                        self.canvas, WHITE_COLOR, (70 * col + 35, 70 * row + 35), 30
                    )
                elif self.game_state.board.board[row][col] == Player.BLACK:
                    pygame.draw.circle(
                        self.canvas, BLACK_COLOR, (70 * col + 35, 70 * row + 35), 30
                    )

    def draw_selected_piece(self):
        if self.selected_piece != None:
            pygame.draw.circle(
                self.canvas,
                YELLOW_COLOR,
                (70 * self.selected_piece[1] + 35, 70 * self.selected_piece[0] + 35),
                30,
                5,
            )

    def handle_human_moves(self):
        back_button = self.get_back_button()
        for move in self.available_moves:
            if isinstance(move, MotionMove) and self.selected_piece == (
                move.row_origin,
                move.col_origin,
            ):
                pygame.draw.circle(
                    self.canvas,
                    GREEN_COLOR,
                    (
                        70 * move.col_destination + 35,
                        70 * move.row_destination + 35,
                    ),
                    30,
                    3,
                )
        for move in self.selected_moves:
            row, col = move.get_first_to_kill()
            gray = (
                GRAY_COLOR_WHITE
                if self.game_state.get_board_matrix()[row][col] == Player.WHITE
                else GRAY_COLOR_BLACK
            )
            pygame.draw.circle(self.canvas, gray, (70 * col + 35, 70 * row + 35), 30)
            pygame.draw.circle(
                self.canvas,
                STRONG_RED_COLOR,
                (70 * col + 35, 70 * row + 35),
                30,
                5,
            )

        for event in pygame.event.get():
            self.check_exit_event(event)

            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                passMove = PassMove()
                if passMove in self.available_moves:
                    self.execute_human_move(passMove)
                    pygame.display.update()
                    return

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                if back_button is not None and back_button.mouse_collision(x, y):
                    back_button.action()
                    return
                col = x // 70
                row = y // 70
                if (
                    self.game_state.get_board_matrix()[row][col]
                    == self.game_state.player
                ):
                    self.selected_piece = (row, col)
                    self.selected_moves = []
                elif self.game_state.get_board_matrix()[row][col] == Player.EMPTY:
                    self.selected_moves = []
                    for move in self.available_moves:
                        if (
                            isinstance(move, MotionMove)
                            and self.selected_piece
                            == (
                                move.row_origin,
                                move.col_origin,
                            )
                            and (row, col)
                            == (
                                move.row_destination,
                                move.col_destination,
                            )
                        ):
                            self.selected_moves.append(move)

                    if len(self.selected_moves) == 1:
                        self.execute_human_move(self.selected_moves[0])
                        pygame.display.update()
                        return
                else:
                    for move in self.selected_moves:
                        if (row, col) == move.get_first_to_kill():
                            self.execute_human_move(move)
                            self.selected_moves = []
                            pygame.display.update()
                            return

    def is_minimax(self, mode):
        return (
            mode == PlayerModes.MINIMAX_VERY_EASY
            or mode == PlayerModes.MINIMAX_EASY
            or mode == PlayerModes.MINIMAX_DEFENSIVE_EASY
            or mode == PlayerModes.MINIMAX_DEFENSIVE_HARD
            or mode == PlayerModes.MINIMAX_AGRESSIVE_EASY
            or mode == PlayerModes.MINIMAX_AGRESSIVE_HARD
        )
    def is_random(self, mode):
        return mode == PlayerModes.RANDOM

    def is_mcts(self, mode):
        return (
            mode == PlayerModes.MCTS_QUICK
            or mode == PlayerModes.MCTS_BETTER
            or mode == PlayerModes.MCTS_HEURISTICS
        )

    def train_mcts(self, mode, alg):
        match mode:
            case PlayerModes.MCTS_QUICK:
                alg.train_until(100)
            case PlayerModes.MCTS_BETTER:
                alg.train_until(1000)
            case PlayerModes.MCTS_HEURISTICS:
                alg.train_until(2000)

    def get_current_npc(self):
        if self.game_state.player == Player.WHITE:
            return self.white_mode, self.white_alg
        return self.black_mode, self.black_alg

    def get_other_npc(self):
        if self.game_state.player == Player.WHITE:
            return self.black_mode, self.black_alg
        return self.white_mode, self.white_alg

    def get_pass_move_text(self, color):
        return self.font.render("Pressione Enter para passar a jogada", True, color)

    def get_white_move_text(self):
        return self.font.render("Vez das brancas", True, WHITE_COLOR)

    def get_black_move_text(self):
        return self.font.render("Vez das pretas", True, BLACK_COLOR)

    def get_npc_move_text(self, color):
        return self.font.render("Pressione Enter para continuar", True, color)

    def draw_playing_text(self):
        """Draws text indicating current player's turn"""
        textList = []
        if self.game_state.player == Player.BLACK:
            if self.black_mode == PlayerModes.HUMAN:
                textList.append(self.get_black_move_text())
                if PassMove() in self.available_moves:
                    textList.append(self.get_pass_move_text(BLACK_COLOR))
            else:
                textList += [
                    self.get_black_move_text(),
                    self.get_npc_move_text(BLACK_COLOR),
                ]
        elif self.game_state.player == Player.WHITE:
            if self.white_mode == PlayerModes.HUMAN:
                textList.append(self.get_white_move_text())
                if PassMove() in self.available_moves:
                    textList.append(self.get_pass_move_text(WHITE_COLOR))
            else:
                textList += [
                    self.get_white_move_text(),
                    self.get_npc_move_text(WHITE_COLOR),
                ]

        for i, text in enumerate(textList):
            textRect = text.get_rect(
                center=(self.width * 70 // 2, self.height * 70 + 7 + 15 * i)
            )
            self.canvas.blit(text, textRect)

    def board(self):
        """Draws the board and handles the game logic"""

        back_button = self.get_back_button()

        self.canvas.fill(BG_RED_COLOR)  # Background
        self.draw_static_board_elements()
        self.draw_pieces()

        if self.selected_piece != None:
            self.draw_selected_piece()

        self.available_moves = self.game_state.get_available_moves()
        self.draw_playing_text()

        if back_button is not None:
            back_button.draw()

        if (
            self.white_mode == PlayerModes.HUMAN
            and self.game_state.player == Player.WHITE
        ) or (
            self.black_mode == PlayerModes.HUMAN
            and self.game_state.player == Player.BLACK
        ):
            self.handle_human_moves()
        else:
            move = None
            mode, alg = self.get_current_npc()
            other_mode, other_alg = self.get_other_npc()

            for event in pygame.event.get():
                self.check_exit_event(event)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    if back_button is not None and back_button.mouse_collision(x, y):
                        back_button.action()
                        return

                elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    if self.is_minimax(mode) or self.is_random(mode):
                        move = alg(self.game_state)
                        self.game_state = self.game_state.execute_move(move)
                        break
                    elif self.is_mcts(mode):
                        self.train_mcts(mode, alg)
                        move = alg.get_best_move()
                        self.game_state = self.game_state.execute_move(move)
                        alg.update_move(move)
                        break

            if move is not None and self.is_mcts(other_mode):
                other_alg.update_move(move)

        pygame.display.update()

    def get_winner_text(self):
        if self.winner == Player.BLACK:
            return "Pretas vencem!", BLACK_COLOR
        elif self.winner == Player.WHITE:
            return "Brancas vencem!", WHITE_COLOR
        else:
            return "Empate!", BLACK_COLOR

    def draw_text(self, x, y, text, color):
        text = self.font.render(text, True, color)
        textRect = text.get_rect(center=(x, y))
        self.canvas.blit(text, textRect)

    def game_over(self):
        """Draws the game over screen"""

        self.canvas.fill(BG_RED_COLOR)  # Background

        color = None
        text, color = self.get_winner_text()
        self.draw_text(self.width * 35, self.height * 35, text, color)

        textList = [
            "Obrigado por jogar Fanorona!",
            "Félix Martins, Pedro Lima e Pedro Januário",
            "Pressione ESC ou feche a janela para sair",
            "Para jogar novamente, clique em voltar",
        ]
        height_offsets = [20, 50, 90, 110]
        for text, offset in zip(textList, height_offsets):
            self.draw_text(self.width * 35, self.height * 35 + offset, text, color)

        back_button = self.get_back_button()
        if back_button is not None:
            back_button.draw()

        pygame.display.update()

        for event in pygame.event.get():
            self.check_exit_event(event)
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                if back_button is not None and back_button.mouse_collision(x, y):
                    back_button.action()
                    return

    def play(self):
        """Handles program main loop, calling drawers/handlers according to the current state"""

        while True:
            if self.window_state == WindowState.PLAYING:
                self.winner = self.game_state.check_winner()
                if self.winner != Player.EMPTY:
                    self.window_state = WindowState.GAME_OVER
                    continue
                self.board()
            elif self.window_state == WindowState.BOARD_SIZE_SEL:
                self.size_sel()
            elif (
                self.window_state == WindowState.WHITE_MODE_SEL
                or self.window_state == WindowState.BLACK_MODE_SEL
            ):
                self.mode_sel()
            elif self.window_state == WindowState.GAME_OVER:
                self.game_over()


if __name__ == "__main__":
    game = Game()
    game.play()
