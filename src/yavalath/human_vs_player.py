import argparse
import math

import pygame

from yavalath.core.board import Board, CellState, PutResult
from yavalath.core.player import Player
from yavalath.renderer import PygameRenderer
from yavalath.players.inoue.player import InouePlayer
from yavalath.players.random.player import RandomPlayer
from yavalath.players.inoue.player2 import AInouePlayer


PLAYER_FACTORIES = {
    "inoueModel": InouePlayer,
    "ainoueModel": AInouePlayer,
}

COLOR_P1 = (229, 192, 123)
COLOR_P2 = (97, 175, 239)


class HumanPlayer(Player):
    def __init__(self, name: str, color: tuple[int, int, int]):
        super().__init__(name, color)

    def calc_best(self, board: Board, player: CellState) -> tuple[int, int, int]:
        raise NotImplementedError("Human player does not calculate moves.")


def _build_ai(name: str):
    factory = PLAYER_FACTORIES.get(name)
    if factory is None:
        raise ValueError(f"Unknown player: {name}")
    try:
        return factory()
    except TypeError:
        return factory(name.capitalize())


def _build_cell_centers(renderer: PygameRenderer, board: Board):
    centers = {}
    for pos in board.board.keys():
        x, y, z = pos
        cx, cy = renderer._hex_to_pixel(x, z)
        centers[pos] = (cx, cy)
    return centers


def _pick_cell_from_mouse(mouse_pos, centers, max_dist):
    mx, my = mouse_pos
    best_pos = None
    best_dist = None
    for pos, (cx, cy) in centers.items():
        dist = math.hypot(mx - cx, my - cy)
        if dist <= max_dist and (best_dist is None or dist < best_dist):
            best_dist = dist
            best_pos = pos
    return best_pos


def main():
    parser = argparse.ArgumentParser(description="Human vs AI Yavalath")
    parser.add_argument(
        "--player",
        choices=sorted(PLAYER_FACTORIES.keys()),
        default="ainoueModel",
        help="AI player type",
    )
    parser.add_argument(
        "--human",
        choices=["p1", "p2"],
        default="p1",
        help="Which side the human plays",
    )
    parser.add_argument("--radius", type=int, default=4, help="Board radius")
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    pygame.display.set_caption("Yavalath Human vs AI")

    board = Board(args.radius)
    renderer = PygameRenderer(screen, args.radius)
    centers = _build_cell_centers(renderer, board)

    ai_player = _build_ai(args.player)
    human_player = HumanPlayer("Human", COLOR_P1 if args.human == "p1" else COLOR_P2)

    if args.human == "p1":
        p1 = human_player
        p2 = ai_player
        p2.color = COLOR_P2
    else:
        p1 = ai_player
        p1.color = COLOR_P1
        p2 = human_player

    turn = CellState.PLAYER1
    last_move = None
    message = ""
    game_over = False
    game_over_waited = False
    ai_delay_until = None

    clock = pygame.time.Clock()

    while True:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                return

            if game_over:
                continue

            is_human_turn = (turn == CellState.PLAYER1 and args.human == "p1") or (
                turn == CellState.PLAYER2 and args.human == "p2"
            )

            if is_human_turn and event.type == pygame.MOUSEBUTTONDOWN:
                if event.button != 1:
                    continue
                pos = _pick_cell_from_mouse(
                    event.pos, centers, max_dist=renderer.hex_size * 0.9
                )
                if pos is None or not board.can_put(pos):
                    continue

                result = board.put(pos, turn)
                last_move = pos

                if result == PutResult.WIN:
                    message = "Human wins!"
                    game_over = True
                elif result == PutResult.LOSE:
                    message = f"{ai_player.name} wins!"
                    game_over = True
                elif not board.get_empty_cells():
                    message = "Draw"
                    game_over = True
                else:
                    turn = (
                        CellState.PLAYER2
                        if turn == CellState.PLAYER1
                        else CellState.PLAYER1
                    )
                    ai_delay_until = pygame.time.get_ticks() + 1000

        if not game_over:
            is_human_turn = (turn == CellState.PLAYER1 and args.human == "p1") or (
                turn == CellState.PLAYER2 and args.human == "p2"
            )

            if not is_human_turn:
                try:
                    if ai_delay_until is not None:
                        now = pygame.time.get_ticks()
                        if now < ai_delay_until:
                            pass
                        else:
                            ai_delay_until = None

                    if ai_delay_until is not None:
                        # 待機中は描画だけ行い、AIの着手は行わない
                        pass
                    else:
                        pos = ai_player.calc_best(board, turn)
                        result = board.put(pos, turn)
                        last_move = pos

                        if result == PutResult.WIN:
                            message = f"{ai_player.name} wins!"
                            game_over = True
                        elif result == PutResult.LOSE:
                            message = "Human wins!"
                            game_over = True
                        elif not board.get_empty_cells():
                            message = "Draw"
                            game_over = True
                        else:
                            turn = (
                                CellState.PLAYER2
                                if turn == CellState.PLAYER1
                                else CellState.PLAYER1
                            )
                except Exception as e:
                    message = f"AI error: {e}"
                    game_over = True
                    ai_delay_until = None

        if not game_over:
            if (turn == CellState.PLAYER1 and args.human == "p1") or (
                turn == CellState.PLAYER2 and args.human == "p2"
            ):
                message = "Human's turn"
            else:
                message = f"{ai_player.name} thinking..."

        renderer.draw_game(
            board,
            p1,
            p2,
            last_move=last_move,
            message=message,
        )

        if game_over and not game_over_waited:
            pygame.time.wait(1200)
            game_over_waited = True


if __name__ == "__main__":
    main()
