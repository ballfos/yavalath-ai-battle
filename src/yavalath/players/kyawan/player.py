"""
AlphaBetaで探索するプレイヤー

"""

import random

from yavalath.core.board import Board, CellState, PutResult
from yavalath.core.player import Player

DEPTH = 4
WIN_SCORE = 10000
LOSE_SCORE = -20000


class KyawanPlayer(Player):
    def __init__(self):
        super().__init__(
            name="Kyawan",
            color=(255, 215, 0),
        )

    def calc_best(self, board, player) -> tuple[int, int, int]:
        best_score, best_pos = self.negamax(
            player, board, DEPTH, float("-inf"), float("inf")
        )
        if best_pos is None:
            best_pos = random.choice(board.get_empty_cells())
        return best_pos

    def negamax(
        self,
        player: CellState,
        board: Board,
        depth: int,
        alpha: float,
        beta: float,
    ) -> tuple[float, tuple[int, int, int]]:
        if depth == 0:
            return 0, None  # TODO:評価関数

        best_score = float("-inf")
        best_pos = None

        next_player = (
            CellState.PLAYER1 if player == CellState.PLAYER2 else CellState.PLAYER2
        )
        pos_list = board.get_empty_cells()
        random.shuffle(pos_list)
        for pos in pos_list:
            result = board.put(pos, player)

            # 勝敗が決まった場合はそのスコアを返す
            if result == PutResult.WIN:
                score = WIN_SCORE * (depth + 1)
            elif result == PutResult.LOSE:
                score = LOSE_SCORE * (depth + 1)
            else:
                # 再帰的に探索
                score, _ = self.negamax(
                    next_player,
                    board,
                    depth - 1,
                    -beta,
                    -alpha,
                )
                score = -score  # ミニマックスの反転

            board.pick(pos)

            if score > best_score:
                best_score = score
                best_pos = pos

            alpha = max(alpha, best_score)
            if beta <= alpha:
                break  # βカット
        return best_score, best_pos
