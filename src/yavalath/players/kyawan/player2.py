"""
AlphaBetaで探索するプレイヤー

評価関数を実装したバージョン
- 打てない場所の数だけマイナス評価
"""

import random

from yavalath.core.board import Board, CellState, PutResult
from yavalath.core.player import Player

DEPTH_PHASE1 = 2
DEPTH_PHASE2 = 3
DEPTH_PHASE3 = 4
DEPTH_PHASE4 = 8
PHASE1_EMPTY_CELLS_THRESHOLD = 24
PHASE2_EMPTY_CELLS_THRESHOLD = 12
PHASE3_EMPTY_CELLS_THRESHOLD = 8

WIN_SCORE = 10000
LOSE_SCORE = -20000
MINUS_PER_LOSE_CELL = 10
PLUS_PER_WIN_CELL = 5


class KyawanPlayerV2(Player):
    def __init__(self):
        super().__init__(
            name="KyawanV2",
            color=(200, 100, 240),
        )

    def calc_best(self, board, player) -> tuple[int, int, int]:
        # フェーズに応じて探索深度を変更
        empty_cells = board.get_empty_cells()
        if len(empty_cells) > PHASE1_EMPTY_CELLS_THRESHOLD:
            depth = DEPTH_PHASE1
        elif len(empty_cells) > PHASE2_EMPTY_CELLS_THRESHOLD:
            depth = DEPTH_PHASE2
        elif len(empty_cells) > PHASE3_EMPTY_CELLS_THRESHOLD:
            depth = DEPTH_PHASE3
        else:
            depth = DEPTH_PHASE4

        best_score, best_pos = self.negamax(
            player, board, depth, float("-inf"), float("inf")
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
            return self.evaluate(board, player), None  # 評価関数を使用

        best_score = float("-inf")
        best_pos = None

        next_player = (
            CellState.PLAYER1 if player == CellState.PLAYER2 else CellState.PLAYER2
        )
        pos_list = board.get_empty_cells()
        if len(pos_list) == 0:
            return 0, None  # 引き分け
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

    def evaluate(self, board: Board, player: CellState) -> float:
        empty_cells = board.get_empty_cells()
        score = 0

        for pos in empty_cells:
            result = board.put(pos, player)
            board.pick(pos)
            match result:
                case PutResult.WIN:
                    score += PLUS_PER_WIN_CELL
                case PutResult.LOSE:
                    score -= MINUS_PER_LOSE_CELL
        return score
