import sys

import pygame
from core.board import Board, CellState
from renderer import PygameRenderer  # 前回作成した描画クラスを流用


class ReplayViewer:
    def __init__(self, move_history, p1_name, p2_name, radius=4):
        """
        move_history: [(CellState, (x, y, z)), ...] のリスト
        """
        self.moves = move_history
        self.radius = radius
        self.p1_name = p1_name
        self.p2_name = p2_name
        self.board = Board(radius)
        self.current_step = 0  # 現在何手目まで表示しているか

        # Pygame setup
        pygame.init()
        screen = pygame.display.set_mode((1280, 720))
        pygame.display.set_caption("Yavalath Replay Viewer")
        self.renderer = PygameRenderer(screen, radius)

        # UI用フォント
        self.font = pygame.font.SysFont("Arial", 24)

    def run(self):
        running = True
        clock = pygame.time.Clock()

        while running:
            # 1. イベント処理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:
                        self._step_forward()
                    elif event.key == pygame.K_LEFT:
                        self._step_backward()
                    elif event.key == pygame.K_ESCAPE:
                        running = False

            # 2. 描画
            self._draw()
            clock.tick(24)
        pygame.quit()

    def _step_forward(self):
        """1手進める"""
        if self.current_step < len(self.moves):
            color, pos = self.moves[self.current_step]
            # putメソッドを使うが、勝敗判定は無視して盤面更新だけ利用
            # (Boardの実装によってはputが例外を吐く可能性があるので直接代入でも可だが、put推奨)
            try:
                self.board.board[pos] = color
                # self.board.put(pos, color) # 勝敗判定ロジックが走るが描画だけなら直接代入でOK
            except:
                pass
            self.current_step += 1

    def _step_backward(self):
        """1手戻す"""
        if self.current_step > 0:
            self.current_step -= 1
            color, pos = self.moves[self.current_step]
            # Boardのpickメソッドを使って石を取り除く
            try:
                self.board.pick(pos)
            except:
                pass

    def _draw(self):
        # メッセージ作成
        msg = f"Step: {self.current_step} / {len(self.moves)}"
        if self.current_step == len(self.moves):
            msg += " (Finished)"

        # 直前の手（ハイライト用）
        last_move = None
        if self.current_step > 0:
            _, last_move = self.moves[self.current_step - 1]

        # レンダラーに描画させる
        self.renderer.draw_game(
            self.board,
            p1_name=self.p1_name,
            p2_name=self.p2_name,
            last_move=last_move,
            message=msg,
        )

        # 操作説明を追記
        hint = self.font.render(
            "[<-] Back  [->] Next  [ESC] Quit", True, (200, 200, 200)
        )
        self.renderer.screen.blit(hint, (20, 720 - 40))
        pygame.display.flip()
