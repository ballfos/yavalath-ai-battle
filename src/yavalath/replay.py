import sys
from dataclasses import dataclass

import pygame

from yavalath.core.board import Board, CellState
from yavalath.core.player import Player
from yavalath.renderer import PygameRenderer  # 前回作成した描画クラスを流用


@dataclass
class ReplayData:
    p1: Player
    p2: Player
    first_player: CellState
    history: list[tuple[int, int, int]]
    radius: int


class ReplayViewer:
    def __init__(self, replay_data: ReplayData):
        """
        move_history: [(CellState, (x, y, z)), ...] のリスト
        """
        self.replay_data = replay_data
        self.board = Board(replay_data.radius)
        self.current_step = 0  # 現在何手目まで表示しているか

        # Pygame setup
        pygame.init()
        screen = pygame.display.set_mode((1280, 720))
        pygame.display.set_caption("Yavalath Replay Viewer")
        self.renderer = PygameRenderer(screen, replay_data.radius)

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
        """current_stepの手を適用してから1手進める"""
        if self.current_step < len(self.replay_data.history):
            player = self.replay_data.first_player
            if self.current_step % 2 == 1:
                player = player.opposite()
            pos = self.replay_data.history[self.current_step]
            try:
                self.board.put(pos, player)
            except:
                pass
            self.current_step += 1

    def _step_backward(self):
        """1手戻してからcurrent_stepの手を取り除く"""
        if self.current_step > 0:
            self.current_step -= 1
            player = self.replay_data.first_player
            if self.current_step % 2 == 1:
                player = player.opposite()
            pos = self.replay_data.history[self.current_step]
            try:
                self.board.pick(pos)
            except:
                pass

    def _draw(self):
        # メッセージ作成
        msg = f"Step: {self.current_step} / {len(self.replay_data.history)}"
        if self.current_step == len(self.replay_data.history):
            msg += " (Finished)"

        # 直前の手（ハイライト用）
        last_move = None
        if self.current_step > 0:
            last_move = self.replay_data.history[self.current_step - 1]

        # レンダラーに描画させる
        if self.replay_data.first_player == CellState.PLAYER1:
            first_player = self.replay_data.p1
            second_player = self.replay_data.p2
        else:
            first_player = self.replay_data.p2
            second_player = self.replay_data.p1
        self.renderer.draw_game(
            self.board,
            first_player,
            second_player,
            last_move=last_move,
            message=msg,
        )

        # 操作説明を追記
        hint = self.font.render(
            "[<-] Back  [->] Next  [ESC] Quit", True, (200, 200, 200)
        )
        self.renderer.screen.blit(hint, (20, 720 - 40))
        pygame.display.flip()
