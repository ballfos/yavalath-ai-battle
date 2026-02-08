import math

import pygame

from yavalath.core.board import Board, CellState

# 色の定義
COLOR_BG = (40, 44, 52)  # 背景色 (ダークグレー)
COLOR_HEX = (171, 178, 191)  # 六角形の枠線
COLOR_HEX_FILL = (60, 64, 72)  # 六角形の塗りつぶし
COLOR_P1 = (229, 192, 123)  # Player 1 (Gold)
COLOR_P2 = (97, 175, 239)  # Player 2 (Blue)
COLOR_LAST_MOVE = (224, 108, 117)  # 直前の手のハイライト (Red)
TEXT_COLOR = (255, 255, 255)


class PygameRenderer:
    def __init__(self, screen: pygame.Surface, board_radius: int):
        self.screen = screen
        self.width = screen.get_width()
        self.height = screen.get_height()

        # 画面サイズに合わせて六角形のサイズを自動調整
        # 横幅は sqrt(3) * size * (2*R + 1) くらい必要
        self.hex_size = min(self.width, self.height) / (board_radius * 4.5)
        self.center_x = self.width // 2
        self.center_y = self.height // 2

        self.font = pygame.font.SysFont("Arial", 14)
        self.big_font = pygame.font.SysFont("Arial", 32, bold=True)

    def draw_game(
        self,
        board: Board,
        first_player_name: str,
        second_player_name: str,
        last_move=None,
        message="",
    ):
        self.screen.fill(COLOR_BG)

        # 1. 六角形グリッドと駒の描画
        # 辞書の中身を回すのではなく、座標計算して全マス描画する
        for pos, state in board.board.items():
            x, y, z = pos
            cx, cy = self._hex_to_pixel(x, z)

            # 六角形を描く
            self._draw_hexagon(cx, cy, COLOR_HEX, width=2)

            # 駒があれば描く
            if state == CellState.PLAYER1:
                self._draw_piece(cx, cy, COLOR_P1)
            elif state == CellState.PLAYER2:
                self._draw_piece(cx, cy, COLOR_P2)

            # 直前の手なら赤枠をつける
            if last_move == pos:
                pygame.draw.circle(
                    self.screen,
                    COLOR_LAST_MOVE,
                    (int(cx), int(cy)),
                    int(self.hex_size * 0.8),
                    3,
                )

            # デバッグ用に座標を表示（小さく）
            # text = self.font.render(f"{x},{y},{z}", True, (100, 100, 100))
            # self.screen.blit(text, (cx - 10, cy - 5))

        # 2. UI情報の描画
        if first_player_name and second_player_name:
            info_text = f"{first_player_name} (Gold) vs {second_player_name} (Blue)"
            self.screen.blit(
                self.big_font.render(info_text, True, TEXT_COLOR), (20, 20)
            )

        if message:
            msg_surf = self.big_font.render(message, True, COLOR_LAST_MOVE)
            rect = msg_surf.get_rect(center=(self.width // 2, self.height - 50))
            self.screen.blit(msg_surf, rect)

        pygame.display.flip()

    def _hex_to_pixel(self, x, z):
        # Pointy-topped (頂点が上) の変換式
        # xは右下、zは左方向
        px = self.center_x + self.hex_size * math.sqrt(3) * (x + z / 2)
        py = self.center_y + self.hex_size * (3 / 2 * z)
        return px, py

    def _draw_hexagon(self, cx, cy, color, width=0):
        points = []
        for i in range(6):
            angle_deg = 60 * i - 30  # -30度して尖った頂点を真上にする
            angle_rad = math.radians(angle_deg)
            px = cx + self.hex_size * math.cos(angle_rad)
            py = cy + self.hex_size * math.sin(angle_rad)
            points.append((px, py))

        # 塗りつぶし用
        if width > 0:
            pygame.draw.polygon(self.screen, COLOR_HEX_FILL, points)
        pygame.draw.polygon(self.screen, color, points, width)

    def _draw_piece(self, cx, cy, color):
        radius = int(self.hex_size * 0.7)
        # 影をつけて立体感を出す
        pygame.draw.circle(
            self.screen, (30, 30, 30), (int(cx) + 2, int(cy) + 2), radius
        )
        pygame.draw.circle(self.screen, color, (int(cx), int(cy)), radius)
