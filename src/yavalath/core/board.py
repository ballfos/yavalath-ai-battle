from enum import Enum

import numpy as np

# 3軸座標系における基底方向ベクトル
DIRECTIONS = [
    (1, -1, 0),
    (1, 0, -1),
    (0, 1, -1),
]


class CellState(Enum):
    EMPTY = 0
    PLAYER1 = 1
    PLAYER2 = 2

    def opposite(self):
        if self == CellState.PLAYER1:
            return CellState.PLAYER2
        elif self == CellState.PLAYER2:
            return CellState.PLAYER1
        else:
            raise ValueError("EMPTY cell has no opposite.")


class PutResult(Enum):
    CONTINUE = 0
    WIN = 1
    LOSE = -1


class Board:
    """
    六角形のセルからなる盤面を表すクラス

    Attributes:
        radius (int):
            盤面の半径
            盤面の1辺の長さは radius + 1 となる

        board (dict):
            盤面のセルの状態を保持する辞書
            3軸座標系 (x, y, z) をキー、CellState を値とする
            reference: https://www.redblobgames.com/grids/hexagons/#coordinates-cube
    """

    def __init__(self, radius: int):
        self.radius = radius
        self.reset()

    def reset(self):
        self.board: dict[tuple[int, int, int], CellState] = {}
        for x in range(-self.radius, self.radius + 1):
            y_min = max(-self.radius, -x - self.radius)
            y_max = min(self.radius, -x + self.radius)
            for y in range(y_min, y_max + 1):
                z = -x - y
                self.board[(x, y, z)] = CellState.EMPTY

    def get_empty_cells(self):
        return [pos for pos, state in self.board.items() if state == CellState.EMPTY]

    def can_put(
        self,
        position: tuple[int, int, int],
    ) -> bool:
        return self.board.get(position) == CellState.EMPTY

    def put(
        self,
        position: tuple[int, int, int],
        player: CellState,
    ) -> PutResult:
        """指定した位置に駒を置く"""
        if not self.can_put(position):
            raise ValueError("Cannot put at the specified position.")

        self.board[position] = player
        return self._check_local_win(position, player)

    def pick(
        self,
        position: tuple[int, int, int],
    ):
        """指定した位置の駒を取る"""
        if position not in self.board:
            raise ValueError("Position is out of bounds.")
        if self.board.get(position) == CellState.EMPTY:
            raise ValueError("Cannot pick from an empty cell.")

        self.board[position] = CellState.EMPTY

    def to_numpy(self, player: CellState) -> np.ndarray:
        """
        盤面を2次元のNumPy配列に変換

        :return:
            shape = (2, 2 * radius + 1, 2 * radius + 1)

            - channel 0: playerの駒の位置が1.0、その他が0.0
            - channel 1: 相手の駒の位置が1.0、その他が0.0
        """

        # 深層学習するならdtypeとかは好きに変えて
        size = 2 * self.radius + 1
        array = np.zeros((2, size, size), dtype=np.float32)

        for (x, y, z), state in self.board.items():
            if state == player:
                array[0, x + self.radius, y + self.radius] = 1.0
            elif state != CellState.EMPTY:
                array[1, x + self.radius, y + self.radius] = 1.0

        return array

    # === Helper Methods ===

    def _check_local_win(
        self,
        position: tuple[int, int, int],
        player: CellState,
    ) -> PutResult:
        """
        駒を置いたことによる勝利条件の判定

        :return:
            - WIN: positionに駒を置くことで4並びができた場合
            - LOSE: positionに駒を置くことで3並びができた場合
            - CONTINUE: それ以外の場合
        """

        made_3 = False
        made_4 = False

        for dx, dy, dz in DIRECTIONS:
            count = (
                1
                + self._count_in_direction(position, (dx, dy, dz), player)
                + self._count_in_direction(position, (-dx, -dy, -dz), player)
            )

            if count >= 4:
                made_4 = True
            elif count == 3:
                made_3 = True

        # 4と3の両方が成立する場合はWINを優先する
        if made_4:
            return PutResult.WIN
        elif made_3:
            return PutResult.LOSE
        else:
            return PutResult.CONTINUE

    def _count_in_direction(
        self,
        start: tuple[int, int, int],
        direction: tuple[int, int, int],
        player: CellState,
    ) -> int:
        """指定した方向に連続する駒の数を数える"""
        count = 0
        x, y, z = start
        dx, dy, dz = direction

        while True:
            x += dx
            y += dy
            z += dz

            target = self.board.get((x, y, z))
            if target is None or target == CellState.EMPTY:
                break

            if self.board.get((x, y, z)) == player:
                count += 1
            else:
                break

        return count
