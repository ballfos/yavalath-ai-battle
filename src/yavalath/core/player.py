from abc import ABC, abstractmethod

from yavalath.core.board import Board, CellState


class Player(ABC):
    def __init__(
        self,
        name: str = "Anonymous",
        color: tuple[int, int, int] = (255, 255, 255),
    ):
        self.name = name
        self.color = color

    @abstractmethod
    def calc_best(self, board: Board, player: CellState) -> tuple[int, int, int]:
        """
        次の一手を計算して返す抽象メソッド

        Args:
            board (Board):
                現在の盤面

            player (CellState):
                自分のプレイヤー状態 (CellState.PLAYER1 or CellState.PLAYER2)

        Returns:
            tuple[int, int, int]:
                次に置く位置の3軸座標 (x, y, z)
        """
        pass
