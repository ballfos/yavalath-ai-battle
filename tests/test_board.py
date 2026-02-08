import pytest

from yavalath.core.board import Board, CellState, PutResult


@pytest.fixture
def board():
    # 半径2のボードでテスト（手狭でテストしやすい）
    return Board(radius=2)


def test_initialization(board):
    """初期化のテスト"""
    # 半径2の場合、一辺3マス。
    # セル数は 3*R*(R+1) + 1 = 3*2*3 + 1 = 19
    assert len(board.board) == 19
    assert len(board.get_empty_cells()) == 19


def test_put_and_can_put(board):
    """置く動作のテスト"""
    pos = (0, 0, 0)
    assert board.can_put(pos) is True

    result = board.put(pos, CellState.PLAYER1)

    assert result == PutResult.CONTINUE
    assert board.board[pos] == CellState.PLAYER1
    assert board.can_put(pos) is False


def test_win_condition(board):
    """4目並んだら勝ち"""
    # X X X _ の状態を作る
    board.board[(0, 0, 0)] = CellState.PLAYER1
    board.board[(1, -1, 0)] = CellState.PLAYER1
    board.board[(2, -2, 0)] = CellState.PLAYER1

    # 4つ目を置く
    result = board.put((-1, 1, 0), CellState.PLAYER1)
    assert result == PutResult.WIN


def test_lose_condition(board):
    """3目並んだら負け"""
    # X X _ の状態
    board.board[(0, 0, 0)] = CellState.PLAYER1
    board.board[(1, -1, 0)] = CellState.PLAYER1

    # 3つ目を置く
    result = board.put((-1, 1, 0), CellState.PLAYER1)
    assert result == PutResult.LOSE


def test_priority_win_over_lose(board):
    """4目と3目が同時にできたら勝ち優先"""
    #      X
    #      _
    # X X (X) X
    # この(X)を置くことで、縦に3、横に4ができる状況

    # 横方向 (あと1つで4目)
    board.board[(-1, 0, 1)] = CellState.PLAYER1
    board.board[(-2, 0, 2)] = CellState.PLAYER1
    board.board[(1, 0, -1)] = CellState.PLAYER1

    # 縦(斜め)方向 (あと1つで3目)
    board.board[(0, -1, 1)] = CellState.PLAYER1
    board.board[(0, -2, 2)] = CellState.PLAYER1  # 遠いところ

    # 中心に置く (0,0,0)
    # これで横は4連、縦(右下)は3連になるはず
    result = board.put((0, 0, 0), CellState.PLAYER1)

    assert result == PutResult.WIN


def test_five_in_a_row(board):
    """5目並んでも勝ち（4以上扱い）"""
    # _ X X X X _ の間に置くケースなど
    board.board[(-2, 2, 0)] = CellState.PLAYER1
    board.board[(-1, 1, 0)] = CellState.PLAYER1
    # (0,0,0) 空ける
    board.board[(1, -1, 0)] = CellState.PLAYER1
    board.board[(2, -2, 0)] = CellState.PLAYER1

    result = board.put((0, 0, 0), CellState.PLAYER1)
    assert result == PutResult.WIN


def test_to_numpy(board):
    """NumPy変換のテスト"""
    board.put((0, 0, 0), CellState.PLAYER1)
    board.put((1, -1, 0), CellState.PLAYER2)

    arr = board.to_numpy()

    assert arr.shape == (2, 5, 5)  # R=2 -> size=5

    # 中心 (0,0,0) -> インデックス (2,2)
    assert arr[0, 2, 2] == 1.0  # Player1
    assert arr[1, 2, 2] == 0.0

    # (1, -1, 0) -> インデックス (1+2, -1+2) = (3, 1)
    assert arr[0, 3, 1] == 0.0
    assert arr[1, 3, 1] == 1.0  # Player2


def test_out_of_bounds(board):
    """範囲外アクセス"""
    assert board.can_put((10, 10, -20)) is False
    with pytest.raises(ValueError):
        board.put((10, 10, -20), CellState.PLAYER1)
