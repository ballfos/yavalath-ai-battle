import random
from pathlib import Path

import numpy as np
import torch

from yavalath.core.board import Board, CellState, PutResult
from yavalath.core.player import Player
from yavalath.players.inoue.dqn import (
    DQN,
    build_action_space,
    encode_state,
    legal_action_mask,
    select_greedy_action,
)


class InouePlayer(Player):
    """ルールベースの吸着(勝利・防御)とDQNを組み合わせたプレイヤー。"""

    def __init__(self):
        super().__init__("Inoue", (0, 191, 255))
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._action_space = None
        # ファイルパスは環境に合わせて調整してください
        self._model_path = Path(__file__).with_name("model.pt")

    def _ensure_model(self, radius: int):
        if self._action_space is None:
            self._action_space = build_action_space(radius)

        if self._model is not None:
            return

        if not self._model_path.exists():
            # モデルがない場合はランダム動作などを許容するためNoneのままにする
            # print(f"Warning: Model file not found at {self._model_path}")
            self._model = None
            return

        board_size = 2 * radius + 1
        self._model = DQN(board_size=board_size, action_dim=self._action_space.size)
        try:
            state_dict = torch.load(self._model_path, map_location=self._device)
            self._model.load_state_dict(state_dict)
            self._model.to(self._device)
            self._model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            self._model = None

    def calc_best(self, board: Board, player: CellState) -> tuple[int, int, int]:
        """
        優先順位:
        1. 自分が勝てる手があるなら即採用
        2. 相手が次に勝つ手があるなら、その場所を塞ぐ (候補をその場所に限定)
        3. 自殺手 (3目) は除外する (ただし候補がなくなる場合は自殺手も許容)
        4. 残った候補からDQNで選択
        """
        empty_cells = list(board.get_empty_cells())
        if not empty_cells:
            raise ValueError("No empty cells available.")

        # --- Phase 1: 自分の即勝利チェック (Winning Move) ---
        for pos in empty_cells:
            result = board.put(pos, player)
            board.pick(pos)
            if self._is_win_result(result):
                return pos

        # --- Phase 2: 候補手の絞り込み (Blocking) ---
        candidates = list(empty_cells)

        opponent = self._opponent_of(player)

        threats = []
        for pos in empty_cells:
            # 相手の駒を置いてみる
            res = board.put(pos, opponent)
            board.pick(pos)

            if self._is_win_result(res):
                threats.append(pos)

        # もし脅威(相手の勝利手)があるなら、候補手は「脅威を塞ぐ手」のみに絞られる
        if threats:
            candidates = threats

        # --- Phase 3: 自殺手の除外 (Avoid Suicide) ---
        # candidates の中から「自分が打つとLOSE(3目)になる手」を除外する
        safe_candidates = []
        for pos in candidates:
            res = board.put(pos, player)
            board.pick(pos)

            if not self._is_lose_result(res):
                safe_candidates.append(pos)

        if safe_candidates:
            final_candidates = safe_candidates
        else:
            final_candidates = candidates

        # --- Phase 4: DQNによる選択 ---
        return self._choose_with_dqn(board, player, final_candidates)

    def _choose_with_dqn(
        self,
        board: Board,
        player: CellState,
        candidates: list[tuple[int, int, int]],
    ) -> tuple[int, int, int]:
        """指定された候補の中からDQNの評価値が最も高い手を選ぶ"""

        if len(candidates) == 1:
            return candidates[0]

        self._ensure_model(board.radius)
        if self._model is None:
            return random.choice(candidates)

        state = encode_state(board, player, self._device)

        # 全体の合法手マスク
        legal_mask = legal_action_mask(board, self._action_space)

        # candidates に含まれる手だけをTrueにするマスクを作成
        candidate_mask = np.zeros_like(legal_mask, dtype=bool)
        for pos in candidates:
            # action_space内に存在する場合のみマスクを有効化
            idx = self._action_space.pos_to_index.get(pos)
            if idx is not None:
                candidate_mask[idx] = True

        # 両方の条件を満たすマスク (盤面が空いている AND 候補に含まれる)
        final_mask = np.logical_and(legal_mask, candidate_mask)

        # マスクが空（念のための安全策）ならランダム
        if not np.any(final_mask):
            return random.choice(candidates)

        action_idx = select_greedy_action(self._model, state, final_mask)
        return self._action_space.positions[action_idx]

    def _is_win_result(self, result) -> bool:
        name = getattr(result, "name", None)
        if name is not None:
            return name == "WIN"
        value = getattr(result, "value", None)
        return value == PutResult.WIN.value

    def _is_lose_result(self, result) -> bool:
        name = getattr(result, "name", None)
        if name is not None:
            return name == "LOSE"
        value = getattr(result, "value", None)
        return value == PutResult.LOSE.value

    def _opponent_of(self, player):
        player_cls = getattr(player, "__class__", CellState)
        try:
            p1 = player_cls.PLAYER1
            p2 = player_cls.PLAYER2
        except Exception:
            p1 = CellState.PLAYER1
            p2 = CellState.PLAYER2

        p_value = getattr(player, "value", None)
        if p_value is None:
            return p2
        return p2 if p_value == p1.value else p1
