import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from yavalath.core.board import Board, CellState, PutResult
from yavalath.core.player import Player
from yavalath.players.inoue.dqn import (
    DQN,
    build_action_space,
    encode_state,
    legal_action_mask,
    select_greedy_action,
)

# --- 定数設定 ---
SEARCH_START_EMPTY_COUNT = 14  # 空きマスがこれ以下なら探索開始
SEARCH_DEPTH_LIMIT = 4  # 何手先まで読むか
INF = 10000.0  # 勝利スコア基準


# --- ヘルパー関数 (エラー回避のためここに定義) ---
def mask_q_values(q_values: torch.Tensor, legal_mask: np.ndarray) -> torch.Tensor:
    """合法手以外のQ値を -1e9 にして選択されないようにする"""
    device = q_values.device
    mask_t = torch.from_numpy(legal_mask).to(device)
    masked_q = q_values.clone()

    # 次元に合わせてマスクを適用
    if masked_q.dim() == 2 and mask_t.dim() == 1:
        masked_q[:, ~mask_t] = -1e9
    else:
        masked_q[~mask_t] = -1e9
    return masked_q


class AInouePlayer(Player):
    """ルールベースの吸着(勝利・防御)とDQN、さらに終盤探索を組み合わせたプレイヤー。"""

    def __init__(self, name: str = "AInoue"):
        super().__init__(name)
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
        0. (終盤のみ) Alpha-Beta探索で必勝手順または最善手を探す
        1. 自分が勝てる手があるなら即採用
        2. 相手が次に勝つ手があるなら、その場所を塞ぐ (候補をその場所に限定)
        3. 自殺手 (3目) は除外する (ただし候補がなくなる場合は自殺手も許容)
        4. 残った候補からDQNで選択
        """
        empty_cells = list(board.get_empty_cells())
        if not empty_cells:
            raise ValueError("No empty cells available.")

        # モデルをロード (探索でもDQNを使うため早めにロード)
        self._ensure_model(board.radius)

        # --- Phase 0: 終盤探索モード (Alpha-Beta with DQN evaluation) ---
        # 空きマスが少ない場合、少し先の未来まで読んで判断する
        if len(empty_cells) <= SEARCH_START_EMPTY_COUNT:
            # print(f"Endgame Search: depth={SEARCH_DEPTH_LIMIT}, empty={len(empty_cells)}")
            best_move, _ = self._alpha_beta_search(
                board,
                player,
                depth=SEARCH_DEPTH_LIMIT,
                alpha=-INF * 2,
                beta=INF * 2,
            )
            if best_move is not None:
                return best_move

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

    # === Alpha-Beta Search Implementation ===

    def _alpha_beta_search(
        self,
        board: Board,
        player: CellState,
        depth: int,
        alpha: float,
        beta: float,
    ) -> tuple[tuple[int, int, int] | None, float]:
        """
        Alpha-Beta探索 (Negamax法)
        戻り値: (最善手, 評価値)
        """
        opponent = self._opponent_of(player)

        # 1. 候補手の取得
        legal_moves = list(board.get_empty_cells())
        if not legal_moves:
            return None, 0.0

        # 2. Move Ordering: DQNを使って候補手をソート
        # (有望な手を先に探索することで枝刈りを増やす)
        if self._model is not None:
            sorted_moves = self._sort_moves_by_dqn(board, player, legal_moves)
        else:
            sorted_moves = legal_moves

        best_move = sorted_moves[0]
        max_val = -INF * 2

        for pos in sorted_moves:
            # 手を打つ
            result = board.put(pos, player)

            val = 0.0
            is_terminal = False

            # 終端判定（既存の型安全メソッドを使用）
            if self._is_win_result(result):
                val = INF + depth  # 早く勝つほど高評価
                is_terminal = True
            elif self._is_lose_result(result):
                val = -INF - depth  # 早く負けるほど低評価（自殺手）
                is_terminal = True
            elif not board.get_empty_cells():  # 引き分け
                val = 0.0
                is_terminal = True

            if not is_terminal:
                if depth == 0:
                    # 探索深さ限界：DQNで盤面静的評価を行う
                    # 次は相手の手番なので、相手視点の評価値を出し、反転させる
                    val = self._evaluate_board_by_dqn(board, opponent)
                    val = -val  # Negamax: 相手の利益は自分の損失
                else:
                    # 再帰探索 (Negamax: -alpha, -beta を渡す)
                    _, opp_val = self._alpha_beta_search(
                        board, opponent, depth - 1, -beta, -alpha
                    )
                    val = -opp_val

            # 手を戻す (必須)
            board.pick(pos)

            # 更新処理
            if val > max_val:
                max_val = val
                best_move = pos

            alpha = max(alpha, val)
            if alpha >= beta:
                break  # Beta Cutoff

        return best_move, max_val

    def _sort_moves_by_dqn(
        self, board: Board, player: CellState, moves: list[tuple[int, int, int]]
    ) -> list[tuple[int, int, int]]:
        """候補手をDQNのQ値が高い順にソートする"""
        if not self._model:
            return moves

        state = encode_state(board, player, self._device)
        with torch.no_grad():
            # バッチサイズ1の出力を取得し、CPUへ
            q_values = self._model(state).cpu().numpy()[0]

        move_scores = []
        for pos in moves:
            idx = self._action_space.pos_to_index.get(pos)
            # マスに対応するQ値を取得。不明な場合は最低値
            score = q_values[idx] if idx is not None else -1e9
            move_scores.append((score, pos))

        # スコア降順にソート
        move_scores.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in move_scores]

    def _evaluate_board_by_dqn(self, board: Board, current_player: CellState) -> float:
        """
        現在の盤面をDQNで評価する。
        その盤面で打てる手の中で、最も高いQ値を返す (V(s) ≈ max Q(s, a))
        """
        if self._model is None:
            return 0.0

        state = encode_state(board, current_player, self._device)
        legal_mask = legal_action_mask(board, self._action_space)

        with torch.no_grad():
            q_values = self._model(state)
            # ここで定義した mask_q_values を使用
            q_values = mask_q_values(q_values, legal_mask)
            # 最大のQ値を返す (-1.0 ~ 1.0 の範囲を想定)
            return q_values.max().item()

    # === Helper Methods (Original) ===

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
