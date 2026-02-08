import copy

from tqdm import tqdm

from yavalath.core.board import Board, CellState, PutResult
from yavalath.core.player import Player


class BenchmarkRunner:
    def __init__(self, p1: Player, p2: Player, radius=4):
        self.p1 = p1
        self.p2 = p2
        self.radius = radius
        self.stats = {
            p1.name: {"win": 0, "lose": 0, "draw": 0, "error": 0},
            p2.name: {"win": 0, "lose": 0, "draw": 0, "error": 0},
        }

    def run(self, num_games: int):
        longest_game_moves = []  # 最長試合の棋譜
        longest_game_len = -1

        print(f"Starting {num_games} matches: {self.p1.name} vs {self.p2.name}")

        for i in tqdm(range(num_games)):
            # 先手後手の入れ替え
            if i % 2 == 0:
                first, second = self.p1, self.p2
                first_color, second_color = CellState.PLAYER1, CellState.PLAYER2
            else:
                first, second = self.p2, self.p1
                first_color, second_color = CellState.PLAYER2, CellState.PLAYER1

            # 1試合実行
            winner_name, moves = self._play_one_game(
                first, second, first_color, second_color
            )

            # 統計更新
            self._update_stats(winner_name, first, second)

            # 最長試合の更新チェック
            if len(moves) > longest_game_len:
                longest_game_len = len(moves)
                longest_game_moves = moves

        self._print_results(num_games)
        return longest_game_moves

    def _play_one_game(
        self,
        p_first: Player,
        p_second: Player,
        c_first: CellState,
        c_second: CellState,
    ):
        board = Board(self.radius)

        # プレイヤーと色の対応
        players = {c_first: p_first, c_second: p_second}
        turn = c_first  # 先手から

        # 棋譜: [(CellState, (x, y, z)), ...]
        # リプレイ時に「誰がどこに置いたか」を復元するため色も保存
        move_history = []

        while True:
            current_player = players[turn]

            # 空きマスがない＝引き分け
            if not board.get_empty_cells():
                return "Draw", move_history

            try:
                board_copy = copy.deepcopy(board)
                pos = current_player.calc_best(board_copy, turn)

                # 置く (ルール違反ならValueErrorが出る -> 即負け)
                result = board.put(pos, turn)

                # 記録
                move_history.append((turn, pos))

            except Exception as e:
                # エラーを出したプレイヤーの負け
                print(f"\nError by {current_player.name}: {e}")
                winner = p_second if current_player == p_first else p_first
                return winner.name, move_history

            # 勝敗判定
            if result == PutResult.WIN:
                return current_player.name, move_history
            elif result == PutResult.LOSE:
                # 打った側が負け -> 相手の勝ち
                winner = p_second if current_player == p_first else p_first
                return winner.name, move_history

            # ターン交代
            turn = c_second if turn == c_first else c_first

    def _update_stats(self, winner_name, p1, p2):
        if winner_name == "Draw":
            self.stats[p1.name]["draw"] += 1
            self.stats[p2.name]["draw"] += 1
        elif winner_name == p1.name:
            self.stats[p1.name]["win"] += 1
            self.stats[p2.name]["lose"] += 1
        elif winner_name == p2.name:
            self.stats[p2.name]["win"] += 1
            self.stats[p1.name]["lose"] += 1

    def _print_results(self, num_games):
        print("\n" + "=" * 40)
        print(f"BENCHMARK RESULT ({num_games} games)")
        print(f"{'Player':<15} | Win | Lose | Draw | Rate")
        print("-" * 40)
        for name, res in self.stats.items():
            win = res["win"]
            rate = (win / num_games) * 100
            print(
                f"{name:<15} | {win:<3} | {res['lose']:<4} | {res['draw']:<4} | {rate:.1f}%"
            )
        print("=" * 40)
