from yavalath.benchmark import BenchmarkRunner
from yavalath.players.kyawan.player import KyawanPlayer
from yavalath.players.random.player import RandomPlayer
from yavalath.replay import ReplayViewer


def main():
    # 1. プレイヤー準備
    p1 = KyawanPlayer()
    p2 = RandomPlayer()

    # 2. ベンチマーク実行 (100戦)
    print(">>> Running Benchmark...")
    runner = BenchmarkRunner(p1, p2, radius=4)
    best_game_moves = runner.run(num_games=4)

    # 3. リプレイ起動
    if best_game_moves:
        print(f"\n>>> Replaying the longest game ({len(best_game_moves)} moves)")
        print(">>> Press RIGHT arrow to proceed, LEFT to go back.")

        viewer = ReplayViewer(best_game_moves, p1.name, p2.name, radius=4)
        viewer.run()
    else:
        print("No moves recorded.")


if __name__ == "__main__":
    main()
