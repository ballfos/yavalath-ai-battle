from benchmark import BenchmarkRunner
from players.inoue.player import InouePlayer
from players.random.player import RandomPlayer
from replay import ReplayViewer


def main():
    # 1. プレイヤー準備
    p1 = InouePlayer("Inoue")
    p2 = RandomPlayer("Random Player")

    # 2. ベンチマーク実行 (100戦)
    print(">>> Running Benchmark...")
    runner = BenchmarkRunner(p1, p2, radius=4)
    best_game_moves, best_game_names = runner.run(num_games=100)

    # 3. リプレイ起動
    if best_game_moves:
        p1_name, p2_name = best_game_names or (p1.name, p2.name)
        print(f"\n>>> Replaying the longest game ({len(best_game_moves)} moves)")
        print(">>> Press RIGHT arrow to proceed, LEFT to go back.")

        viewer = ReplayViewer(best_game_moves, p1_name, p2_name, radius=4)
        viewer.run()
    else:
        print("No moves recorded.")


if __name__ == "__main__":
    main()
