from benchmark import BenchmarkRunner
from players.random.player import RandomPlayer
from replay import ReplayViewer


def main():
    # 1. プレイヤー準備
    p1 = RandomPlayer("Random Player 1")
    p2 = RandomPlayer("Random Player 2")

    # 2. ベンチマーク実行 (100戦)
    print(">>> Running Benchmark...")
    runner = BenchmarkRunner(p1, p2, radius=4)
    best_game_moves = runner.run(num_games=100)

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
