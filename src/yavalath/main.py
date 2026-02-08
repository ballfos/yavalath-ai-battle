from yavalath.benchmark import BenchmarkRunner
from yavalath.players.inoue.player import InouePlayer
from yavalath.players.kyawan.player import KyawanPlayer
from yavalath.players.random.player import RandomPlayer
from yavalath.replay import ReplayViewer


def main():
    # 1. プレイヤー準備
    p1 = InouePlayer("Inoue")
    p2 = RandomPlayer("Random Player")

    # 2. ベンチマーク実行 (100戦)
    print(">>> Running Benchmark...")
    runner = BenchmarkRunner(p1, p2, radius=4)
    replay_data = runner.run(num_games=10)

    # 3. リプレイ起動
    if replay_data.history:
        print(f"\n>>> Replaying the longest game ({len(replay_data.history)} moves)")
        print(">>> Press RIGHT arrow to proceed, LEFT to go back.")

        viewer = ReplayViewer(replay_data)
        viewer.run()
    else:
        print("No moves recorded.")


if __name__ == "__main__":
    main()
