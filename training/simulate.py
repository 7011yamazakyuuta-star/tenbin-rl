"""
てんびん エージェント対戦シミュレーション

複数のエージェント構成で1万ゲームを実行し、
勝率・平均順位・平均最終ポイントを集計して JSON に保存する。

使い方:
    python training/simulate.py
"""

import json
import os
import sys
import time

import numpy as np

# training/ ディレクトリをパスに追加（スクリプト直接実行用）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import (
    CognitiveHierarchyAgent,
    FixedAgent,
    LevelKAgent,
    RandomAgent,
)
from tenbin_env import process_round


# =============================================================================
# ゲームシミュレーション
# =============================================================================

def simulate_game(agents, max_rounds=200) -> dict:
    """1ゲームを完全にシミュレーションする。

    Args:
        agents: エージェントのリスト（len = プレイヤー数）
        max_rounds: 最大ラウンド数（無限ループ防止）

    Returns:
        dict: rounds, ranks, final_points, winner_indices
    """
    num_players = len(agents)
    points = [0] * num_players
    alive = [True] * num_players
    eliminated_count = 0
    elimination_order: list[int] = []
    round_num = 0

    while sum(alive) > 1 and round_num < max_rounds:
        round_num += 1

        # 生存プレイヤーの選択を収集
        choices: dict[int, int] = {}
        for i in range(num_players):
            if alive[i]:
                choices[i] = agents[i].act()

        # ラウンド処理（JSと同一ロジック）
        result = process_round(choices, alive, points, eliminated_count)

        # 状態更新
        points = result["new_points"]
        alive = result["new_alive"]
        eliminated_count = result["new_eliminated_count"]
        elimination_order.extend(result["newly_eliminated"])

    # ランキング算出
    ranks: dict[int, int] = {}
    # 脱落順にランク付け（最初の脱落者 = 最下位）
    for idx, player_idx in enumerate(elimination_order):
        ranks[player_idx] = num_players - idx

    # 生存者のランク（複数生存時はポイント順）
    survivors = [
        (i, points[i]) for i in range(num_players) if alive[i]
    ]
    survivors.sort(key=lambda x: -x[1])  # ポイント降順
    for rank_offset, (player_idx, _) in enumerate(survivors):
        ranks[player_idx] = rank_offset + 1

    return {
        "rounds": round_num,
        "ranks": ranks,
        "final_points": points,
        "winner_indices": [i for i in range(num_players) if ranks.get(i) == 1],
    }


def run_simulation(
    agent_configs: list[tuple[str, callable]],
    num_games: int = 10000,
    seed: int = 42,
) -> dict:
    """複数ゲームのシミュレーションを実行し統計を集計する。

    Args:
        agent_configs: [(名前, エージェント生成関数), ...] のリスト
        num_games: シミュレーションするゲーム数
        seed: マスター乱数シード

    Returns:
        dict: 各エージェントの統計（勝率、平均順位、平均最終ポイント）
    """
    rng = np.random.default_rng(seed)
    num_players = len(agent_configs)

    stats = {
        name: {"wins": 0, "ranks": [], "final_points": []}
        for name, _ in agent_configs
    }

    for _ in range(num_games):
        # 各ゲームごとにエージェントを新規生成
        agents = []
        for _, factory in agent_configs:
            agent_rng = np.random.default_rng(rng.integers(2**63))
            agents.append(factory(agent_rng))

        result = simulate_game(agents)

        for i, (name, _) in enumerate(agent_configs):
            rank = result["ranks"].get(i, num_players)
            stats[name]["ranks"].append(rank)
            stats[name]["final_points"].append(result["final_points"][i])
            if rank == 1:
                stats[name]["wins"] += 1

    # 集計
    summary = {}
    for name, data in stats.items():
        summary[name] = {
            "win_rate": round(data["wins"] / num_games, 4),
            "avg_rank": round(float(np.mean(data["ranks"])), 3),
            "avg_final_points": round(float(np.mean(data["final_points"])), 2),
            "games": num_games,
        }
    return summary


# =============================================================================
# メイン
# =============================================================================

# シミュレーション構成の定義
CONFIGURATIONS = [
    {
        "name": "Random vs Level1(40) vs Level2(32) vs LevelK(3)",
        "agents": [
            ("Random", lambda rng: RandomAgent(rng)),
            ("Level1(40)", lambda _: FixedAgent(40)),
            ("Level2(32)", lambda _: FixedAgent(32)),
            ("LevelK(3)", lambda _: LevelKAgent(3)),
        ],
    },
    {
        "name": "All Random (4 players)",
        "agents": [
            ("Random-A", lambda rng: RandomAgent(rng)),
            ("Random-B", lambda rng: RandomAgent(rng)),
            ("Random-C", lambda rng: RandomAgent(rng)),
            ("Random-D", lambda rng: RandomAgent(rng)),
        ],
    },
    {
        "name": "CH(1.5) vs Random vs Level1 vs Level2",
        "agents": [
            ("CH(1.5)", lambda _: CognitiveHierarchyAgent(1.5)),
            ("Random", lambda rng: RandomAgent(rng)),
            ("Level1(40)", lambda _: FixedAgent(40)),
            ("Level2(32)", lambda _: FixedAgent(32)),
        ],
    },
    {
        "name": "Level-k Tournament (k=0,1,2,3)",
        "agents": [
            ("LevelK(0)=50", lambda _: LevelKAgent(0)),
            ("LevelK(1)=40", lambda _: LevelKAgent(1)),
            ("LevelK(2)=32", lambda _: LevelKAgent(2)),
            ("LevelK(3)=26", lambda _: LevelKAgent(3)),
        ],
    },
    {
        "name": "CH variants (tau=0.5, 1.0, 1.5, 2.0)",
        "agents": [
            ("CH(0.5)", lambda _: CognitiveHierarchyAgent(0.5)),
            ("CH(1.0)", lambda _: CognitiveHierarchyAgent(1.0)),
            ("CH(1.5)", lambda _: CognitiveHierarchyAgent(1.5)),
            ("CH(2.0)", lambda _: CognitiveHierarchyAgent(2.0)),
        ],
    },
]


def main():
    """全構成でシミュレーションを実行し結果を保存する。"""
    num_games = 10000
    print(f"=== てんびん シミュレーション ({num_games:,} games/config) ===\n")

    all_results = {}

    for config in CONFIGURATIONS:
        name = config["name"]
        print(f"--- {name} ---")
        t0 = time.time()
        result = run_simulation(config["agents"], num_games=num_games)
        elapsed = time.time() - t0

        all_results[name] = result

        for agent_name, data in result.items():
            print(
                f"  {agent_name:20s}  "
                f"勝率: {data['win_rate']:6.1%}  "
                f"平均順位: {data['avg_rank']:.2f}  "
                f"平均最終pt: {data['avg_final_points']:6.1f}"
            )
        print(f"  ({elapsed:.1f}秒)\n")

    # 結果を JSON に保存
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "baseline.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"結果を {output_path} に保存しました。")


if __name__ == "__main__":
    main()
