"""
てんびん ラウンドロビン総当たりトーナメント

全エージェントの4人組み合わせを網羅し、勝率マトリクスと
簡易Eloレーティングを算出する。

使い方:
    python training/tournament.py
"""

import json
import os
import sys
import time
from itertools import combinations

import numpy as np

# training/ ディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import (
    AdaptiveAgent,
    BombThrowerAgent,
    CognitiveHierarchyAgent,
    FixedAgent,
    HistoryLevelKAgent,
    HumanLikeAgent,
    LevelKAgent,
    RandomAgent,
)
from simulate import simulate_game

# =============================================================================
# トーナメント用エージェント一覧
# =============================================================================

AGENT_DEFS = [
    ("Random", lambda rng: RandomAgent(rng)),
    ("Fixed(40)", lambda _: FixedAgent(40)),
    ("Fixed(32)", lambda _: FixedAgent(32)),
    ("LevelK(3)", lambda _: LevelKAgent(3)),
    ("CH(1.5)", lambda _: CognitiveHierarchyAgent(1.5)),
    ("Adaptive", lambda _: AdaptiveAgent()),
    ("HistLK(2)", lambda _: HistoryLevelKAgent(k=2)),
    ("Bomber", lambda rng: BombThrowerAgent(rng=rng)),
    ("HumanLike", lambda rng: HumanLikeAgent(rng=rng)),
]


# =============================================================================
# Elo レーティング
# =============================================================================

def update_elo(elo: dict, rankings: list[tuple[str, int]], k_factor: float = 4.0):
    """多人数ゲームの結果からペアワイズでEloを更新する。

    Args:
        elo: {エージェント名: レーティング} の辞書（in-place更新）
        rankings: [(名前, 順位), ...] のリスト（順位は1が最良）
        k_factor: 1ペアあたりのK係数
    """
    n = len(rankings)
    for i in range(n):
        for j in range(i + 1, n):
            name_i, rank_i = rankings[i]
            name_j, rank_j = rankings[j]

            # 期待スコア
            e_i = 1.0 / (1.0 + 10 ** ((elo[name_j] - elo[name_i]) / 400))

            # 実際のスコア
            if rank_i < rank_j:
                s_i = 1.0
            elif rank_i > rank_j:
                s_i = 0.0
            else:
                s_i = 0.5

            # 更新
            delta = k_factor * (s_i - e_i)
            elo[name_i] += delta
            elo[name_j] -= delta


# =============================================================================
# トーナメント実行
# =============================================================================

def run_tournament(
    agent_defs: list[tuple[str, callable]],
    games_per_matchup: int = 1000,
    seed: int = 42,
) -> dict:
    """ラウンドロビン総当たりトーナメントを実行する。

    Args:
        agent_defs: [(名前, ファクトリ関数), ...] のリスト
        games_per_matchup: 各組み合わせのゲーム数
        seed: マスター乱数シード

    Returns:
        dict: overall stats, pairwise win rates, elo ratings
    """
    rng = np.random.default_rng(seed)
    agent_names = [name for name, _ in agent_defs]
    n_agents = len(agent_names)

    # 統計初期化
    stats = {
        name: {"games": 0, "wins": 0, "ranks": []}
        for name in agent_names
    }
    pairwise_wins = {
        name: {other: 0 for other in agent_names if other != name}
        for name in agent_names
    }
    pairwise_games = {
        name: {other: 0 for other in agent_names if other != name}
        for name in agent_names
    }
    elo = {name: 1500.0 for name in agent_names}

    # 全4人組み合わせ
    combos = list(combinations(range(n_agents), 4))
    total_games = len(combos) * games_per_matchup
    print(f"組み合わせ数: {len(combos)}, 総ゲーム数: {total_games:,}\n")

    games_done = 0
    t0 = time.time()

    for combo in combos:
        combo_names = [agent_names[idx] for idx in combo]

        for _ in range(games_per_matchup):
            # エージェント生成
            agents = []
            for idx in combo:
                _, factory = agent_defs[idx]
                agent_rng = np.random.default_rng(rng.integers(2**63))
                agents.append(factory(agent_rng))

            result = simulate_game(agents)

            # 統計更新
            rankings = []
            for seat, idx in enumerate(combo):
                name = agent_names[idx]
                rank = result["ranks"].get(seat, 4)
                stats[name]["games"] += 1
                stats[name]["ranks"].append(rank)
                if rank == 1:
                    stats[name]["wins"] += 1
                rankings.append((name, rank))

            # ペアワイズ勝敗
            for i in range(4):
                for j in range(i + 1, 4):
                    name_i, rank_i = rankings[i]
                    name_j, rank_j = rankings[j]
                    pairwise_games[name_i][name_j] += 1
                    pairwise_games[name_j][name_i] += 1
                    if rank_i < rank_j:
                        pairwise_wins[name_i][name_j] += 1
                    elif rank_j < rank_i:
                        pairwise_wins[name_j][name_i] += 1

            # Elo 更新
            update_elo(elo, rankings)

            games_done += 1

        # 進捗表示
        elapsed = time.time() - t0
        pct = games_done / total_games * 100
        print(
            f"\r  進捗: {games_done:,}/{total_games:,} ({pct:.0f}%) "
            f"[{elapsed:.0f}秒]",
            end="", flush=True,
        )

    print(f"\n\n完了 ({time.time() - t0:.1f}秒)\n")

    # 勝率マトリクス
    win_rate_matrix = {}
    for name in agent_names:
        win_rate_matrix[name] = {}
        for other in agent_names:
            if name == other:
                win_rate_matrix[name][other] = None
            else:
                g = pairwise_games[name][other]
                win_rate_matrix[name][other] = (
                    round(pairwise_wins[name][other] / g, 4) if g > 0 else 0
                )

    # 全体統計
    overall = {}
    for name in agent_names:
        g = stats[name]["games"]
        overall[name] = {
            "games": g,
            "wins": stats[name]["wins"],
            "win_rate": round(stats[name]["wins"] / g, 4) if g > 0 else 0,
            "avg_rank": round(float(np.mean(stats[name]["ranks"])), 3),
            "elo": round(elo[name], 1),
        }

    return {
        "agents": agent_names,
        "overall": overall,
        "pairwise_win_rate": win_rate_matrix,
        "config": {
            "games_per_matchup": games_per_matchup,
            "total_games": total_games,
            "seed": seed,
        },
    }


# =============================================================================
# 表示
# =============================================================================

def print_results(results: dict):
    """トーナメント結果をコンソールに表形式で出力する。"""
    overall = results["overall"]
    agents = results["agents"]
    pw = results["pairwise_win_rate"]

    # === 勝率ランキング ===
    ranked = sorted(overall.items(), key=lambda x: -x[1]["elo"])
    print("=" * 72)
    print("  勝率ランキング（Elo順）")
    print("=" * 72)
    print(f"  {'#':>2}  {'Agent':14s}  {'Win%':>6s}  {'Elo':>6s}  {'AvgRank':>7s}  {'Games':>6s}")
    print("-" * 72)
    for i, (name, d) in enumerate(ranked, 1):
        print(
            f"  {i:2d}  {name:14s}  {d['win_rate']:5.1%}  "
            f"{d['elo']:6.0f}  {d['avg_rank']:7.2f}  {d['games']:6d}"
        )
    print()

    # === 勝率マトリクス ===
    # 短縮名（幅8文字に収める）
    short = {n: n[:8] for n in agents}
    ranked_names = [name for name, _ in ranked]

    print("=" * 72)
    print("  ペアワイズ勝率マトリクス（行が勝率）")
    print("=" * 72)
    header = "          " + "".join(f"{short[n]:>9s}" for n in ranked_names)
    print(header)
    print("-" * len(header))
    for name in ranked_names:
        row = f"  {short[name]:8s}"
        for other in ranked_names:
            if name == other:
                row += "      ---"
            else:
                wr = pw[name][other]
                row += f"    {wr:4.1%}" if wr is not None else "      N/A"
        print(row)
    print()


# =============================================================================
# メイン
# =============================================================================

def main():
    print("=== てんびん ラウンドロビントーナメント ===\n")
    results = run_tournament(AGENT_DEFS, games_per_matchup=1000)

    print_results(results)

    # JSON保存
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "tournament.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"結果を {output_path} に保存しました。")


if __name__ == "__main__":
    main()
