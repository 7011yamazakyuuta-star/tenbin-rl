"""
てんびん トーナメント結果の分析・可視化

tournament.json を読み込み、matplotlib でグラフを生成する。

使い方:
    python training/analyze.py
"""

import json
import os
import sys

import numpy as np

# matplotlib を非インタラクティブモードで使用
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


def load_results() -> dict:
    """tournament.json を読み込む。"""
    path = os.path.join(RESULTS_DIR, "tournament.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_win_rate_bar(results: dict):
    """各エージェントの勝率棒グラフを作成する。"""
    overall = results["overall"]
    # Elo順にソート
    items = sorted(overall.items(), key=lambda x: -x[1]["elo"])
    names = [n for n, _ in items]
    win_rates = [d["win_rate"] * 100 for _, d in items]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(names)))
    bars = ax.barh(range(len(names)), win_rates, color=colors)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Win Rate (%)", fontsize=11)
    ax.set_title("Agent Win Rates (Round-Robin Tournament)", fontsize=13, fontweight="bold")
    ax.invert_yaxis()

    # 値ラベル
    for bar, wr in zip(bars, win_rates):
        ax.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{wr:.1f}%", va="center", fontsize=9,
        )

    ax.set_xlim(0, max(win_rates) * 1.2 + 1)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "bar_chart.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> {path}")


def plot_heatmap(results: dict):
    """勝率マトリクスのヒートマップを作成する。"""
    overall = results["overall"]
    pw = results["pairwise_win_rate"]

    # Elo順にソート
    items = sorted(overall.items(), key=lambda x: -x[1]["elo"])
    names = [n for n, _ in items]
    n = len(names)

    matrix = np.zeros((n, n))
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if ni == nj:
                matrix[i, j] = 0.5  # 対角は0.5（自分自身）
            else:
                val = pw.get(ni, {}).get(nj)
                matrix[i, j] = val if val is not None else 0.5

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title("Pairwise Win Rate Matrix (row vs col)", fontsize=12, fontweight="bold")

    # セル内に数値表示
    for i in range(n):
        for j in range(n):
            if i == j:
                text = "-"
            else:
                text = f"{matrix[i, j]:.0%}"
            color = "white" if matrix[i, j] < 0.3 or matrix[i, j] > 0.7 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(im, ax=ax, label="Win Rate", shrink=0.8)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> {path}")


def plot_elo_ranking(results: dict):
    """Eloレーティングの順位表を作成する。"""
    overall = results["overall"]
    items = sorted(overall.items(), key=lambda x: -x[1]["elo"])
    names = [n for n, _ in items]
    elos = [d["elo"] for _, d in items]

    fig, ax = plt.subplots(figsize=(10, 5))

    # 1500を基準線に
    baseline = 1500
    deltas = [e - baseline for e in elos]
    colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in deltas]

    bars = ax.barh(range(len(names)), elos, color=colors)
    ax.axvline(x=baseline, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Elo Rating", fontsize=11)
    ax.set_title("Elo Ratings (Round-Robin Tournament)", fontsize=13, fontweight="bold")
    ax.invert_yaxis()

    # 値ラベル
    for bar, elo_val in zip(bars, elos):
        ax.text(
            bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
            f"{elo_val:.0f}", va="center", fontsize=9,
        )

    # x軸の余白
    ax.set_xlim(min(elos) - 50, max(elos) + 80)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "elo_ranking.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> {path}")


def main():
    print("=== てんびん トーナメント結果分析 ===\n")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    results = load_results()

    print("グラフ生成中...")
    plot_win_rate_bar(results)
    plot_heatmap(results)
    plot_elo_ranking(results)

    print(f"\n全グラフを {PLOTS_DIR}/ に保存しました。")


if __name__ == "__main__":
    main()
