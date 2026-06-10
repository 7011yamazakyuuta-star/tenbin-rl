"""
Phase C-3: 学習済みPPOモデルの評価

PPO エージェントを seat0 に固定し、TenbinEnv 上で各種対戦相手シナリオに対する
勝率・平均報酬・平均順位を測定する。結果を results/ppo_evaluation.json に保存する。

評価は train_ppo.py と同じ TenbinEnv を使うため、モデルは訓練時とまったく同じ
観測ベクトル（obs_dim = 5*num_players + 7）を受け取る。

使い方:
    python training/evaluate.py
    python training/evaluate.py --model training/models/ppo_tenbin_final --games 3000
"""

import argparse
import json
import os
import sys

import numpy as np

# training/ ディレクトリをパスに追加（スクリプト直接実行用）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import (
    AdaptiveAgent,
    FixedAgent,
    HistoryLevelKAgent,
    LevelKAgent,
    RandomAgent,
)
from tenbin_env import TenbinEnv

from stable_baselines3 import PPO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

NUM_PLAYERS = 4

# 相手 factory（rng -> agent）
F_LK3 = lambda r: LevelKAgent(3)            # 固定26（C-2勝率トップ）
F_HLK2 = lambda r: HistoryLevelKAgent(k=2)  # 動的適応型
F_ADAPT = lambda r: AdaptiveAgent()         # 目標値追従
F_FIX32 = lambda r: FixedAgent(32)          # Level-2固定
F_RAND = lambda r: RandomAgent(r)           # ランダム


def fixed_lineup(factories):
    """指定 factory 群から相手を生成する sampler を返す（固定編成）。"""
    def sampler(rng):
        return [f(np.random.default_rng(rng.integers(2**63))) for f in factories]
    return sampler


def pool_lineup(factories):
    """プールから (NUM_PLAYERS-1) 体を毎ゲーム復元抽出する sampler。"""
    def sampler(rng):
        idxs = rng.integers(0, len(factories), size=NUM_PLAYERS - 1)
        return [
            factories[int(i)](np.random.default_rng(rng.integers(2**63)))
            for i in idxs
        ]
    return sampler


# 評価シナリオ（PPO はつねに seat0）
SCENARIOS = {
    "vs LevelK(3) x3": fixed_lineup([F_LK3, F_LK3, F_LK3]),
    "vs HistLK(2) x3": fixed_lineup([F_HLK2, F_HLK2, F_HLK2]),
    "vs Adaptive x3": fixed_lineup([F_ADAPT, F_ADAPT, F_ADAPT]),
    "vs Fixed(32) x3": fixed_lineup([F_FIX32, F_FIX32, F_FIX32]),
    "vs Random x3": fixed_lineup([F_RAND, F_RAND, F_RAND]),
    "vs Top3 (LK3+HLK2+Adapt)": fixed_lineup([F_LK3, F_HLK2, F_ADAPT]),
    "vs Pool (random 3 of 5)": pool_lineup([F_LK3, F_HLK2, F_ADAPT, F_FIX32, F_RAND]),
}


def compute_rank(env):
    """エピソード終了後の env 状態から seat0（PPO）の順位を算出する。

    simulate.py / tournament.py の順位ロジックと同一。
    """
    num = env.num_players
    ranks = {}
    for idx, pidx in enumerate(env._elimination_order):
        ranks[pidx] = num - idx
    survivors = [(i, env._points[i]) for i in range(num) if env._alive[i]]
    survivors.sort(key=lambda x: -x[1])
    for off, (pidx, _) in enumerate(survivors):
        ranks[pidx] = off + 1
    return ranks.get(0, num)


def evaluate_scenario(model, sampler, n_games, seed):
    """1シナリオを n_games 回プレイし、勝率・平均報酬・平均順位を返す。"""
    rng = np.random.default_rng(seed)
    wins = 0
    rewards = []
    ranks = []
    for _ in range(n_games):
        opponents = sampler(rng)
        env = TenbinEnv(num_players=NUM_PLAYERS, opponents=opponents)
        obs, _ = env.reset(seed=int(rng.integers(2**31)))
        done = False
        total_r = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_r += reward
            done = terminated or truncated
        # ゲーム勝利 = PPO（seat0）が単独生存（env仕様）
        won = env._alive[0] and sum(env._alive) == 1
        if won:
            wins += 1
        rewards.append(total_r)
        ranks.append(compute_rank(env))
    return {
        "win_rate": round(wins / n_games, 4),
        "avg_reward": round(float(np.mean(rewards)), 3),
        "avg_rank": round(float(np.mean(ranks)), 3),
        "games": n_games,
    }


def main():
    parser = argparse.ArgumentParser(description="てんびん PPO 評価")
    parser.add_argument(
        "--model",
        default=os.path.join(MODELS_DIR, "ppo_tenbin_final"),
        help="モデルのパス（.zip は省略可）",
    )
    parser.add_argument("--games", type=int, default=3000, help="シナリオ毎のゲーム数")
    parser.add_argument("--seed", type=int, default=123, help="乱数シード")
    parser.add_argument("--device", default="cpu", help="推論デバイス（cpu / cuda）")
    args = parser.parse_args()

    if not os.path.exists(args.model) and not os.path.exists(args.model + ".zip"):
        raise SystemExit(
            f"モデルが見つかりません: {args.model}(.zip)\n"
            f"先に `python training/train_ppo.py` で訓練してください。"
        )

    print(f"モデルをロード: {args.model}")
    model = PPO.load(args.model, device=args.device)

    print(f"\n=== PPO 評価（各シナリオ {args.games:,} ゲーム）===\n")
    results = {}
    for name, sampler in SCENARIOS.items():
        stats = evaluate_scenario(model, sampler, args.games, args.seed)
        results[name] = stats
        print(
            f"  {name:28s}  勝率 {stats['win_rate']:6.1%}  "
            f"平均報酬 {stats['avg_reward']:7.2f}  平均順位 {stats['avg_rank']:.2f}"
        )

    summary = {
        "win_rate_vs_LevelK3": results["vs LevelK(3) x3"]["win_rate"],
        "win_rate_vs_HistLK2": results["vs HistLK(2) x3"]["win_rate"],
        "win_rate_vs_Adaptive": results["vs Adaptive x3"]["win_rate"],
        "win_rate_vs_Top3": results["vs Top3 (LK3+HLK2+Adapt)"]["win_rate"],
    }

    out = {
        "model": os.path.basename(args.model),
        "config": {
            "num_players": NUM_PLAYERS,
            "games_per_scenario": args.games,
            "seed": args.seed,
        },
        "scenarios": results,
        "summary": summary,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "ppo_evaluation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n結果を {out_path} に保存しました。")

    print("\n--- Phase C-3 で倒すべき目標との比較 ---")
    print(f"  vs LevelK(3)=26 : {summary['win_rate_vs_LevelK3']:.1%}")
    print(f"  vs HistLK(2)    : {summary['win_rate_vs_HistLK2']:.1%}")
    print(f"  vs Adaptive     : {summary['win_rate_vs_Adaptive']:.1%}")
    print(f"  vs Top3 混在     : {summary['win_rate_vs_Top3']:.1%}")
    print("  （いずれも PPO が seat0 で単独優勝した割合。4人戦の理論期待値は 25%）")


if __name__ == "__main__":
    main()
