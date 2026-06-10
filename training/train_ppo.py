"""
Phase C-3: PPO強化学習訓練

TenbinEnv 上で PPO エージェント（seat0）を訓練する。対戦相手はエピソード毎に
対戦相手プールから (num_players-1) 体をサンプリングするため、特定の相手に
過学習せず多様な戦略に対応できるようになる。

訓練後の評価は evaluate.py で行う。

使い方:
    python training/train_ppo.py                 # 50万ステップ訓練
    python training/train_ppo.py --timesteps 1000000
    python training/train_ppo.py --smoke         # スモークテストのみ
"""

import argparse
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
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")

NUM_PLAYERS = 4

# =============================================================================
# 対戦相手プール
# =============================================================================
# Phase C-2 トーナメントの上位＋ベースラインを混在させる。
# 各 factory は rng を受け取り 1 体のエージェントを返す。
# エピソード毎に (NUM_PLAYERS-1) 体を復元抽出するので、
# 「LevelK(3)×3」や「Adaptive+Random+Fixed」など多様な卓に対して訓練される。
OPPONENT_POOL = [
    ("LevelK(3)", lambda rng: LevelKAgent(3)),
    ("HistLK(2)", lambda rng: HistoryLevelKAgent(k=2)),
    ("Adaptive", lambda rng: AdaptiveAgent()),
    ("Fixed(32)", lambda rng: FixedAgent(32)),
    ("Random", lambda rng: RandomAgent(rng)),
]


# =============================================================================
# 相手をランダム化する環境ラッパー
# =============================================================================

class RandomOpponentEnv(TenbinEnv):
    """エピソード毎に対戦相手をプールからサンプリングする TenbinEnv。

    TenbinEnv は __init__ 時に相手を固定するため、reset() のたびに
    self.opponents をプールから引き直すことで、多様な相手に対して訓練する。
    既存の tenbin_env.py には一切手を加えない（サブクラスで上書きするだけ）。
    """

    def __init__(self, pool, num_players=NUM_PLAYERS, seed=None, **kwargs):
        self._pool = pool
        self._np_rng = np.random.default_rng(seed)
        self._n_opponents = num_players - 1
        # 初期相手を1組サンプリングして親クラスを初期化
        opponents = self._sample_opponents()
        super().__init__(num_players=num_players, opponents=opponents, **kwargs)

    def _sample_opponents(self):
        """プールから (num_players-1) 体を独立復元抽出で生成する。"""
        idxs = self._np_rng.integers(0, len(self._pool), size=self._n_opponents)
        opponents = []
        for i in idxs:
            _, factory = self._pool[int(i)]
            agent_rng = np.random.default_rng(self._np_rng.integers(2**63))
            opponents.append(factory(agent_rng))
        return opponents

    def reset(self, seed=None, options=None):
        # 相手を引き直してから通常のリセット（親が opp.reset() を呼ぶ）
        self.opponents = self._sample_opponents()
        return super().reset(seed=seed, options=options)


def make_env(pool, seed):
    """VecEnv 用の環境生成クロージャを返す。"""

    def _init():
        env = RandomOpponentEnv(pool, num_players=NUM_PLAYERS, seed=seed)
        return Monitor(env)

    return _init


def build_model(venv, seed, tensorboard_log=None, device="cpu"):
    """RESUME 準拠のハイパーパラメータで PPO を構築する。"""
    return PPO(
        "MlpPolicy",
        venv,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        seed=seed,
        device=device,  # 小さなMLP方策はCPUの方が速い（SB3公式推奨）。--device で変更可
        tensorboard_log=tensorboard_log,
    )


# =============================================================================
# スモークテスト
# =============================================================================

def smoke_test():
    """本番訓練の前に環境とPPOが例外なく動くことを確認する。"""
    print("[smoke] 環境の reset / step を確認 ...")
    env = RandomOpponentEnv(OPPONENT_POOL, seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (env.obs_dim,), f"obs shape mismatch: {obs.shape}"
    for _ in range(20):
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs), "obs が観測空間外"
        if terminated or truncated:
            obs, _ = env.reset()

    print("[smoke] 短時間の PPO 学習を確認 ...")
    venv = DummyVecEnv([make_env(OPPONENT_POOL, seed=0)])
    model = PPO(
        "MlpPolicy", venv, n_steps=256, batch_size=64,
        device="cpu", seed=0, verbose=0,
    )
    model.learn(total_timesteps=512)

    # predict も確認
    obs, _ = env.reset(seed=1)
    action, _ = model.predict(obs, deterministic=True)
    assert 0 <= int(action) <= 100, f"action 範囲外: {action}"
    print("[smoke] OK — reset/step/learn/predict すべて正常")


# =============================================================================
# メイン
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="てんびん PPO 訓練")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="総訓練ステップ数（デフォルト 500,000）")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="並列環境数（デフォルト 4）")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    parser.add_argument("--device", default="cpu",
                        help="訓練デバイス（cpu / cuda）。MLP方策はCPU推奨")
    parser.add_argument("--smoke", action="store_true",
                        help="スモークテストのみ実行して終了")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    if args.smoke:
        smoke_test()
        return

    print("=== てんびん PPO 訓練 ===")
    print(f"  総ステップ数 : {args.timesteps:,}")
    print(f"  並列環境数   : {args.n_envs}")
    print(f"  相手プール   : {[name for name, _ in OPPONENT_POOL]}")
    print(f"  デバイス     : {args.device}\n")

    # ベクトル化環境（環境ごとに別シードで相手サンプリング）
    venv = DummyVecEnv([
        make_env(OPPONENT_POOL, seed=args.seed + i) for i in range(args.n_envs)
    ])
    # 注: 純Pythonの軽量環境のため DummyVecEnv で十分。
    #     より高速化したい場合は SubprocVecEnv に差し替え可能。

    model = build_model(venv, seed=args.seed, tensorboard_log=LOGS_DIR, device=args.device)

    # 10万ステップごとにチェックポイント保存（save_freq は env あたりのステップ数）
    checkpoint_cb = CheckpointCallback(
        save_freq=max(100_000 // args.n_envs, 1),
        save_path=MODELS_DIR,
        name_prefix="ppo_tenbin",
    )

    model.learn(total_timesteps=args.timesteps, callback=checkpoint_cb)

    final_path = os.path.join(MODELS_DIR, "ppo_tenbin_final")
    model.save(final_path)
    print(f"\n最終モデルを {final_path}.zip に保存しました。")
    print("評価は `python training/evaluate.py` で実行してください。")


if __name__ == "__main__":
    main()
