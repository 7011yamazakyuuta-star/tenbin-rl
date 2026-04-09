"""
てんびん エージェント定義

基本エージェント（ランダム・固定値・Level-k・Cognitive Hierarchy）を提供する。
全エージェントは BaseAgent を継承し、act() メソッドで 0〜100 の整数を返す。
"""

from abc import ABC, abstractmethod
from math import exp, factorial

import numpy as np


class BaseAgent(ABC):
    """エージェントの抽象基底クラス。

    全てのエージェントはこのクラスを継承し、
    act() で 0〜100 の整数を返す必要がある。
    """

    @abstractmethod
    def act(self, observation=None) -> int:
        """行動（0〜100の整数）を返す。

        Args:
            observation: 観測ベクトル（不要なエージェントは無視してよい）
        """

    def reset(self):
        """エピソード開始時のリセット（必要な場合のみオーバーライド）。"""

    def observe(self, choices: dict, target: float, round_num: int):
        """ラウンド終了後に呼ばれる。動的エージェントはオーバーライドして履歴を更新する。

        Args:
            choices: {プレイヤーindex: 選択値} 生存プレイヤーの選択
            target: このラウンドの目標値（平均×0.8）
            round_num: ラウンド番号
        """


class RandomAgent(BaseAgent):
    """0〜100 の一様乱数を返すエージェント。

    JSの aiChoice('random') と同等: Math.floor(Math.random()*101)
    """

    def __init__(self, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()

    def act(self, observation=None) -> int:
        return int(self.rng.integers(0, 101))


class FixedAgent(BaseAgent):
    """常に固定値を返すエージェント。

    JSの aiChoice('level1')=40, aiChoice('level2')=32 に対応。
    """

    def __init__(self, value: int):
        self.value = int(np.clip(value, 0, 100))

    def act(self, observation=None) -> int:
        return self.value

    def __repr__(self):
        return f"FixedAgent({self.value})"


class LevelKAgent(BaseAgent):
    """Level-k 思考モデルエージェント。

    Level-k 理論に基づき、再帰的に最適応答を計算する:
      - Level 0: 50（一様乱数の期待値）
      - Level k: 50 × 0.8^k の四捨五入

    例:
      k=0 → 50, k=1 → 40, k=2 → 32, k=3 → 26, k=4 → 20
    """

    def __init__(self, k: int):
        self.k = k
        self.value = int(round(50 * (0.8 ** k)))

    def act(self, observation=None) -> int:
        return self.value

    def __repr__(self):
        return f"LevelKAgent(k={self.k}, value={self.value})"


class CognitiveHierarchyAgent(BaseAgent):
    """Camerer, Ho, & Chung (2004) の Poisson-CH モデルエージェント。

    ポアソン分布 Poisson(τ) に従うレベル分布を仮定し、
    各レベルのプレイヤーが下位レベルの行動分布に対して
    ベストレスポンスを選ぶ。最終的な行動は全レベルの
    加重平均（ポアソン重み）として算出する。

    Args:
        tau: ポアソン分布のパラメータ（平均思考レベル）
        max_level: 計算する最大レベル
    """

    def __init__(self, tau: float = 1.5, max_level: int = 10):
        self.tau = tau
        self.max_level = max_level
        self.value = self._compute_action()

    def _poisson_pmf(self, k: int, tau: float) -> float:
        """ポアソン確率質量関数 P(X=k)"""
        return exp(-tau) * (tau ** k) / factorial(k)

    def _compute_action(self) -> int:
        """全レベルの加重平均行動を計算する。"""
        weights = [
            self._poisson_pmf(k, self.tau) for k in range(self.max_level + 1)
        ]

        # 各レベルの最適行動を順に計算
        choices = [50.0]  # Level 0: 一様分布の期待値

        for k in range(1, self.max_level + 1):
            # レベル k は、レベル 0〜k-1 の分布に対してベストレスポンスを選ぶ
            belief_weights = weights[:k]
            total_belief = sum(belief_weights)

            if total_belief < 1e-15:
                choices.append(choices[-1])
                continue

            normalized = [w / total_belief for w in belief_weights]
            predicted_avg = sum(
                c * w for c, w in zip(choices[:k], normalized)
            )
            # ベストレスポンス = 予測平均 × 0.8
            choices.append(predicted_avg * 0.8)

        # ポアソン重みによる加重平均
        total_weight = sum(weights)
        expected = sum(c * w for c, w in zip(choices, weights)) / total_weight
        return int(round(expected))

    def act(self, observation=None) -> int:
        return self.value

    def __repr__(self):
        return f"CognitiveHierarchyAgent(tau={self.tau}, value={self.value})"


# =============================================================================
# 動的エージェント（Phase C-2 追加）
# =============================================================================


class AdaptiveAgent(BaseAgent):
    """適応型エージェント。

    過去 N ラウンドの目標値の移動平均 × 0.8 を選択する。
    相手の戦略変化に追従し、目標値が下がれば自分も下げる。

    Args:
        window: 移動平均のウィンドウサイズ（デフォルト3）
    """

    def __init__(self, window: int = 3):
        self.window = window
        self._targets: list[float] = []

    def reset(self):
        self._targets = []

    def observe(self, choices: dict, target: float, round_num: int):
        self._targets.append(target)

    def act(self, observation=None) -> int:
        if not self._targets:
            return 32  # 初回は Level-2 相当
        recent = self._targets[-self.window :]
        avg_target = sum(recent) / len(recent)
        return int(round(np.clip(avg_target * 0.8, 0, 100)))

    def __repr__(self):
        return f"AdaptiveAgent(window={self.window})"


class HistoryLevelKAgent(BaseAgent):
    """動的 Level-k エージェント。

    過去ラウンドの「全員の平均値」の移動平均を Level-0 の推定値とし、
    それに 0.8^k を掛けて選択を決定する。
    固定 Level-k と異なり、ゲーム中の動向に適応する。

    Args:
        k: 思考レベル（デフォルト2）
        window: 移動平均のウィンドウサイズ（デフォルト3）
    """

    def __init__(self, k: int = 2, window: int = 3):
        self.k = k
        self.window = window
        self._averages: list[float] = []

    def reset(self):
        self._averages = []

    def observe(self, choices: dict, target: float, round_num: int):
        if choices:
            avg = sum(choices.values()) / len(choices)
            self._averages.append(avg)

    def act(self, observation=None) -> int:
        if not self._averages:
            return int(round(50 * (0.8 ** self.k)))  # 初回は固定 Level-k
        recent = self._averages[-self.window :]
        level0_est = sum(recent) / len(recent)
        value = level0_est * (0.8 ** self.k)
        return int(round(np.clip(value, 0, 100)))

    def __repr__(self):
        return f"HistoryLevelKAgent(k={self.k}, window={self.window})"


class BombThrowerAgent(BaseAgent):
    """爆弾投下型エージェント（荒らし役）。

    通常は Level-2 相当の数字（32前後）を出すが、
    一定確率で「爆弾」として 60〜90 の高い数字を投げる。
    作品中のユウタの友人戦で見られた撹乱戦略を再現。

    Args:
        bomb_prob: 爆弾を投げる確率（デフォルト0.2）
        bomb_range: 爆弾の値の範囲（デフォルト (60, 90)）
        rng: 乱数生成器
    """

    def __init__(self, bomb_prob: float = 0.2, bomb_range: tuple = (60, 90), rng=None):
        self.bomb_prob = bomb_prob
        self.bomb_lo, self.bomb_hi = bomb_range
        self.rng = rng if rng is not None else np.random.default_rng()

    def act(self, observation=None) -> int:
        if self.rng.random() < self.bomb_prob:
            return int(self.rng.integers(self.bomb_lo, self.bomb_hi + 1))
        return 32

    def __repr__(self):
        return f"BombThrowerAgent(p={self.bomb_prob})"


class HumanLikeAgent(BaseAgent):
    """人間模倣エージェント。

    実験経済学 Nagel (1995) のデータに基づく人間の選択分布を模倣。
    平均 27、標準偏差 15 の正規分布からサンプリングし、0〜100 にクリップ。

    Args:
        mean: 正規分布の平均（デフォルト27）
        std: 正規分布の標準偏差（デフォルト15）
        rng: 乱数生成器
    """

    def __init__(self, mean: float = 27, std: float = 15, rng=None):
        self.mean = mean
        self.std = std
        self.rng = rng if rng is not None else np.random.default_rng()

    def act(self, observation=None) -> int:
        value = self.rng.normal(self.mean, self.std)
        return int(np.clip(round(value), 0, 100))

    def __repr__(self):
        return f"HumanLikeAgent(mean={self.mean}, std={self.std})"
