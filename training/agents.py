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
