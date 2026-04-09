"""
てんびん（ダイヤのキング） — Gymnasium互換ゲーム環境

index.html 内のJSロジックと完全に同じ動作をするPython実装。
コアロジック（process_round等）は独立関数として分離し、
pytestで個別テスト可能な構造にしている。
"""

from collections import Counter

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# =============================================================================
# コアロジック（純粋関数 — 状態を変更せず新しい値を返す）
# =============================================================================

def get_active_rules(eliminated_count: int) -> list[int]:
    """脱落者数に応じて有効な追加ルールのリストを返す。

    JSの getActiveRules() と同一ロジック:
      eliminatedCount >= 1 → ルール1（同値無効）
      eliminatedCount >= 2 → ルール2（ピタリ賞）
      eliminatedCount >= 3 → ルール3（0→100）
    """
    rules = []
    if eliminated_count >= 1:
        rules.append(1)
    if eliminated_count >= 2:
        rules.append(2)
    if eliminated_count >= 3:
        rules.append(3)
    return rules


def process_round(
    choices: dict[int, int],
    alive: list[bool],
    points: list[int],
    eliminated_count: int,
) -> dict:
    """1ラウンドの処理を行い結果を返す。

    index.html の processRound() と完全に同一のロジック。
    状態を直接変更せず、新しい points / alive / eliminated_count を返す。

    Args:
        choices: {プレイヤーindex: 選択値(0-100)} 生存プレイヤーのみ
        alive: 各プレイヤーの生存フラグ（変更しない）
        points: 各プレイヤーの現在ポイント（変更しない）
        eliminated_count: このラウンド開始時点の脱落者数

    Returns:
        dict:
            avg, target, winners, invalid_players, point_changes,
            exact_match, rule3_triggered, active_rules,
            newly_eliminated, new_points, new_alive, new_eliminated_count
    """
    alive_indices = [i for i, a in enumerate(alive) if a]

    # --- 平均値・目標値 ---
    values = [choices[i] for i in alive_indices]
    avg = sum(values) / len(values)
    target = avg * 0.8

    active_rules = get_active_rules(eliminated_count)

    winners: list[int] = []
    invalid_players: list[int] = []
    rule3_triggered = False
    exact_match = False

    # --- ルール3: 0→100 ---
    if 3 in active_rules:
        has_zero = any(choices[i] == 0 for i in alive_indices)
        pickers_100 = [i for i in alive_indices if choices[i] == 100]
        if has_zero and len(pickers_100) > 0:
            winners = pickers_100
            rule3_triggered = True

    if not rule3_triggered:
        # --- ルール1: 同値無効 ---
        if 1 in active_rules:
            counts = Counter(choices[i] for i in alive_indices)
            invalid_players = [
                i for i in alive_indices if counts[choices[i]] > 1
            ]

        # --- 有効プレイヤーから勝者を決定 ---
        valid_indices = [i for i in alive_indices if i not in invalid_players]
        if valid_indices:
            min_dist = min(abs(choices[i] - target) for i in valid_indices)
            winners = [
                i
                for i in valid_indices
                if abs(choices[i] - target) - min_dist < 0.0001
            ]

            # --- ルール2: ピタリ賞 ---
            if 2 in active_rules:
                exact_match = any(
                    abs(choices[i] - target) < 0.005 for i in winners
                )

    # --- スコアリング ---
    base_penalty = -2 if exact_match else -1
    point_changes: dict[int, int] = {}
    for i in alive_indices:
        point_changes[i] = 0 if i in winners else base_penalty

    # --- ポイント適用（新リストを作成） ---
    new_points = list(points)
    for i in alive_indices:
        new_points[i] += point_changes[i]

    # --- 脱落判定 ---
    new_alive = list(alive)
    newly_eliminated: list[int] = []
    new_eliminated_count = eliminated_count
    for i in alive_indices:
        if new_points[i] <= -10 and new_alive[i]:
            new_alive[i] = False
            newly_eliminated.append(i)
            new_eliminated_count += 1

    return {
        "avg": avg,
        "target": target,
        "winners": winners,
        "invalid_players": invalid_players,
        "point_changes": point_changes,
        "exact_match": exact_match,
        "rule3_triggered": rule3_triggered,
        "active_rules": active_rules,
        "newly_eliminated": newly_eliminated,
        "new_points": new_points,
        "new_alive": new_alive,
        "new_eliminated_count": new_eliminated_count,
    }


# =============================================================================
# Gymnasium 環境
# =============================================================================

class TenbinEnv(gym.Env):
    """てんびん（p-Beauty Contest）の Gymnasium 互換環境。

    エージェント（プレイヤー0）が 0〜100 の整数を選び、
    対戦相手（opponents）と共に p-Beauty Contest を行う。

    観測空間 (obs_dim = 5*num_players + 7):
        - 自分のポイント (/10 で正規化): 1
        - 全プレイヤーのポイント (/10): num_players
        - 生存フラグ: num_players
        - 有効ルール (3つの bool): 3
        - 過去3ラウンドの履歴: 3 × (num_players選択 + 目標値) = 3*(num_players+1)

    行動空間: Discrete(101) — 0〜100の整数

    報酬:
        - ラウンド勝利: +1
        - ラウンド敗北: 0
        - 最終優勝（最後の生存者）: +10
        - 脱落: -10
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, num_players=4, opponents=None, render_mode=None):
        super().__init__()
        self.num_players = num_players
        self.render_mode = render_mode

        # 対戦相手（未指定時は RandomAgent）
        if opponents is None:
            from training.agents import RandomAgent

            opponents = [RandomAgent() for _ in range(num_players - 1)]
        self.opponents = opponents
        assert len(opponents) == num_players - 1

        # 観測空間・行動空間
        self.obs_dim = 5 * num_players + 7
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(101)

        # 内部状態（reset で初期化）
        self._points: list[int] = []
        self._alive: list[bool] = []
        self._eliminated_count = 0
        self._round = 0
        self._history: list[tuple[dict, float]] = []
        self._elimination_order: list[int] = []
        self._done = False

    def reset(self, seed=None, options=None):
        """環境をリセットし、初期観測を返す。"""
        super().reset(seed=seed)
        self._points = [0] * self.num_players
        self._alive = [True] * self.num_players
        self._eliminated_count = 0
        self._round = 0
        self._history = []
        self._elimination_order = []
        self._done = False
        return self._get_obs(), {}

    def step(self, action):
        """1ラウンドを実行する。

        Args:
            action: エージェント（プレイヤー0）の選択 (0-100)

        Returns:
            observation, reward, terminated, truncated, info
        """
        action = int(np.clip(action, 0, 100))

        if self._done or not self._alive[0]:
            return self._get_obs(), 0.0, True, False, {}

        self._round += 1

        # 全プレイヤーの選択を収集
        choices: dict[int, int] = {}
        for i in range(self.num_players):
            if not self._alive[i]:
                continue
            if i == 0:
                choices[i] = action
            else:
                choices[i] = self.opponents[i - 1].act()

        # ラウンド処理
        result = process_round(
            choices, self._alive, self._points, self._eliminated_count
        )

        # 状態更新
        self._points = result["new_points"]
        self._alive = result["new_alive"]
        self._eliminated_count = result["new_eliminated_count"]
        self._elimination_order.extend(result["newly_eliminated"])
        self._history.append((choices, result["target"]))

        # ゲーム終了判定
        alive_count = sum(self._alive)
        terminated = alive_count <= 1 or not self._alive[0]
        self._done = terminated

        # 報酬計算
        reward = 0.0
        if 0 in result["winners"]:
            reward = 1.0  # ラウンド勝利

        if terminated:
            if self._alive[0] and alive_count == 1:
                reward += 10.0  # 最終優勝
            elif not self._alive[0]:
                reward = -10.0  # 脱落

        info = {"round": self._round, "result": result}
        return self._get_obs(), reward, terminated, False, info

    def _get_obs(self) -> np.ndarray:
        """現在の観測ベクトルを構築する。"""
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        idx = 0

        # 自分のポイント
        obs[idx] = self._points[0] / 10.0
        idx += 1

        # 全プレイヤーのポイント
        for i in range(self.num_players):
            obs[idx] = self._points[i] / 10.0
            idx += 1

        # 生存フラグ
        for i in range(self.num_players):
            obs[idx] = 1.0 if self._alive[i] else 0.0
            idx += 1

        # 有効ルール
        rules = get_active_rules(self._eliminated_count)
        for r in [1, 2, 3]:
            obs[idx] = 1.0 if r in rules else 0.0
            idx += 1

        # 過去3ラウンドの履歴（最新から順に）
        for r in range(3):
            hist_idx = len(self._history) - 1 - r
            if hist_idx >= 0:
                choices, target = self._history[hist_idx]
                for i in range(self.num_players):
                    obs[idx] = choices.get(i, -1) / 100.0
                    idx += 1
                obs[idx] = target / 100.0
                idx += 1
            else:
                # データなし — ゼロ埋め（初期化済み）
                idx += self.num_players + 1

        return obs

    def render(self):
        """現在の状態をテキストで表示する。"""
        lines = [f"=== ラウンド {self._round} ==="]
        rules = get_active_rules(self._eliminated_count)
        if rules:
            rule_names = {1: "同値無効", 2: "ピタリ賞", 3: "0→100"}
            lines.append(
                "有効ルール: " + ", ".join(rule_names[r] for r in rules)
            )
        for i in range(self.num_players):
            status = "生存" if self._alive[i] else "脱落"
            name = "あなた" if i == 0 else f"AI {i}"
            lines.append(f"  {name}: {self._points[i]}pt ({status})")
        text = "\n".join(lines)
        if self.render_mode == "human":
            print(text)
        return text
