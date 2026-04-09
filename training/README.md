# training/ — てんびん 強化学習訓練環境

てんびん（ダイヤのキング）の AI エージェントを訓練するための Python 環境。

## ファイル構成

| ファイル | 役割 |
|---|---|
| `tenbin_env.py` | Gymnasium 互換のゲーム環境。JS版と完全に同じロジック |
| `agents.py` | ベースラインエージェント（Random, Fixed, Level-k, Cognitive Hierarchy） |
| `simulate.py` | エージェント対戦シミュレーション（1万ゲーム×5構成） |
| `requirements.txt` | Python パッケージ依存関係 |
| `results/` | シミュレーション結果の出力先 |

## セットアップ

```bash
pip install -r training/requirements.txt
```

## 実行

### ベースラインシミュレーション

```bash
python training/simulate.py
```

5つのエージェント構成で各1万ゲームを実行し、結果を `training/results/baseline.json` に保存します。

### Gymnasium 環境の使用例

```python
from training.tenbin_env import TenbinEnv
from training.agents import RandomAgent, FixedAgent

# 4人プレイ（エージェント + AI×3）
env = TenbinEnv(
    num_players=4,
    opponents=[RandomAgent(), FixedAgent(40), FixedAgent(32)]
)

obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # ランダム行動
    obs, reward, done, truncated, info = env.step(action)
    env.render()
```

## エージェント一覧

| エージェント | 戦略 | 出力例 |
|---|---|---|
| `RandomAgent` | 0〜100 一様乱数 | 0〜100 |
| `FixedAgent(v)` | 常に固定値 v | 40, 32 等 |
| `LevelKAgent(k)` | 50 × 0.8^k | k=1→40, k=2→32, k=3→26 |
| `CognitiveHierarchyAgent(τ)` | Poisson-CH モデル | τ=1.5→39 |

## 開発ロードマップ

| Phase | 内容 | 状態 |
|---|---|---|
| C-1 | Python訓練環境・ベースラインエージェント | ✅ 完了 |
| C-2 | PPO による強化学習訓練 | 🔲 予定 |
| C-3 | ONNX 形式へのモデル変換 | 🔲 予定 |
| C-4 | PWA への ONNX モデル組み込み | 🔲 予定 |

## テスト

コアロジック（`process_round`, `get_active_rules`）は純粋関数として分離されており、pytest で直接テスト可能です:

```python
from training.tenbin_env import process_round, get_active_rules

# ルール判定のテスト例
assert get_active_rules(0) == []
assert get_active_rules(1) == [1]
assert get_active_rules(3) == [1, 2, 3]

# ラウンド処理のテスト例
result = process_round(
    choices={0: 40, 1: 50, 2: 30, 3: 60},
    alive=[True, True, True, True],
    points=[0, 0, 0, 0],
    eliminated_count=0,
)
assert result["target"] == (40 + 50 + 30 + 60) / 4 * 0.8  # 36.0
```
