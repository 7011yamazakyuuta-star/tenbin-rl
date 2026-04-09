# training/ — てんびん 強化学習訓練環境

てんびん（ダイヤのキング）の AI エージェントを訓練するための Python 環境。

## ファイル構成

| ファイル | 役割 |
|---|---|
| `tenbin_env.py` | Gymnasium 互換のゲーム環境。JS版と完全に同じロジック |
| `agents.py` | 全エージェント（静的5種 + 動的4種） |
| `simulate.py` | エージェント対戦シミュレーション（1万ゲーム×7構成） |
| `tournament.py` | ラウンドロビン総当たり戦 + Eloレーティング |
| `analyze.py` | トーナメント結果の可視化（matplotlib） |
| `requirements.txt` | Python パッケージ依存関係 |
| `results/` | シミュレーション結果・グラフの出力先 |

## セットアップ

```bash
pip install -r training/requirements.txt
```

## 実行

### ベースラインシミュレーション

```bash
python training/simulate.py
```

7つのエージェント構成で各1万ゲームを実行し、結果を `training/results/baseline.json` に保存します。

### ラウンドロビントーナメント

```bash
python training/tournament.py
```

全9エージェントの4人組み合わせ（C(9,4)=126通り）×1000ゲーム = 126,000ゲームを実行。勝率マトリクス・Eloレーティングを算出し、`training/results/tournament.json` に保存します。

### 結果の可視化

```bash
python training/analyze.py
```

`tournament.json` を読み込み、`training/results/plots/` に以下のグラフを生成:
- `bar_chart.png` — 勝率棒グラフ
- `heatmap.png` — ペアワイズ勝率マトリクス
- `elo_ranking.png` — Eloレーティング順位表

### Gymnasium 環境の使用例

```python
from training.tenbin_env import TenbinEnv
from training.agents import RandomAgent, FixedAgent, AdaptiveAgent

# 4人プレイ（エージェント + AI×3）
env = TenbinEnv(
    num_players=4,
    opponents=[RandomAgent(), FixedAgent(32), AdaptiveAgent()]
)

obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    env.render()
```

## エージェント一覧

### 静的エージェント（Phase C-1）

| エージェント | 戦略 | 出力例 |
|---|---|---|
| `RandomAgent` | 0〜100 一様乱数 | 0〜100 |
| `FixedAgent(v)` | 常に固定値 v | 40, 32 等 |
| `LevelKAgent(k)` | 50 × 0.8^k | k=1→40, k=2→32, k=3→26 |
| `CognitiveHierarchyAgent(τ)` | Poisson-CH モデル | τ=1.5→39 |

### 動的エージェント（Phase C-2）

| エージェント | 戦略 | 特徴 |
|---|---|---|
| `AdaptiveAgent(window)` | 過去N目標値の移動平均×0.8 | 相手の変化に追従 |
| `HistoryLevelKAgent(k, window)` | 過去N平均値×0.8^k | 動的Level-k |
| `BombThrowerAgent(prob)` | 通常32、確率pで60-90 | 撹乱・荒らし役 |
| `HumanLikeAgent(mean, std)` | N(27,15)からサンプリング | Nagel(1995)準拠 |

## Phase C-2 トーナメント結果

### Elo ランキング

| # | Agent | Win% | Elo | Avg Rank |
|---|---|---|---|---|
| 1 | HumanLike | 16.8% | 1951 | 2.39 |
| 2 | HistLK(2) | 57.8% | 1871 | 1.47 |
| 3 | LevelK(3) | 60.7% | 1765 | 1.48 |
| 4 | Adaptive | 52.3% | 1636 | 1.84 |
| 5 | Fixed(32) | 20.3% | 1582 | 2.29 |
| 6 | Bomber | 8.6% | 1423 | 2.88 |
| 7 | CH(1.5) | 3.8% | 1113 | 2.95 |
| 8 | Fixed(40) | 3.6% | 1082 | 3.39 |
| 9 | Random | 1.1% | 1078 | 3.81 |

### 主な知見

1. **HumanLike が Elo 1位、勝率は中位** — 勝率16.8%なのにElo最高。これはペアワイズ勝率が関係している：HumanLikeは上位エージェント（LevelK(3)やHistLK(2)）には負けるが、下位エージェント（Fixed(40)、CH(1.5)、Random）に対して80-86%の圧倒的勝率を持つ。「広く浅く勝つ」戦略がEloを押し上げた。

2. **HistLK(2) と LevelK(3) が実質的な最強** — 勝率はそれぞれ57.8%、60.7%で突出。特にLevelK(3)=26は固定値にもかかわらず、4人戦では最も目標値に近い選択をすることが多い。

3. **Adaptive は LevelK(3) と互角** — ペアワイズ50.4%でほぼ五分。Adaptiveはゲーム中に戦略を調整できるが、目標値に収束した後はLevelK(3)と同程度の値に落ち着く。

4. **BombThrower の弱さ** — 撹乱戦略は自分自身も巻き込む。爆弾（60-90）を投げると平均値が上がり、自身の通常選択値32が有利になるはずだが、爆弾ラウンドで大きく負けるため帳消しに。

5. **CH(1.5) の意外な弱さ** — 理論的に洗練されたモデルだが、出力値39はLevel1(40)に近く、その近辺はLevel2(32)やLevelK(3)=26に負ける。

### Phase C-3 で倒すべきターゲット

PPO で訓練する RL エージェントが倒すべき最強候補:
- **LevelK(3)** — 勝率トップ（60.7%）、固定値26で驚異的に強い
- **HistLK(2)** — 勝率2位（57.8%）かつ動的適応型。より手強い相手
- **Adaptive** — 適応型の代表格。RLエージェントの戦略変化に追従する

## 開発ロードマップ

| Phase | 内容 | 状態 |
|---|---|---|
| C-1 | Python訓練環境・ベースラインエージェント | ✅ 完了 |
| C-2 | 動的エージェント・トーナメント分析 | ✅ 完了 |
| C-3 | PPO による強化学習訓練 | 🔲 予定 |
| C-4 | ONNX 形式へのモデル変換 | 🔲 予定 |
| C-5 | PWA への ONNX モデル組み込み | 🔲 予定 |

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
