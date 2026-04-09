# RESUME.md — てんびんプロジェクト 再開ガイド

> **最終更新**: 2026-04-09
> **最新コミット**: `3482554` (Phase B-3 完了)
> **次のステップ**: Phase C-3（PPO強化学習訓練）

---

## 1. 現在の進捗状況

| Phase | 内容 | コミット | 状態 |
|---|---|---|---|
| A | PWA本体（index.html + manifest + SW + icons） | `263409e` | ✅ 完了 |
| B | UI調整（v0.1 BETAバッジ、AI開発中注釈） | `d7f3141` | ✅ 完了 |
| B-2 | 多言語対応、AIキャラ選択、モード選択、Pass-and-play | `bad4772` | ✅ 完了 |
| B-3 | 動物キャラ刷新、ゲーム中断機能、ポーズ/ルールモーダル | `3482554` | ✅ 完了 |
| C-1 | Python訓練環境（Gymnasium環境 + 静的エージェント5種 + シミュレーション） | `4390c10` | ✅ 完了 |
| C-2 | 動的エージェント4種 + ラウンドロビントーナメント + 分析可視化 | `e977b3e` | ✅ 完了 |
| C-3 | PPO強化学習訓練 | — | 🔲 **次はここ** |
| C-4 | ONNX変換 | — | 🔲 予定 |
| C-5 | PWAへのONNXモデル組み込み | — | 🔲 予定 |

### 各フェーズで作ったもの

**Phase A（PWA本体）**
- `index.html` — 全画面SPA（タイトル/ゲーム/結果/ゲームオーバー/戦績）
- `manifest.json` + `service-worker.js` — オフライン対応PWA
- `icons/` — SVGアプリアイコン（ダイヤ+てんびん+K）
- ゲームロジック: 平均×0.8目標値、3種の追加ルール、localStorage戦績保存

**Phase B（UI調整）**
- タイトル画面に `v0.1 BETA - AI開発中` バッジ
- ゲーム画面に「※現在のAIは簡易版」注釈

**Phase B-2（多言語・キャラ選択・マルチプレイヤー基盤）**
- 多言語対応（日本語/英語）: ブラウザ言語自動検出、🌐ボタンで切替、localStorageに保存
- AIキャラクター選択（スマブラ方式）: 7体のキャラからスロット割当、全員ランダム、ランダム演出
- モード選択画面: vs AI / みんなで遊ぶ(Pass-and-play) / オンライン対戦(Coming Soon) / 戦績
- Pass-and-play モード: プライバシー画面付きの1台回し遊び
- 戦績タブ: AI戦とPnPを分離管理

**Phase B-3（キャラクター刷新・ゲーム中断機能）**
- 全7体のAIキャラを動物に全面刷新（戦略は非公開、ミニマル表示）
- 🏠ホームボタン・⏸️ポーズボタン（44x44px、AI/PnP両対応）
- 中断確認モーダル（日英対応）
- ポーズオーバーレイ: 再開 / ルール確認 / ホームに戻る
- ルール確認モーダル: タイトル画面・ポーズメニューから開ける
- 結果画面に「今日はここまで」ボタン（戦績未記録で中断）
- ゲームオーバー画面: もう一度 / キャラ選択に戻る / ホームの3ボタン

**Phase C-1（Python訓練環境）**
- `training/tenbin_env.py` — `process_round()` 純粋関数 + `TenbinEnv(gym.Env)`
- `training/agents.py` — RandomAgent, FixedAgent, LevelKAgent, CognitiveHierarchyAgent
- `training/simulate.py` — 1万ゲーム×5構成のシミュレーション

**Phase C-2（動的エージェント + トーナメント）**
- `training/agents.py` に追加: AdaptiveAgent, HistoryLevelKAgent, BombThrowerAgent, HumanLikeAgent
- `training/tournament.py` — 9エージェント×C(9,4)=126組×1000ゲーム=12.6万ゲーム + Elo算出
- `training/analyze.py` — matplotlib 可視化（棒グラフ/ヒートマップ/Elo順位表）
- `training/results/tournament.json` + `training/results/plots/` にグラフ3枚

---

## 2. キャラクター一覧（PWA ↔ 訓練環境の対応）

| PWA上の名前 | 絵文字 | レベル | PWA内部ID | 訓練環境の対応エージェント | 固定値/戦略 |
|---|---|---|---|---|---|
| ウサギ / Rabbit | 🐰 | ⭐1 | `randomizer` | RandomAgent | 0〜100ランダム |
| フクロウ / Owl | 🦉 | ⭐2 | `textbook` | FixedAgent(40) / LevelK(1) | 常に40 |
| カメ / Turtle | 🐢 | ⭐3 | `equilibrist` | FixedAgent(32) / LevelK(2) | 常に32 |
| タコ / Octopus | 🐙 | ⭐5 | `oracle` | LevelK(3) | 常に26（勝率60.7%） |
| カメレオン / Chameleon | 🦎 | ⭐4 | `adapter` | AdaptiveAgent | 直近目標値の移動平均×0.8 |
| ゴリラ / Gorilla | 🦍 | ⭐3 | `bomber` | BombThrowerAgent | 通常32、20%で60〜90の爆弾 |
| サル / Monkey | 🐵 | ⭐3 | `humanoid` | HumanLikeAgent | 正規分布(平均27, σ15) |

### Phase C-2 トーナメント結果との対応

| 訓練環境 Agent | PWAキャラ | 勝率 | Elo | 備考 |
|---|---|---|---|---|
| LevelK(3) = 26 | 🐙 タコ | **60.7%** | 1765 | 固定値なのに勝率トップ |
| HistLK(2) | （未実装） | 57.8% | 1871 | C-3でPPOが倒すべき目標 |
| Adaptive | 🦎 カメレオン | 52.3% | 1636 | 目標値追従型 |
| HumanLike | 🐵 サル | 16.8% | **1951** | Eloパラドックス（勝率低いがElo1位） |
| Fixed(32) | 🐢 カメ | 20.3% | 1582 | Level-2固定 |
| BombThrower | 🦍 ゴリラ | 9.3% | 1423 | 爆弾で場を荒らす |

---

## 3. Phase C-2 までの主要な発見

### Elo ランキング（126,000ゲームの総当たり結果）

| # | Agent | 勝率 | Elo | 平均順位 | 特徴 |
|---|---|---|---|---|---|
| 1 | HumanLike | 16.8% | **1951** | 2.39 | Elo1位だが勝率は中位（パラドックス） |
| 2 | HistLK(2) | **57.8%** | 1871 | **1.47** | 動的適応型で実質最強 |
| 3 | LevelK(3) | **60.7%** | 1765 | 1.48 | 固定値26なのに勝率トップ |
| 4 | Adaptive | 52.3% | 1636 | 1.84 | 目標値追従型 |
| 5 | Fixed(32) | 20.3% | 1582 | 2.29 | JSのLevel2に対応 |
| 6-9 | Bomber/CH/Fixed(40)/Random | 1-9% | 1078-1423 | 2.88-3.81 | 弱い |

### ガクチカ向け考察ポイント

**HumanLikeのEloパラドックス**
- 勝率16.8%（6位相当）なのにElo 1位。理由: 下位エージェント（Random, Fixed(40), CH(1.5)等）に80-86%で圧勝するが、上位3強にはほぼ勝てない。Eloはペアワイズ計算なので「広く浅く勝つ」パターンが有利に働いた。
- → ゲーム理論で言う「支配戦略がない」状況の実例。指標の選び方で「最強」が変わる。

**固定値26の驚異的な強さ**
- LevelK(3)=26は何も考えずに26を出し続けるだけで勝率60.7%。4人の平均値（大体30-40）×0.8＝24-32 の範囲に常に収まるため。
- → 「複雑な適応戦略が必ずしも単純な固定戦略に勝てない」というゲーム理論の教訓。

**Adaptive vs HistLK(2) のペアワイズ0.0%**
- Adaptiveは「目標値の移動平均×0.8」、HistLK(2)は「平均値の移動平均×0.8^2」。後者の方が1段深い推論をしているため、常にAdaptiveを下回る値を出せる。
- → Level-kの思考の深さが勝敗を決定的に分ける事例。

### Phase C-3 で倒すべき目標

1. **LevelK(3)=26（🐙タコ）** — 勝率60.7%。固定値なので「これに勝てないRLは話にならない」という最低ライン
2. **HistLK(2)** — 勝率57.8%。動的適応型。RLが本当に賢いかのテスト
3. **Adaptive（🦎カメレオン）** — 勝率52.3%。RLの行動変化に追従してくるので、単純なパターンは通じない

---

## 4. Colab 環境の再セットアップ手順

### 新しいノートブックでゼロから再開する場合

```python
# ===== セル1: リポジトリのクローン =====
from google.colab import userdata
import os

# GitHubトークン（Colabシークレット名: GITHUB_TOKEN_TENBIN）
token = userdata.get('GITHUB_TOKEN_TENBIN')
os.environ['GITHUB_TOKEN'] = token

!git clone https://{token}@github.com/7011yamazakyuuta-star/tenbin-rl.git /content/tenbin-rl
%cd /content/tenbin-rl
```

```python
# ===== セル2: Python依存パッケージのインストール =====
!pip install -r training/requirements.txt -q
```

```python
# ===== セル3: 動作確認（オプション） =====
!python training/simulate.py
```

```python
# ===== セル4: Claude Codeのインストールと起動 =====
!npm install -g @anthropic-ai/claude-code

# Anthropic APIキー（Colabシークレット名を確認して設定）
os.environ['ANTHROPIC_API_KEY'] = userdata.get('ANTHROPIC_API_KEY')

# Claude Code 起動
!claude
```

### 必要な環境変数 / シークレット

| 変数名 | 用途 | 設定場所 |
|---|---|---|
| `GITHUB_TOKEN_TENBIN` | GitHubへのpush/pull | Colabシークレット |
| `ANTHROPIC_API_KEY` | Claude Code API認証 | Colabシークレット |

### Git設定（初回のみ）

```bash
git config --global user.name "7011yamazakyuuta-star"
git config --global user.email "your-email@example.com"
```

---

## 5. Phase C-3 開始時の手順

### Step 1: Colab ランタイムを T4 GPU に切り替え

1. Colab メニュー → 「ランタイム」→「ランタイムのタイプを変更」
2. ハードウェアアクセラレータ → **T4 GPU** を選択
3. 「保存」→ ランタイムが再起動される
4. 上記セクション4 の手順でリポジトリを再クローン

### Step 2: Claude Code への投入プロンプト

以下をそのままコピペしてClaude Codeに投入する:

```
Phase C-3: PPO強化学習訓練を実行してください。

## 背景
- training/tenbin_env.py に Gymnasium 互換の TenbinEnv がある
- training/agents.py に9種のエージェントがある
- Phase C-2 のトーナメントで最強は LevelK(3)=26（勝率60.7%）、HistLK(2)（57.8%）、Adaptive（52.3%）

## 作業内容

### 1. training/train_ppo.py を作成
- Stable-Baselines3 の PPO を使用
- 対戦相手プール: LevelK(3), HistLK(2), Adaptive, Fixed(32), RandomAgent をランダムに3体選んで対戦
- 訓練ステップ数: 500,000〜1,000,000（時間に応じて調整）
- ハイパーパラメータ: learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99
- TensorBoard ログを training/logs/ に保存
- 10万ステップごとにチェックポイントを training/models/ に保存
- 最終モデルを training/models/ppo_tenbin_final.zip に保存
- デバイスは自動検出（GPU可ならGPU）

### 2. training/evaluate.py を作成
- 訓練済みPPOモデルをロードし、全9エージェントとトーナメント形式で評価
- Phase C-2 の tournament.py と同様の形式で結果を出力
- 結果を training/results/ppo_evaluation.json に保存

### 3. 実行
- train_ppo.py を実行して訓練（30分〜2時間想定）
- 訓練完了後に evaluate.py を実行
- 結果を README.md に追記

## 注意
- 既存ファイルへの変更は最小限に
- PWA本体には触らない
- 完了後 git add / commit / push
```

### Step 3: 予想される承認ポイント

Claude Code 実行中に以下の場面で承認を求められる可能性がある:

1. **`pip install` 実行** — 追加パッケージのインストール時
2. **`python training/train_ppo.py` 実行** — 長時間実行コマンド（30分〜2時間）
3. **ファイル作成** — train_ppo.py, evaluate.py 等の新規ファイル
4. **git push** — リモートへのプッシュ

---

## 6. 次回の候補タスク

### Phase C-3: PPO強化学習（最優先）
- SB3のPPOでニューラルネット型AIを訓練
- 全9エージェントとのトーナメントで評価
- 訓練済みモデルをPWAの8体目キャラとして追加予定（Phase C-5）

### 友人フィードバック待ち項目
- 演出の追加（勝利/脱落時のアニメーション）
- 効果音・SE
- ハプティクス（バイブレーション強化）
- カラーテーマ（ライトモード/ダークモードのバランス）
- キャラクターのバランス調整
- UI/UXの使い勝手全般

### 将来検討
- オンライン対戦（WebRTC P2P）: シグナリングサーバー不要のP2P通信で実装可能か調査
- 戦績のグラフ可視化（勝率推移・選択値分布）
- チュートリアルモード（初めてのプレイヤー向け）

---

## 7. 友人からのフィードバック

> ここに実際にプレイした友人からのフィードバックを記録する。

### テスター1: （名前/日付）
- 
- 
- 

### テスター2: （名前/日付）
- 
- 
- 

### 共通して出た意見
- 

### 対応方針
- 

---

## 8. Phase C-4 以降の予定

### Phase C-4: ONNX 変換

```python
# 概略コード（Phase C-3 完了後に実装）
import torch
from stable_baselines3 import PPO

model = PPO.load("training/models/ppo_tenbin_final")
# PyTorch → ONNX
dummy_input = torch.randn(1, obs_dim)
torch.onnx.export(model.policy, dummy_input, "training/models/ppo_tenbin.onnx")
```

- SB3 の policy ネットワークを ONNX 形式にエクスポート
- onnxruntime で推論テストし、Python版と同じ出力が出ることを確認

### Phase C-5: PWA への ONNX モデル組み込み

- `onnxruntime-web` (WASM版) を使って ONNX モデルをブラウザで実行
- index.html に ONNX 推論コードを追加
- 新しいキャラクター（8体目の動物）としてゲームに組み込み
- レベル⭐5、訓練済みニューラルネットAI

### Phase C-6: 評価とガクチカ化

- 人間 vs PPO AI の対戦ログ収集
- 勝率・戦略分析のまとめ
- README.md をポートフォリオ向けに整備
- GitHub Pages のデモURLを確定

---

## 9. トラブルシューティング

### GitHub トークンのシークレットが消えた場合

1. GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens
2. 新しいトークンを生成（リポジトリ `tenbin-rl` に対する Read/Write 権限）
3. Colab → 左サイドバーの鍵アイコン → `GITHUB_TOKEN_TENBIN` を更新

### Colab のランタイムが切れた場合

- ランタイムが切れても GitHub にプッシュ済みのコードは失われない
- セクション4 の手順で新しいランタイムから再開すればOK
- **注意**: `training/models/` 内の未プッシュの訓練モデルは失われる。訓練中は定期的にコミット&プッシュすること
- 対策: train_ppo.py にチェックポイント保存機能を入れ、中断時も途中モデルが残るようにする

### Claude Code の認証が切れた場合

```bash
# APIキーを再設定
export ANTHROPIC_API_KEY="sk-ant-..."

# Claude Code を再起動
claude
```

- Colab シークレットから取得する場合:
```python
from google.colab import userdata
import os
os.environ['ANTHROPIC_API_KEY'] = userdata.get('ANTHROPIC_API_KEY')
```

### pip install でエラーが出る場合

```bash
# キャッシュクリアして再試行
pip install --no-cache-dir -r training/requirements.txt

# PyTorch が GPU を認識しない場合
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 訓練が遅い / GPU が使われていない場合

```python
import torch
print(torch.cuda.is_available())  # True であること
print(torch.cuda.get_device_name(0))  # "Tesla T4" 等
```

GPU が False の場合:
1. ランタイム → ランタイムのタイプを変更 → T4 GPU を選択
2. ランタイムを再起動
3. PyTorch を再インストール

---

## 10. 現在の技術スタック一覧

### PWA側（フロントエンド）

| 技術 | 用途 |
|---|---|
| HTML5 / CSS3 / JavaScript | 単一ファイルSPA |
| PWA (manifest.json, Service Worker) | オフライン対応、ホーム画面追加 |
| localStorage | 戦績保存、言語設定保存 |
| CSS Variables + prefers-color-scheme | ダークモード対応 |
| i18n (JSオブジェクト辞書) | 日本語/英語切替 |

### 訓練側（バックエンド / ML）

| 技術 | 用途 |
|---|---|
| Python 3.12 | 訓練環境 |
| Gymnasium | RL環境インターフェース |
| NumPy | 数値計算 |
| Stable-Baselines3 | PPO実装（Phase C-3） |
| PyTorch | ニューラルネットワーク（Phase C-3） |
| ONNX / onnxruntime | モデル変換・推論（Phase C-4） |
| matplotlib | 結果の可視化 |

### インフラ

| 技術 | 用途 |
|---|---|
| GitHub | コード管理 |
| GitHub Pages | PWA ホスティング |
| Google Colab | GPU 訓練環境 |
| Claude Code | AI ペアプログラミング |

---

## リポジトリ構成（現時点）

```
tenbin-rl/
├── index.html              # PWA本体（v0.3: 動物キャラ、中断機能、i18n）
├── manifest.json            # PWAマニフェスト
├── service-worker.js        # オフラインキャッシュ（v3）
├── icons/
│   ├── icon-192.svg
│   └── icon-512.svg
├── training/
│   ├── __init__.py
│   ├── tenbin_env.py        # Gymnasium環境 + コアロジック
│   ├── agents.py            # 全9種エージェント
│   ├── simulate.py          # 基本シミュレーション
│   ├── tournament.py        # ラウンドロビントーナメント
│   ├── analyze.py           # matplotlib可視化
│   ├── requirements.txt
│   ├── README.md
│   └── results/
│       ├── baseline.json
│       ├── tournament.json
│       └── plots/
│           ├── bar_chart.png
│           ├── heatmap.png
│           └── elo_ranking.png
├── README.md                # プロジェクト全体の説明
├── RESUME.md                # ← このファイル
└── LICENSE
```
