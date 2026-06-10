# Phase C-3：PPO 訓練 実行ガイド

学習済みニューラルネットAI（PWAの8体目キャラ候補）を作るための手順です。

> **GPUは不要**です。観測27次元・行動101・MLP 64×64 という極小ワークロードで、
> PPO+MlpPolicy は **CPUバウンド**（SB3公式もCPU推奨）。RTX4060 でも Colab でもOK。

## 成果物
| スクリプト | 役割 | 出力 |
|---|---|---|
| `training/train_ppo.py` | PPO訓練 | `training/models/ppo_tenbin_final.zip` |
| `training/evaluate.py` | 既存9エージェントとの対戦で勝率評価 | `training/results/ppo_evaluation.json` |

---

## A. ローカル（RTX4060ノート）で実行 ← おすすめ

### 1. リポジトリ取得
```bash
git clone https://github.com/7011yamazakyuuta-star/tenbin-rl.git
cd tenbin-rl
# 訓練スクリプトは PR #1 のブランチにあります。
# PR #1 を main にマージ済みなら不要。未マージなら↓でブランチに切り替え:
git checkout claude/kind-cerf-ox188s
```

### 2. 仮想環境 ＋ 依存インストール
```bash
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Mac / Linux:
source .venv/bin/activate

# CPU版torchで軽量に（この規模ではCPUが最速）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r training/requirements.txt
```
> GPUで回したい場合は上の torch 行を飛ばして `pip install -r training/requirements.txt` だけ
> （CUDA版torchが入る）。その場合は訓練時に `--device cuda`。ただし体感速度はほぼ同じ。

### 3. 動作確認（数秒）
```bash
python training/train_ppo.py --smoke
```
`[smoke] OK — reset/step/learn/predict すべて正常` が出れば準備完了。

### 4. 訓練（CPUで概ね 10〜40 分）
```bash
python training/train_ppo.py --timesteps 500000
# 物足りなければ: --timesteps 1000000
# GPUを使うなら:  --device cuda
```
- 進捗は SB3 のログ表で確認できます（`ep_rew_mean` が上がっていけば学習中）
- TensorBoardログ: `training/logs/`、10万stepごとのチェックポイント: `training/models/`
- 完了で `training/models/ppo_tenbin_final.zip` が生成されます

### 5. 評価
```bash
python training/evaluate.py
```
LevelK(3)=26 / HistLK(2) / Adaptive / Top3混在 などに対する**勝率**が表示され、
`training/results/ppo_evaluation.json` に保存されます。
（4人戦なので理論期待値は25%。これを上回れれば「学習が効いている」サイン）

### 6. コミット & プッシュ
```bash
git add training/models/ppo_tenbin_final.zip training/results/ppo_evaluation.json
git commit -m "Phase C-3: trained PPO model + evaluation"
git push
```
> `.gitignore` でログと中間チェックポイントは除外済み。**最終モデルと評価結果だけ**追跡されます。

---

## B. Google Colab で実行（GPU不要・CPUランタイムでOK）

```python
!git clone https://github.com/7011yamazakyuuta-star/tenbin-rl.git
%cd tenbin-rl
!pip install -q -r training/requirements.txt
!python training/train_ppo.py --smoke
!python training/train_ppo.py --timesteps 500000
!python training/evaluate.py
```
モデルは `training/models/ppo_tenbin_final.zip`。ダウンロードするか、GitHubトークンを設定して push。
（ランタイムが切れると未push成果物は消えるので、訓練後はこまめにコミット）

---

## 次のステップ（私が担当）
`ppo_tenbin_final.zip` と `ppo_evaluation.json` ができたら教えてください：

- **C-4：ONNX変換** — PyTorch方策を `onnxruntime` で動く形式へ
- **C-5：PWA組込み** — `onnxruntime-web`(WASM) で `index.html` に推論を実装し、
  **8体目の「学習済みAI」キャラ（⭐5）** としてゲームに追加

評価結果（勝率）も README に追記して、ポートフォリオの「学習で強くしたAI」の根拠にできます。
