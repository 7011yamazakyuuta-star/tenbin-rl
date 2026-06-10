"""
Phase C-4: 学習済みPPOモデルを ONNX 形式へ変換

PWA (onnxruntime-web) でブラウザ推論するため、SB3 の PPO 方策ネットワークを
ONNX にエクスポートする。出力 "logits"（101次元）の argmax が決定論的な選択値 (0-100)。

torch の新しい dynamo エクスポータは SB3 の分布ベース forward を辿れないことがあるため、
- 方策ネットを「obs -> logits」の純テンソル演算だけに切り出し、
- 安定したレガシー(TorchScript)エクスポータ (dynamo=False) を使う。

使い方:
    python training/export_onnx.py
"""

import os
import sys

import numpy as np
import torch as th

# training/ ディレクトリをパスに追加（スクリプト直接実行用）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")


class PolicyLogits(th.nn.Module):
    """obs -> action logits（101次元）。決定論的な選択は argmax(logits)。

    SB3 の分布生成・サンプリングを通さず、方策ネットの純テンソル演算だけを
    切り出すことで、ONNX へ安定して変換できるようにする。
    """

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        features = self.policy.extract_features(obs)
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        return self.policy.action_net(latent_pi)


def main():
    model_path = os.path.join(MODELS_DIR, "ppo_tenbin_final")
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        raise SystemExit(
            f"モデルが見つかりません: {model_path}(.zip)\n"
            f"先に `python training/train_ppo.py` で訓練してください。"
        )

    print(f"モデルをロード: {model_path}")
    model = PPO.load(model_path, device="cpu")
    model.policy.eval()

    net = PolicyLogits(model.policy)
    obs_dim = model.observation_space.shape[0]
    dummy = th.randn(1, obs_dim, dtype=th.float32)

    out_path = os.path.join(MODELS_DIR, "ppo_tenbin.onnx")
    export_kwargs = dict(
        opset_version=17,
        input_names=["obs"],
        output_names=["logits"],
        dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}},
    )
    try:
        # 安定したレガシー(TorchScript)エクスポータを優先
        th.onnx.export(net, dummy, out_path, dynamo=False, **export_kwargs)
    except TypeError:
        # 古い torch には dynamo 引数が無い（その場合はそのままレガシー）
        th.onnx.export(net, dummy, out_path, **export_kwargs)
    print(f"ONNX を保存: {out_path}  (obs_dim={obs_dim}, 出力=logits[101])")

    # onnxruntime で PyTorch と出力が一致するか検証
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        test = np.random.randn(1, obs_dim).astype(np.float32)
        onnx_logits = sess.run(None, {"obs": test})[0]
        with th.no_grad():
            th_logits = net(th.tensor(test)).numpy()
        print(
            f"検証: ONNX argmax={int(onnx_logits.argmax())}  "
            f"/  PyTorch argmax={int(th_logits.argmax())}"
        )
        print("→ 一致していれば変換成功。次は PWA への組込み (C-5) です。")
    except Exception as e:  # noqa: BLE001
        print(f"onnxruntime 検証はスキップ: {e}")


if __name__ == "__main__":
    main()
