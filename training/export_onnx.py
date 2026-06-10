"""
Phase C-4: 学習済みPPOモデルを ONNX 形式へ変換

PWA (onnxruntime-web) でブラウザ推論するため、SB3 の PPO 方策ネットワークを
ONNX にエクスポートする。出力 "action" が決定論的な選択値 (0-100) になるので、
ブラウザ側はそれをそのまま使えばよい。

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


class OnnxablePolicy(th.nn.Module):
    """SB3 方策を ONNX 化可能にする薄いラッパー。

    forward は deterministic=True で (action, value, log_prob) を返す。
    ブラウザ側は最初の出力 "action"（決定論的な選択値）だけ使えばよい。
    """

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation):
        return self.policy(observation, deterministic=True)


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

    onnx_policy = OnnxablePolicy(model.policy)
    obs_shape = model.observation_space.shape  # 例: (27,)
    dummy = th.randn(1, *obs_shape, dtype=th.float32)

    out_path = os.path.join(MODELS_DIR, "ppo_tenbin.onnx")
    th.onnx.export(
        onnx_policy,
        dummy,
        out_path,
        opset_version=17,
        input_names=["obs"],
        output_names=["action", "value", "log_prob"],
        dynamic_axes={"obs": {0: "batch"}},
    )
    print(f"ONNX を保存: {out_path}  (obs_dim={obs_shape[0]})")

    # onnxruntime で PyTorch と出力が一致するか検証
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        test = np.random.randn(1, *obs_shape).astype(np.float32)
        onnx_action = sess.run(None, {"obs": test})[0]
        with th.no_grad():
            th_action, _, _ = onnx_policy(th.tensor(test))
        print(
            f"検証: ONNX action={int(np.ravel(onnx_action)[0])}  "
            f"/  PyTorch action={int(th.ravel(th_action)[0])}"
        )
        print("→ 一致していれば変換成功。次は PWA への組込み (C-5) です。")
    except Exception as e:  # noqa: BLE001
        print(f"onnxruntime 検証はスキップ: {e}")


if __name__ == "__main__":
    main()
