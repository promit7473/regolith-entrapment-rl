"""
Phase 3 — Export trained Recurrent PPO (GRU) policy + sinkage detector to ONNX.

Usage:
    python sim2real/onnx_export/export_model.py \
        --policy_ckpt experiments/regolith_recovery/ppo_gru_regolith/checkpoints/best_agent.pt \
        --detector_ckpt detection/models/saved/best_detector.pt \
        --out_dir sim2real/onnx_export/output \
        --num_obs 29 --num_actions 10

Outputs:
    recovery_policy.onnx   — inputs: (obs[1,29], h_in[1,1,256])
                             outputs: (action[1,10], h_out[1,1,256])
    sinkage_detector.onnx  — (1, 50, 11) sequence → 3-class logits

ONNX policy inputs/outputs:
    obs     : float32 (1, num_obs)       — current observation
    h_in    : float32 (1, 1, gru_hidden) — GRU hidden state (num_layers, batch, hidden)
    action  : float32 (1, num_actions)   — tanh-clamped drive+steer commands
    h_out   : float32 (1, 1, gru_hidden) — updated GRU hidden state

On the RPi5, keep h_out from each step and pass it as h_in to the next.
Reset h_in to zeros at episode start (power-on or after escape).
"""

import argparse
import os
import sys

import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

GRU_HIDDEN = 256
GRU_LAYERS = 1


# ── Policy wrapper matching GRUPolicyNet in train.py ──────────────────────────

class GRUPolicyONNX(nn.Module):
    """
    Stateful GRU policy for ONNX export.
    Inputs : obs (1, num_obs), h_in (num_layers, 1, hidden)
    Outputs: action (1, num_actions), h_out (num_layers, 1, hidden)
    """
    def __init__(self, num_obs: int = 29, num_actions: int = 10,
                 hidden: int = GRU_HIDDEN, layers: int = GRU_LAYERS):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(num_obs, 128), nn.ELU())
        self.gru      = nn.GRU(128, hidden, num_layers=layers, batch_first=False)
        self.head     = nn.Sequential(
            nn.Linear(hidden, 64), nn.ELU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, obs: torch.Tensor, h_in: torch.Tensor):
        # obs: (1, num_obs)  →  (seq=1, batch=1, num_obs)
        x = self.encoder(obs).unsqueeze(0)          # (1, 1, 128)
        x, h_out = self.gru(x, h_in)               # x: (1, 1, hidden), h_out: (layers, 1, hidden)
        action = torch.tanh(self.head(x.squeeze(0)))  # (1, num_actions)
        return action, h_out


def load_gru_policy(ckpt_path: str, device: torch.device,
                    num_obs: int = 29, num_actions: int = 10) -> nn.Module:
    """Load skrl PPO_RNN checkpoint into a bare GRUPolicyONNX."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("policy", ckpt)

    model = GRUPolicyONNX(num_obs=num_obs, num_actions=num_actions)

    # Map skrl-saved keys to ONNX module keys
    mapping = {}
    for k, v in state.items():
        if k.startswith("encoder."):
            mapping[k] = v
        elif k.startswith("gru."):
            mapping[k] = v
        elif k.startswith("head."):
            mapping[k] = v
    # log_std is not used in ONNX (we take the mean action)
    model.load_state_dict(mapping, strict=True)
    return model.to(device).eval()


def export_policy(model: nn.Module, out_path: str, num_obs: int = 29):
    obs_dummy = torch.zeros(1, num_obs)
    h_dummy   = torch.zeros(GRU_LAYERS, 1, GRU_HIDDEN)
    torch.onnx.export(
        model,
        (obs_dummy, h_dummy),
        out_path,
        input_names=["obs", "h_in"],
        output_names=["action", "h_out"],
        dynamic_axes={
            "obs":    {0: "batch"},
            "h_in":   {1: "batch"},
            "action": {0: "batch"},
            "h_out":  {1: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  Policy exported → {out_path}")


def export_detector(ckpt_path: str, out_path: str, device: torch.device):
    from detection.models.cnn_gru import SinkageDetector
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SinkageDetector()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    dummy = torch.zeros(1, 50, 11)  # (batch, seq_len, features)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["sequence"],
        output_names=["logits"],
        dynamic_axes={"sequence": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  Detector exported → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_ckpt",   type=str, required=True)
    parser.add_argument("--detector_ckpt", type=str, default=None)
    parser.add_argument("--out_dir",       type=str,
                        default=os.path.join(REPO_ROOT,
                                             "sim2real", "onnx_export", "output"))
    parser.add_argument("--num_obs",       type=int, default=29)
    parser.add_argument("--num_actions",   type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu")   # ONNX export always on CPU

    print("\nExporting models to ONNX ...")

    # Policy (GRU)
    policy = load_gru_policy(args.policy_ckpt, device,
                              num_obs=args.num_obs, num_actions=args.num_actions)
    export_policy(policy,
                  os.path.join(args.out_dir, "recovery_policy.onnx"),
                  num_obs=args.num_obs)

    # Detector (optional)
    if args.detector_ckpt and os.path.isfile(args.detector_ckpt):
        export_detector(args.detector_ckpt,
                        os.path.join(args.out_dir, "sinkage_detector.onnx"),
                        device)
    else:
        print("  Skipping detector export (no checkpoint provided).")

    # Verify with onnxruntime
    try:
        import numpy as np
        import onnxruntime as ort
        onnx_path = os.path.join(args.out_dir, "recovery_policy.onnx")
        sess = ort.InferenceSession(onnx_path)
        obs_in = np.zeros((1, args.num_obs), dtype=np.float32)
        h_in   = np.zeros((GRU_LAYERS, 1, GRU_HIDDEN), dtype=np.float32)
        action, h_out = sess.run(None, {"obs": obs_in, "h_in": h_in})
        print(f"\n  ORT verification — action: {action.shape}, h_out: {h_out.shape}  ✓")
    except ImportError:
        print("\n  onnxruntime not installed — skipping ORT verification.")

    print(f"\nDone. Models saved to: {args.out_dir}")
    print("  ONNX inputs : obs (1, num_obs), h_in (num_layers, 1, gru_hidden)")
    print("  ONNX outputs: action (1, num_actions), h_out (num_layers, 1, gru_hidden)")
    print("  On RPi5: pass h_out back as h_in each step; reset to zeros at episode start.")


if __name__ == "__main__":
    main()
