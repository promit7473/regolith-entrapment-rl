"""
Phase 3 — Export trained PPO policy + sinkage detector to ONNX.

Usage:
    python phase3_sim2real/onnx_export/export_model.py \
        --policy_ckpt experiments/jackal_entrapment/ppo_mpm_v1/checkpoints/best_agent.pt \
        --detector_ckpt phase1_detection/models/saved/best_detector.pt \
        --out_dir phase3_sim2real/onnx_export/output

Outputs:
    recovery_policy.onnx   — 12D obs → 4D wheel velocity commands
    sinkage_detector.onnx  — (1, 50, 11) sequence → 3-class logits
"""

import argparse
import os
import sys

import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)


# ── Minimal policy wrapper (mirrors train.py PolicyNet) ────────────────────

class PolicyNet(nn.Module):
    """Matches the architecture in scripts/train.py."""
    def __init__(self, num_obs: int = 12, num_actions: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_obs, 256), nn.ELU(),
            nn.Linear(256, 128),    nn.ELU(),
            nn.Linear(128, 64),     nn.ELU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))   # clamp to [-1, 1]


def load_policy(ckpt_path: str, device: torch.device) -> nn.Module:
    """Load skrl PPO checkpoint into a bare PolicyNet."""
    ckpt = torch.load(ckpt_path, map_location=device)
    # skrl saves under "policy" key
    state = ckpt.get("policy", ckpt)
    model = PolicyNet()
    # Extract only the 'net' weights (skrl wraps with GaussianMixin extras)
    net_state = {k.replace("net.", ""): v for k, v in state.items() if k.startswith("net.")}
    model.net.load_state_dict(net_state, strict=True)
    return model.to(device).eval()


def export_policy(model: nn.Module, out_path: str, num_obs: int = 12):
    dummy = torch.zeros(1, num_obs)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  Policy exported → {out_path}")


def export_detector(ckpt_path: str, out_path: str, device: torch.device):
    from phase1_detection.models.cnn_gru import SinkageDetector
    ckpt  = torch.load(ckpt_path, map_location=device)
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
                                             "phase3_sim2real", "onnx_export", "output"))
    parser.add_argument("--num_obs",       type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu")   # ONNX export always on CPU

    print("\nExporting models to ONNX ...")

    # Policy
    policy = load_policy(args.policy_ckpt, device)
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
        import onnxruntime as ort
        sess = ort.InferenceSession(os.path.join(args.out_dir, "recovery_policy.onnx"))
        out  = sess.run(None, {"obs": torch.zeros(1, args.num_obs).numpy()})
        print(f"\n  ORT verification — policy output shape: {out[0].shape}  ✓")
    except ImportError:
        print("\n  onnxruntime not installed — skipping ORT verification.")

    print(f"\nDone. Models saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
