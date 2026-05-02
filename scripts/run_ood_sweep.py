#!/usr/bin/env python
"""Out-of-distribution sweep: evaluate the trained policy across a grid of
sinkage × friction values that bracket and exceed the training DR range.

Training DR: sinkage ∈ [0.15, 0.28] m, friction ∈ [0.6, 0.9].
This sweep: sinkage ∈ {0.18, 0.23, 0.28, 0.33, 0.38} × friction ∈ {0.4, 0.6, 0.75, 0.9, 1.1}.

Wraps scripts/eval.py with environment overrides exposed via env-vars consumed
by EntrapmentEnvCfg.{sinkage_override, friction_override} (set on cfg before
gym.make). Each cell runs N=50 trials.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SINKAGES = [0.18, 0.23, 0.28, 0.33, 0.38]
FRICTIONS = [0.4, 0.6, 0.75, 0.9, 1.1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num_trials", type=int, default=50)
    ap.add_argument("--num_envs", type=int, default=8)
    ap.add_argument("--out_dir", default=str(REPO / "experiments" / "ood_sweep"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = []
    for s in SINKAGES:
        for mu in FRICTIONS:
            cell_id = f"s{int(s*100):03d}_mu{int(mu*100):03d}"
            cell_json = out_dir / f"{cell_id}.json"
            if cell_json.exists():
                print(f"[ood] {cell_id} cached")
                grid.append({"sinkage": s, "friction": mu,
                             "result": json.loads(cell_json.read_text())})
                continue

            env = os.environ.copy()
            env["OOD_SINKAGE"] = str(s)
            env["OOD_FRICTION"] = str(mu)
            cmd = [
                "./launch.sh", "scripts/eval.py",
                "--checkpoint", args.checkpoint,
                "--num_envs", str(args.num_envs),
                "--num_trials", str(args.num_trials),
                "--out_json", str(cell_json),
            ]
            print(f"[ood] {cell_id}: {' '.join(cmd)}")
            r = subprocess.run(cmd, cwd=REPO, env=env)
            if r.returncode != 0 or not cell_json.exists():
                print(f"[ood] WARN: {cell_id} failed (rc={r.returncode})")
                continue
            grid.append({"sinkage": s, "friction": mu,
                         "result": json.loads(cell_json.read_text())})

    summary_path = out_dir / "sweep_summary.json"
    summary_path.write_text(json.dumps({
        "sinkages": SINKAGES, "frictions": FRICTIONS,
        "num_trials": args.num_trials, "grid": grid,
    }, indent=2))
    print(f"\n[ood] grid summary → {summary_path}")
    print("[ood] Render heatmap with scripts/make_ood_heatmap.py "
          "(consumes sweep_summary.json).")


if __name__ == "__main__":
    main()
