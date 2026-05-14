# Reproduction Runbook

End-to-end commands to reproduce every paper figure & table on a single RTX
5070Ti box. Run each block in order; later blocks consume earlier outputs.

## 0. One-time setup

```bash
cd ~/regolith_entrapment_research
pip install onnxruntime rliable scipy matplotlib  # if missing
```

## 1. Train 5 seeds  (~23 h on 5070Ti @ 64 envs × 200k steps)

```bash
bash scripts/train_multiseed.sh --seeds "0 1 2 3 4" --timesteps 200000 --num_envs 64
```

Per-seed checkpoints land under `experiments/regolith_recovery/seed_{N}/...`.
Pick the final checkpoint of seed 0 as `$CKPT` for downstream eval:

```bash
CKPT=$(ls experiments/regolith_recovery/seed_0/*/checkpoints/agent_200000.pt | tail -1)
```

## 2. Ablations (one seed each; ~4.5 h × 4 = ~18 h)

```bash
bash scripts/train_ablations.sh --timesteps 200000 --num_envs 64 --seed 0
```

## 3. Full statistical eval — 5 seeds × 100 trials × 3 conditions

```bash
./launch.sh scripts/run_full_validation.py \
    --checkpoint "$CKPT" --seeds "0 1 2 3 4" --num_trials 100 --num_envs 8
```

Writes `experiments/full_validation/seed_*.json` and `aggregate_report.json`
(success rates + Mann–Whitney U with Bonferroni correction).

## 4. OOD sweep — 5×5 sinkage×friction grid, 50 trials each

```bash
./launch.sh scripts/run_ood_sweep.py --checkpoint "$CKPT" --num_trials 50 --num_envs 8
```

## 5. ONNX export

```bash
./launch.sh sim2real/onnx_export/export_model.py --policy_ckpt "$CKPT"
```

Produces `sim2real/onnx_export/output/recovery_policy.onnx`.

## 6. Cross-engine validation (Project Chrono — Bekker-Wong terrain)

**Setup (one-time):**
```bash
conda create -n chrono_viz python=3.12 -c conda-forge -y
conda install -n chrono_viz -c conda-forge pychrono=10.0.0 numpy onnxruntime -y
```

The conda-forge `pychrono 10.0.0` ships `core + robot` modules.
The validation uses a custom Bekker-Wong `ChLoad` applied per wheel —
no `pychrono.vehicle` (SCMTerrain) or `pychrono.fsi` (CRMTerrain) required.

**Run:**
```bash
conda run -n chrono_viz python cross_engine/chrono_validation.py \
    --onnx sim2real/onnx_export/output/recovery_policy.onnx \
    --num_trials 50 --seeds 0 1 2 \
    --output cross_engine/results/
```

Output: `cross_engine/results/chrono_bekker_{results.csv,summary.json}`

**What makes this genuinely cross-engine:**
- Rigid-body solver: Chrono `ChSystemNSC` (NSC complementarity + Bullet broadphase)
  vs. training Newton (MPM implicit + MuJoCo contact).
- Terrain physics: Bekker-Wong semi-empirical (`p(z) = (Kφ/b + Kc)·z^n`,
  Janosi-Hanamoto shear) vs. MPM continuum elasto-plasticity — different
  mathematical class, not a parameter perturbation.
- Rover: Curiosity (899 kg, r=0.25 m) vs. AAU rover (~35 kg, r=0.10 m).

### Optional: one-shot orchestrator
```bash
bash scripts/run_all_validation.sh --checkpoint "$CKPT" --seeds "0 1 2 3 4" --num_trials 100
```

## 7. Paper figures

```bash
./launch.sh scripts/make_rliable_figure.py    # IQM + stratified bootstrap CI
./launch.sh scripts/classify_failures.py      # failure-mode stacked bars
```

PDFs land in `paper/figures/`.

## 8. Compile paper

```bash
cd paper && latexmk -pdf paper.tex
```
