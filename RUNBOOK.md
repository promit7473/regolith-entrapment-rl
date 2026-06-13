# Reproduction Runbook

End-to-end commands to reproduce every paper figure & table on a single RTX
5070Ti box. Run each block in order; later blocks consume earlier outputs.

> **v12 (2026-06-13): ALL results produced before this date are invalid for
> the paper.** Physics fixes change the environment: (1) **substep force
> dilution — THE BIG ONE**: every sand force in every prior run was applied at
> 1/NEWTON_SUBSTEPS (=⅛) of its true value (NewtonManager clears forces each
> substep); now delivered continuously via a clear_forces wrapper. The rover
> could never actually drive on sand before this fix. (2) sand bed now
> simulates under the same gravity as the rover (was Earth-g sand vs Mars-g
> rover); (3) MPM rheology solve now converges (`max_iterations` 30→100,
> `tolerance` 1e-5→1e-7 — the old setting leaked bed volume every step);
> (4) bed pre-settled at init, spawn sinkage references the measured surface,
> spawn pocket-carving removes teleport interpenetration; (5) wheel geometry
> measured definitively (`scripts/wheel_geometry.py`): r_eff=0.0994 + 8
> true-size grouser blade colliders. Obs semantics also changed (IMU =
> body-frame specific force / local g; DR noise in true physical units), so
> old checkpoints are incompatible. Retrain everything below from scratch.
> See CLAUDE.md "CURRENT STATE" (untracked — copy alongside the repo).
> Step cost is ~1.5–2× the old runs; budget accordingly.

**Note:** the only live validation pipeline is the cross-engine Chrono
validation on granular (Bekker-Wong) terrain. The sim2sim (MPM→MPM) track
has been removed — see git history for archived scripts.

## 0. One-time setup

```bash
cd ~/regolith_entrapment_research
pip install onnxruntime
```

Chrono validation requires the `chrono_viz` conda environment:

```bash
conda create -n chrono_viz python=3.12 -c conda-forge -y
conda install -n chrono_viz -c conda-forge pychrono=10.0.0 numpy onnxruntime matplotlib -y
```

## 1. Train 5 seeds (~23 h on 5070Ti @ 64 envs × 200k steps)

```bash
bash scripts/train_multiseed.sh --seeds "0 1 2 3 4" --timesteps 200000 --num_envs 64
```

Per-seed checkpoints land under `experiments/regolith_recovery/seed_{N}/...`.
Pick the final checkpoint of seed 0 as `$CKPT` for downstream eval:

```bash
CKPT=$(ls experiments/regolith_recovery/seed_0/*/checkpoints/agent_200000.pt | tail -1)
```

## 1b. Scripted-baseline tuning (no training needed — run BEFORE/while training)

```bash
bash scripts/tune_scripted_baseline.sh                      # gate level (EPS=4)
EPS=50 bash scripts/tune_scripted_baseline.sh               # paper level, winner only
```

Sweeps the compatible-actuation scripted family (rocking / steer_paddle /
rock_paddle; spiral cells appended separately) over period × throttle at the
trapped depths. The best cell = the "tuned scripted" row of the comparison
table — the fairness bar the policy must beat. Results land in
`experiments/scripted_tuning/summary.csv`.

## 2. Ablations (one seed each; ~4.5 h × 4 = ~18 h)

```bash
bash scripts/train_ablations.sh --timesteps 200000 --num_envs 64 --seed 0
```

## 3. ONNX export

```bash
./launch.sh sim2real/onnx_export/export_model.py --policy_ckpt "$CKPT"
```

Produces `sim2real/onnx_export/output/recovery_policy.onnx`.

## 4. Cross-engine validation — granular (Bekker-Wong) terrain

**What makes this genuinely cross-engine:**
- Rigid-body solver: Chrono `ChSystemNSC` (NSC complementarity + Bullet broadphase)
  vs. training Newton (MPM implicit + MuJoCo contact).
- Terrain physics: Bekker-Wong semi-empirical (`p(z) = (Kφ/b + Kc)·zⁿ`,
  Janosi-Hanamoto shear) vs. MPM continuum elasto-plasticity — different
  mathematical class.
- Rover: Curiosity (899 kg, r=0.25 m) vs. AAU rover (~35 kg, r=0.10 m).

### Granular terrain (default, recommended):

```bash
conda run -n chrono_viz python cross_engine_validation/chrono_validation.py \
    --onnx sim2real/onnx_export/output/recovery_policy.onnx \
    --terrain granular \
    --num_trials 50 --seeds 0 1 2 \
    --output cross_engine_validation/results/
```

Uses `pychrono.vehicle.SCMTerrain` with bulldozing enabled + synthetic chassis
velocity damping (BULLDOZE_DAMP=80) to emulate the entrapment feedback loop
at low friction angles. Internal friction angle randomised per trial (φ = 10°–30°)
— lower φ → weaker soil → deeper wheel sinkage → stronger entrapment + more
aggressive synthetic drag.

### Rigid baseline (ablation):

```bash
conda run -n chrono_viz python cross_engine_validation/chrono_validation.py \
    --onnx sim2real/onnx_export/output/recovery_policy.onnx \
    --terrain rigid \
    --num_trials 50 --seeds 0 1 2 \
    --output cross_engine_validation/results/
```

### One-shot orchestrator

```bash
bash scripts/run_all_validation.sh --checkpoint "$CKPT" --seeds "0 1 2" --num_trials 50
```

## 5. Paper figures

```bash
python3 scripts/plot_chrono_transfer.py    # cross-engine transfer diagram
```

PDFs land in `paper/figures/`.

## 6. Compile paper

```bash
cd paper && latexmk -pdf paper.tex
```
