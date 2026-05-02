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

## 6. Cross-engine validation (Project Chrono — two tiers)

Tier A — SCM Bekker–Wong (different rigid solver AND different terrain class):
```bash
conda run -n chrono_viz python cross_engine/chrono_validation.py \
    --onnx sim2real/onnx_export/output/recovery_policy.onnx \
    --num_trials 50 --seeds 0 1 2 --terrain scm
```

Tier B — Chrono CRM/SPH (same rigid solver as Tier A, but continuum granular
SPH instead of SCM — isolates the terrain-physics gap from the rigid-solver gap;
requires pychrono built with `-DENABLE_MODULE_FSI=ON`):
```bash
conda run -n chrono_viz python cross_engine/chrono_validation.py \
    --onnx sim2real/onnx_export/output/recovery_policy.onnx \
    --num_trials 50 --seeds 0 1 2 --terrain crm
```

### Optional: one-shot orchestrator
For unattended overnight runs only — step-by-step is still recommended for the
paper run since intermediate JSONs should be inspected:
```bash
bash scripts/run_all_validation.sh --checkpoint "$CKPT" --seeds "0 1 2 3 4" --num_trials 100
# add --skip-crm if pychrono.fsi is not available
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
