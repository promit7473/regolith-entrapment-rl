# Lab PC Transfer Guide

Copy these folders from this dev PC to the lab PC instead of re-downloading.
Total size: **~25 GB**.

---

## Folders to copy (source → destination)

All destinations assume the lab PC user is `<labuser>` and home is `/home/<labuser>/`.
Keep paths identical to dev PC — Isaac Sim and several scripts hardcode them.

| # | Source (this PC) | Destination (lab PC) | Size | Notes |
|---|------------------|----------------------|------|-------|
| 1 | `~/isaac-sim/` | `~/isaac-sim/` | 13 GB | Isaac Sim install. **Not relocatable** — must go in `~/`. |
| 2 | `~/.local/share/ov/` | `~/.local/share/ov/` | 8.9 GB | Omniverse extension cache + `pxr` ext. Required. |
| 3 | `~/RLRoverLab/` | `~/RLRoverLab/` | 2.6 GB | AAU rover USD assets live here. |
| 4 | `~/IsaacLab/` | `~/IsaacLab/` | 97 MB | Source repo (commit `44c26e31`, branch `feature/newton`). |
| 5 | `~/newton/` | `~/newton/` | 21 MB | Newton physics (commit `551f6ee`). |
| 6 | `~/regolith_entrapment_research/` | `~/regolith_entrapment_research/` | — | This repo. Or just `git clone` it on the lab PC. |

---

## Recommended copy method

Use `rsync` over SSH (resumable, preserves symlinks/perms):

```bash
# From dev PC, run for each folder:
rsync -avh --progress ~/isaac-sim/         <labuser>@<lab-pc>:~/isaac-sim/
rsync -avh --progress ~/.local/share/ov/   <labuser>@<lab-pc>:~/.local/share/ov/
rsync -avh --progress ~/RLRoverLab/        <labuser>@<lab-pc>:~/RLRoverLab/
rsync -avh --progress ~/IsaacLab/          <labuser>@<lab-pc>:~/IsaacLab/
rsync -avh --progress ~/newton/            <labuser>@<lab-pc>:~/newton/
rsync -avh --progress ~/regolith_entrapment_research/  <labuser>@<lab-pc>:~/regolith_entrapment_research/
```

If no network between them: `rsync` to an external SSD, then `rsync` from SSD to lab PC.

---

## What you still need to install on the lab PC

These don't transfer cleanly — install fresh:

1. **Miniconda** — https://docs.conda.io/en/latest/miniconda.html
2. **Conda env `env_isaaclab`** (Python 3.11):
   ```bash
   conda create -n env_isaaclab python=3.11 -y
   conda activate env_isaaclab
   # Then install IsaacLab deps:
   cd ~/IsaacLab && ./isaaclab.sh -i
   # And project deps (skrl, warp-lang, etc.) — see scripts/train.py imports
   ```
3. **NVIDIA driver ≥ 580** + CUDA 13 runtime (must match this PC's `nvidia-smi`).
4. **CPU governor** — set to `performance`:
   ```bash
   sudo apt install linux-tools-common linux-tools-generic
   sudo cpupower frequency-set -g performance
   ```

---

## After copying — verify

On the lab PC:

```bash
cd ~/regolith_entrapment_research

# 1. Apply Newton patch (already in repo, but re-check it's applied):
cat patches/newton_mujoco_bugfixes.patch
# If newton repo doesn't already include these fixes:
cd ~/newton && git apply ~/regolith_entrapment_research/patches/newton_mujoco_bugfixes.patch

# 2. Smoke test (4 envs, 2k steps):
cd ~/regolith_entrapment_research
./launch.sh scripts/train.py --num_envs 4 --timesteps 2000
```

If smoke test runs end-to-end, lab PC is ready.

---

## Things to watch out for

- **Hardcoded paths**: `CLAUDE.md` lists `~/newton/`, `~/IsaacLab/source/`, `~/isaac-sim/`, `~/RLRoverLab/rover_envs/assets/`, `~/.local/share/ov/data/exts/v2/omni.usd.libs-*/`. If lab username differs from `mhpromit7473`, grep the repo for `mhpromit7473` and fix paths in `paths.py`, `paths.sh`, `launch.sh`.
- **Isaac Sim shader cache** (`~/.cache/ov`, `~/.nv/ComputeCache`): don't copy — let it rebuild on first run.
- **`experiments/` folder**: training outputs, gitignored. Copy only if you want to resume training from a checkpoint.
- **First launch will still take 10–15 min** for Warp JIT compilation even with cache copied (different GPU arch may invalidate cache).
