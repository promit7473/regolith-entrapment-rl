# Newton Patches

These patches fix bugs in Newton's MuJoCo solver that prevent the Mars rover from loading.
Apply them after cloning Newton (`~/newton/`):

```bash
cd ~/newton
git apply ~/regolith_entrapment_research/patches/newton_mujoco_bugfixes.patch
```

## Bugs Fixed

| Bug | Symptom | Fix |
|-----|---------|-----|
| `vec3f` not accepted by `add_geom()` | `TypeError: add_geom() incompatible function arguments` | `list(tf.p)` / `list(quat_to_mjc(tf.q))` |
| `contype` int32 overflow (color=31 → 2^31) | `TypeError: ... value 2147483648 out of range` | `ctypes.c_int32(contype).value` |
| `actfrcrange=[0,0]` for passive joints | `ValueError: actfrcrange[0] should be smaller than actfrcrange[1]` | Clamp `effort_limit` to min 1.0 |
