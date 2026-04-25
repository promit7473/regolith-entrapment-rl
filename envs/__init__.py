import gymnasium as gym

from .entrapment_env import EntrapmentEnv, EntrapmentEnvCfg

gym.register(
    id="MarsRover-RegolithEscape-v0",
    entry_point="envs.entrapment_env:EntrapmentEnv",
    kwargs={"env_cfg_entry_point": EntrapmentEnvCfg},
    disable_env_checker=True,
)

# Sim2sim validation env — A→sand→B scenario. Subclasses EntrapmentEnv;
# overrides only spawn geometry + termination. Training env untouched.
gym.register(
    id="MarsRover-Sim2SimValidation-v0",
    entry_point="sim2sim_validation.validation_env:ValidationEnv",
    kwargs={"env_cfg_entry_point": "sim2sim_validation.validation_env:ValidationEnvCfg"},
    disable_env_checker=True,
)
