import gymnasium as gym

from .entrapment_env import EntrapmentEnv, EntrapmentEnvCfg

gym.register(
    id="AAURover-MarsEntrapment-v0",
    entry_point="envs.entrapment_env:EntrapmentEnv",
    kwargs={"env_cfg_entry_point": EntrapmentEnvCfg},
    disable_env_checker=True,
)
