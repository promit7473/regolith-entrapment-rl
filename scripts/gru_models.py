"""GRU actor/critic model definitions — extracted side-effect-free.

These are byte-for-byte the same architecture as the models defined inline in
`scripts/train.py`, but live in a module that does NOT launch a SimulationApp at
import time. `train.py` boots its own Isaac Sim app at module scope (unguarded by
`__main__`), so `from train import GRUPolicyNet` from inside an already-running
app deadlocks trying to start a second app. Import these from here instead.

Keep in sync with the hyperparameters in train.py (GRU_HIDDEN, GRU_LAYERS,
SEQ_LEN, ROLLOUTS, POLICY_OBS_DIM).
"""

import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

GRU_HIDDEN  = 256
GRU_LAYERS  = 1
SEQ_LEN     = 32
ROLLOUTS    = 64
POLICY_OBS_DIM = 29


class GRUPolicyNet(GaussianMixin, Model):
    """Recurrent policy: Linear encoder → GRU → MLP head → Gaussian action."""

    def __init__(self, obs_space, act_space, device, num_envs, clip_actions=False):
        Model.__init__(self, obs_space, act_space, device)
        GaussianMixin.__init__(self, clip_actions=clip_actions,
                               clip_log_std=True, min_log_std=-20, max_log_std=2)
        self._num_envs  = num_envs
        self._hidden    = GRU_HIDDEN
        self._layers    = GRU_LAYERS
        self._seq_len   = SEQ_LEN
        self.encoder = nn.Sequential(nn.Linear(POLICY_OBS_DIM, 128), nn.ELU())
        self.gru = nn.GRU(128, GRU_HIDDEN, num_layers=GRU_LAYERS, batch_first=False)
        self.head = nn.Sequential(
            nn.Linear(GRU_HIDDEN, 64), nn.ELU(),
            nn.Linear(64, self.num_actions),
        )
        self.log_std = nn.Parameter(torch.zeros(self.num_actions))

    def get_specification(self):
        return {"rnn": {"sequence_length": self._seq_len,
                        "sizes": [(self._layers, self._num_envs, self._hidden)]}}

    def compute(self, inputs, role=""):
        states = inputs["states"][:, :POLICY_OBS_DIM]
        rnn_list = inputs.get("rnn", [None])
        hidden = rnn_list[0] if (rnn_list and rnn_list[0] is not None) else None
        if hidden is not None:
            batch = hidden.shape[1]
            seq   = states.shape[0] // batch
        else:
            batch = states.shape[0]
            seq   = 1
        x = self.encoder(states)
        x = x.view(seq, batch, -1)
        x, h_n = self.gru(x, hidden)
        x = x.reshape(seq * batch, -1)
        output = self.head(x)
        return output, self.log_std.expand_as(output), {"rnn": [h_n]}


class GRUValueNet(DeterministicMixin, Model):
    """Recurrent value critic: same GRU architecture, scalar output."""

    def __init__(self, obs_space, act_space, device, num_envs, clip_actions=False):
        Model.__init__(self, obs_space, act_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self._num_envs  = num_envs
        self._hidden    = GRU_HIDDEN
        self._layers    = GRU_LAYERS
        self._seq_len   = SEQ_LEN
        self.encoder = nn.Sequential(nn.Linear(self.num_observations, 128), nn.ELU())
        self.gru = nn.GRU(128, GRU_HIDDEN, num_layers=GRU_LAYERS, batch_first=False)
        self.head = nn.Sequential(
            nn.Linear(GRU_HIDDEN, 64), nn.ELU(),
            nn.Linear(64, 1),
        )

    def get_specification(self):
        return {"rnn": {"sequence_length": self._seq_len,
                        "sizes": [(self._layers, self._num_envs, self._hidden)]}}

    def compute(self, inputs, role=""):
        states = inputs["states"]
        rnn_list = inputs.get("rnn", [None])
        hidden = rnn_list[0] if (rnn_list and rnn_list[0] is not None) else None
        if hidden is not None:
            batch = hidden.shape[1]
            seq   = states.shape[0] // batch
        else:
            batch = states.shape[0]
            seq   = 1
        x = self.encoder(states)
        x = x.view(seq, batch, -1)
        x, h_n = self.gru(x, hidden)
        x = x.reshape(seq * batch, -1)
        output = self.head(x)
        return output, {"rnn": [h_n]}
