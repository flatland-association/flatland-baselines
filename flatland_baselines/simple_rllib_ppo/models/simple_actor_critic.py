import torch.nn as nn
from ray.rllib.core import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils import override


# https://docs.ray.io/en/latest/rllib/getting-started.html#rllib-python-api
class SimpleActorCriticTorchRLModule(TorchRLModule, ValueFunctionAPI):
    def setup(self):
        state_size = self.model_config["state_sz"]
        hidsize1 = self.model_config["hidden_sz"]
        hidsize2 = self.model_config["hidden_sz"]
        action_size = self.model_config["action_sz"]

        self.actor = nn.Sequential(
            nn.Linear(state_size, hidsize1),
            nn.Tanh(),
            nn.Linear(hidsize1, hidsize2),
            nn.Tanh(),
            nn.Linear(hidsize2, action_size),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_size, hidsize1),
            nn.Tanh(),
            nn.Linear(hidsize1, hidsize2),
            nn.Tanh(),
            nn.Linear(hidsize2, 1)
        )

    def _forward(self, batch, **kwargs):
        _batch = batch[Columns.OBS]
        _batch = _batch.unsqueeze(0)
        action_logits = self.actor.forward(_batch).squeeze(0)
        # Return parameters for the default action distribution, which is
        # `TorchCategorical` (due to our action space being `gym.spaces.Discrete`).
        return {Columns.ACTION_DIST_INPUTS: action_logits}

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        _batch = batch[Columns.OBS]
        _batch = _batch.unsqueeze(0)
        # Squeeze out last dimension (single node value head).
        return self.critic.forward(_batch).squeeze(-1).squeeze(0)
