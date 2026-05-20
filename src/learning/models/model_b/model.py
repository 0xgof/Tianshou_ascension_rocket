"""Model B: empty model shell reserved for your implementation."""

from __future__ import annotations

from learning.tianshou.models import (GaussianPolicyConfig, NetworkConfig,
                                      TorchDevice)


def build_model(obs_dim: int,
                action_dim: int,
                actor_network: NetworkConfig,
                critic_network: NetworkConfig,
                gaussian_policy: GaussianPolicyConfig,
                device: TorchDevice = None):
    """Build Model B.

    YOUR CODE AREA:
    - Create a ModelBActor class.
    - Create a ModelBCritic class.
    - Use model_b/mlp.py for hidden layers.
    - Use model_b/heads.py for actor and critic heads.
    - Return ``actor, critic``.

    Required Tianshou contract:

    ```text
    actor(obs)  -> ((mean, std), state)
    critic(obs) -> value_tensor
    ```
    """

    raise NotImplementedError(
        "Model B is intentionally empty. Implement build_model(...) in "
        "src/learning/models/model_b/model.py."
    )
