"""



- PPO/DDPG/SAC/TD3
- 
"""

from __future__ import annotations

import math
import torch.nn as nn


def init_linear(layer: nn.Linear, gain: float, bias: float = 0.0) -> None:
    """"""
    nn.init.orthogonal_(layer.weight, gain=float(gain))
    if layer.bias is not None:
        nn.init.constant_(layer.bias, float(bias))


def hidden_gain() -> float:
    """ReLU """
    return math.sqrt(2.0)


def actor_head_gain() -> float:
    """"""
    return 0.01


def critic_head_gain() -> float:
    """/Q """
    return 1.0
