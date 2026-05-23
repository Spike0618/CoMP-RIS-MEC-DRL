"""
Tianshou

Tianshou
"""

from src.algos.tianshou.networks import (
    CompRISActor,
    CompRISCritic,
    create_actor_critic,
)

from src.algos.tianshou.ppo_config import create_ppo_policy

__all__ = [
    'CompRISActor',
    'CompRISCritic',
    'create_actor_critic',
    'create_ppo_policy',
]
