# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Registry for all environments."""
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import ml_collections
from mujoco import mjx

from mujoco_playground._src import mjx_env

from philab_mujoco import locomotion

DomainRandomizer = Optional[
    Callable[[mjx.Model, jax.Array], Tuple[mjx.Model, mjx.Model]]
]

ALL_ENVS = (
    locomotion.ALL_ENVS
)


def get_default_config(env_name: str):
    if env_name in locomotion.ALL_ENVS:
        return locomotion.get_default_config(env_name)

    raise ValueError(f"Env '{env_name}' not found in default configs.")


def load(
        env_name: str,
        config: Optional[ml_collections.ConfigDict] = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
) -> mjx_env.MjxEnv:
    if env_name in locomotion.ALL_ENVS:
        return locomotion.load(env_name, config, config_overrides)

    raise ValueError(f"Env '{env_name}' not found. Available envs: {ALL_ENVS}")


def get_domain_randomizer(env_name: str) -> Optional[DomainRandomizer]:
    if env_name in locomotion.ALL_ENVS:
        return locomotion.get_domain_randomizer(env_name)

    return None
