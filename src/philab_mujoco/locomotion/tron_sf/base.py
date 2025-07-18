from typing import Any, Dict, Optional, Union
from mujoco_playground import MjxEnv


def get_assets() -> Dict[str, bytes]:
    assets = {}
    return assets


class TronSfBaseEnv(MjxEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._env_name = "TronSf"
