from typing import Any, Dict, Optional, Union
from ml_collections import config_dict

from philab_mujoco.locomotion.tron_sf import base
from philab_mujoco.locomotion.tron_sf import constants as consts


def tron_sf_joystick_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        episode_length=1000,
        Kp=45.0,
        Kd=1.5,
    )


class TronSfJoystickEnv(base.TronSfBaseEnv):
    def __init__(
            self,
            task: str = "flat_terrain",
            config: config_dict.ConfigDict = tron_sf_joystick_config(),
            config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=consts.task_to_xml(task).as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
