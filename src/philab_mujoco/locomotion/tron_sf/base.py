from typing import Any, Dict, Optional, Union
from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict

import mujoco
from mujoco import mjx
from mujoco._structs import MjModel

from mujoco_playground._src import mjx_env

from philab_mujoco import ROBOT_PATH
from philab_mujoco.locomotion.tron_sf import constants as consts


def get_assets() -> Dict[str, bytes]:
    assets = {}
    mjx_env.update_assets(assets, ROBOT_PATH / "SF_TRON1A")
    return assets


class TronSfBaseEnv(mjx_env.MjxEnv):
    def __init__(
            self,
            xml_path: str,
            config: config_dict.ConfigDict,
            config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)

        self._model_assets = get_assets()
        self._mj_model: MjModel = mujoco.MjModel.from_xml_string(
            epath.Path(xml_path).read_text(), assets=self._model_assets
        )

        self._mj_model.opt.timestep = config.sim_dt

        self._mj_model.dof_damping = config.Kd
        self._mj_model.actuator_gainprm[:, 0] = config.Kp
        self._mj_model.actuator_biasprm[:, 1] = -config.Kp

        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160

        self._mjx_model = mjx.put_model(self._mj_model)
        self._xml_path = xml_path
        self._imu_site_id = self._mj_model.site("imu").id

    # Sensor readings.

    def get_upvector(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, consts.UPVECTOR_SENSOR)

    def get_gravity(self, data: mjx.Data) -> jax.Array:
        return data.site_xmat[self._imu_site_id].T @ jp.array([0, 0, -1])

    def get_global_linvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, consts.GLOBAL_LINVEL_SENSOR
        )

    def get_global_angvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, consts.GLOBAL_ANGVEL_SENSOR
        )

    def get_local_linvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, consts.LOCAL_LINVEL_SENSOR
        )

    def get_accelerometer(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, consts.ACCELEROMETER_SENSOR
        )

    def get_gyro(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, consts.GYRO_SENSOR)

    def get_feet_pos(self, data: mjx.Data) -> jax.Array:
        return jp.vstack([
            mjx_env.get_sensor_data(self.mj_model, data, sensor_name)
            for sensor_name in consts.FEET_POS_SENSOR
        ])

    # Accessors.

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
