from typing import Any, Dict, Optional, Union
from ml_collections import config_dict
from etils import epath
import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src import collision

from philab_mujoco.locomotion.tron_sf import base
from philab_mujoco.locomotion.tron_sf import constants as consts


def tron_sf_joystick_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        action_scale=0.5,
        history_len=1,
        soft_joint_pos_limit_factor=0.95,
        noise_config=config_dict.create(
            level=1.0,  # Set to 0.0 to disable noise.
            scales=config_dict.create(
                hip_pos=0.03,  # rad
                kfe_pos=0.05,
                ffe_pos=0.08,
                faa_pos=0.03,
                joint_vel=1.5,  # rad/s
                gravity=0.05,
                linvel=0.1,
                gyro=0.2,  # angvel.
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                # Tracking related rewards.
                tracking_lin_vel=1.0,
                tracking_ang_vel=0.5,
                # Base related rewards.
                lin_vel_z=0.0,
                ang_vel_xy=-0.15,
                orientation=-1.0,
                base_height=0.0,
                # Energy related rewards.
                torques=-2.5e-5,
                action_rate=-0.01,
                energy=0.0,
                # Feet related rewards.
                feet_clearance=0.0,
                feet_air_time=2.0,
                feet_slip=-0.25,
                feet_height=0.0,
                feet_phase=1.0,
                # Other rewards.
                stand_still=0.0,
                alive=0.0,
                termination=-1.0,
                # Pose related rewards.
                joint_deviation_knee=-0.1,
                joint_deviation_hip=-0.25,
                dof_pos_limits=-1.0,
                pose=-1.0,
            ),
            tracking_sigma=0.5,
            max_foot_height=0.1,
            base_height_target=0.5,
        ),
        push_config=config_dict.create(
            enable=True,
            interval_range=[5.0, 10.0],
            magnitude_range=[0.1, 2.0],
        ),
        lin_vel_x=[-1.0, 1.0],
        lin_vel_y=[-1.0, 1.0],
        ang_vel_yaw=[-1.0, 1.0],
    )


class TronSfJoystickEnv(base.TronSfBaseEnv):
    def __init__(
            self,
            task: str = None,
            config: config_dict.ConfigDict = tron_sf_joystick_config(),
            config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=consts.task_to_xml(task).as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def _post_init(self):
        self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
        self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

        # joint limits
        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T  # first joint is free
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

        # joint indices
        side_names = ["L", "R"]

        hip_indices = []
        hip_joint_names = ["abad", "hip"]
        for side in side_names:
            for joint_name in hip_joint_names:
                hip_indices.append(
                    self._mj_model.joint(f"{joint_name}_{side}_Joint").qposadr - 7
                )
            self._hip_indices = jp.array(hip_indices)

        knee_indices = []
        knee_joint_names = ["knee"]
        for side in side_names:
            for joint_name in knee_joint_names:
                knee_indices.append(
                    self._mj_model.joint(f"{joint_name}_{side}_Joint").qposadr - 7
                )
            self._knee_indices = jp.array(knee_indices)

        self._weights = jp.array([])

        # link indices
        self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
        self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
        self._site_id = self._mj_model.site("imu").id

        self._feet_site_id = jp.array(
            [self._mj_model.site(name).id for name in consts.FEET_SITES]
        )
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._feet_geom_id = jp.array(
            [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
        )

        foot_linvel_sensor_adr = []
        for site in consts.FEET_SITES:
            sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
            sensor_adr = self._mj_model.sensor_adr[sensor_id]
            sensor_dim = self._mj_model.sensor_dim[sensor_id]
            foot_linvel_sensor_adr.append(
                list(range(sensor_adr, sensor_adr + sensor_dim))
            )
        self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

        # noise scales
        # TODO: implement noise scales
