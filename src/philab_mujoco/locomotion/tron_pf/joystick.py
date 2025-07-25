from typing import Any, Dict, Optional, Union
from ml_collections import config_dict
from etils import epath
import jax
import jax.numpy as jp
import numpy as np
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math

from mujoco_playground._src import gait
from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding

from philab_mujoco.locomotion.tron_pf import base
from philab_mujoco.locomotion.tron_pf import constants as consts


def tron_pf_joystick_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        action_num=6,
        action_repeat=1,
        action_scale=0.5,
        history_len=1,
        soft_joint_pos_limit_factor=0.95,
        noise_config=config_dict.create(
            level=1.0,  # Set to 0.0 to disable noise.
            scales=config_dict.create(
                hip_pos=0.05,  # rad
                knee_pos=0.05,
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
                lin_vel_z=-0.5,
                ang_vel_xy=-0.05,
                orientation=-10.0,
                base_height=-2.0,
                # Energy related rewards.
                torques=-8.0e-5,
                action_rate=-0.01,
                energy=0.0,
                joint_acc=-2.5e-7,
                # Feet related rewards.
                feet_distance=-100.0,
                feet_regulation=-0.05,
                feet_landing_vel=-0.15,
                feet_clearance=0.0,
                feet_air_time=1.0,
                feet_slip=-0.25,
                feet_height=0.0,
                feet_phase=0.5,
                # Other rewards.
                stand_still=0.0,
                alive=0.0,
                termination=-1.0,
                # Pose related rewards.
                joint_deviation_knee=-0.1,
                joint_deviation_hip=-0.25,
                dof_pos_limits=-2.0,
                pose=-1.0,
            ),
            tracking_sigma=0.4,
            max_foot_height=0.1,
            base_height_target=0.5,
            min_feet_distance=0.115,
            about_landing_threshold=0.08,
        ),
        push_config=config_dict.create(
            enable=True,
            interval_range=[5.0, 10.0],
            magnitude_range=[0.1, 2.0],
        ),
        lin_vel_x=[-1.0, 1.0],
        lin_vel_y=[-1.0, 1.0],
        ang_vel_yaw=[-1.0, 1.0],
        gait_freq_range=[1.5, 2.5],
    )


class TronPfJoystickEnv(base.TronPfBaseEnv):
    def __init__(
            self,
            task: str = None,
            config: config_dict.ConfigDict = tron_pf_joystick_config(),
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
        hip_joint_names = ["abad"]
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

        self._weights = jp.array([
            1.0, 0.01, 0.01,
            1.0, 0.01, 0.01,
        ])  # encourage movement on hips and knees

        # link indices
        self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
        self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
        self._site_id = self._mj_model.site("imu").id

        self._feet_site_id = np.array(
            [self._mj_model.site(name).id for name in consts.FEET_SITES]
        )
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._feet_geom_id = np.array(
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
        qpos_noise_scale = np.zeros(self._config.action_num)
        hip_ids = [0, 1, 3, 4]
        knee_ids = [2, 5]
        qpos_noise_scale[hip_ids] = self._config.noise_config.scales.hip_pos
        qpos_noise_scale[knee_ids] = self._config.noise_config.scales.knee_pos
        self._qpos_noise_scale = jp.array(qpos_noise_scale)

        # state history buffer size
        self._state_history_len = 10

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)

        # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
        qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        new_quat = math.quat_mul(qpos[3:7], quat)
        qpos = qpos.at[3:7].set(new_quat)

        # qpos[7:]=*U(0.5, 1.5)
        rng, key = jax.random.split(rng)
        qpos = qpos.at[7:].set(
            qpos[7:] * jax.random.uniform(key, (self._config.action_num,), minval=0.5, maxval=1.5)
        )

        # d(xyzrpy)=U(-0.5, 0.5)
        rng, key = jax.random.split(rng)
        qvel = qvel.at[0:6].set(
            jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
        )

        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:])

        # Phase, freq=U(1.0, 1.5)
        rng, key = jax.random.split(rng)
        gait_freq = jax.random.uniform(
            key, (1,),
            minval=self._config.gait_freq_range[0],
            maxval=self._config.gait_freq_range[1]
        )
        phase_dt = 2 * jp.pi * self.dt * gait_freq
        phase = jp.array([0, jp.pi])

        rng, cmd_rng = jax.random.split(rng)
        cmd = self.sample_command(cmd_rng)

        # Sample push interval.
        rng, push_rng = jax.random.split(rng)
        push_interval = jax.random.uniform(
            push_rng,
            minval=self._config.push_config.interval_range[0],
            maxval=self._config.push_config.interval_range[1],
        )
        push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

        info = {
            "rng": rng,
            "step": 0,
            "command": cmd,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "motor_targets": jp.zeros(self.mjx_model.nu),
            "feet_air_time": jp.zeros(2),
            "last_contact": jp.zeros(2, dtype=bool),
            "swing_peak": jp.zeros(2),
            # Phase related.
            "phase_dt": phase_dt,
            "phase": phase,
            "gait_freq": gait_freq,
            # Push related.
            "push": jp.array([0.0, 0.0]),
            "push_step": 0,
            "push_interval_steps": push_interval_steps,
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())
        metrics["swing_peak"] = jp.zeros(())

        # contact = jp.array([
        #     geoms_colliding(data, geom_id, self._floor_geom_id)
        #     for geom_id in self._feet_geom_id
        # ])
        contact = []
        for geom_id in self._feet_geom_id:
            contact.append(geoms_colliding(data, geom_id, self._floor_geom_id))
        contact = jp.array(contact)

        obs = self._get_obs(data, info, contact)

        # Initialize state history buffer with current state repeated 10 times
        state_dim = obs["state"].shape[0]
        initial_state_history = jp.tile(obs["state"], (self._state_history_len, 1))
        info["state_history"] = initial_state_history

        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state.info["rng"], push1_rng, push2_rng = jax.random.split(
            state.info["rng"], 3
        )
        push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
        push_magnitude = jax.random.uniform(
            push2_rng,
            minval=self._config.push_config.magnitude_range[0],
            maxval=self._config.push_config.magnitude_range[1],
        )
        push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
        push *= (
                jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"])
                == 0
        )
        push *= self._config.push_config.enable
        qvel = state.data.qvel
        qvel = qvel.at[:2].set(push * push_magnitude + qvel[:2])
        data = state.data.replace(qvel=qvel)
        state = state.replace(data=data)

        motor_targets = self._default_pose + action * self._config.action_scale
        data = mjx_env.step(
            self.mjx_model, state.data, motor_targets, self.n_substeps
        )
        state.info["motor_targets"] = motor_targets

        contact = jp.array([
            geoms_colliding(data, geom_id, self._floor_geom_id)
            for geom_id in self._feet_geom_id
        ])
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        state.info["feet_air_time"] += self.dt
        p_f = data.site_xpos[self._feet_site_id]
        p_fz = p_f[..., -1]
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

        obs = self._get_obs(data, state.info, contact)

        # Update state history buffer - shift left and add new state
        new_state_history = jp.concatenate([
            state.info["state_history"][1:],  # Remove oldest
            obs["state"][None, :]  # Add newest at the end
        ], axis=0)
        state.info["state_history"] = new_state_history

        done = self._get_termination(data)

        rewards = self._get_reward(
            data, action, state.info, state.metrics, done, first_contact, contact
        )
        rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        }
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        state.info["push"] = push
        state.info["step"] += 1
        state.info["push_step"] += 1
        phase_tp1 = state.info["phase"] + state.info["phase_dt"]
        state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
        state.info["command"] = jp.where(
            state.info["step"] > 500,
            self.sample_command(cmd_rng),
            state.info["command"],
        )
        state.info["step"] = jp.where(
            done | (state.info["step"] > 500),
            0,
            state.info["step"],
        )
        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact
        state.info["swing_peak"] *= ~contact
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        fall_termination = self.get_gravity(data)[-1] < 0.0
        return (
                fall_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        )

    def _get_obs(
            self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
    ) -> mjx_env.Observation:
        gyro = self.get_gyro(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gyro = (
                gyro
                + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
                * self._config.noise_config.level
                * self._config.noise_config.scales.gyro
        )

        gravity = data.site_xmat[self._site_id].T @ jp.array([0, 0, -1])
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gravity = (
                gravity
                + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
                * self._config.noise_config.level
                * self._config.noise_config.scales.gravity
        )

        joint_angles = data.qpos[7:]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_angles = (
                joint_angles
                + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
                * self._config.noise_config.level
                * self._qpos_noise_scale
        )

        joint_vel = data.qvel[6:]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_vel = (
                joint_vel
                + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
                * self._config.noise_config.level
                * self._config.noise_config.scales.joint_vel
        )

        cos = jp.cos(info["phase"])
        sin = jp.sin(info["phase"])
        phase = jp.concatenate([cos, sin])

        linvel = self.get_local_linvel(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_linvel = (
                linvel
                + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
                * self._config.noise_config.level
                * self._config.noise_config.scales.linvel
        )

        state = jp.hstack([
            # noisy_linvel,  # 3
            noisy_gyro,  # 3
            noisy_gravity,  # 3
            info["command"],  # 3
            noisy_joint_angles - self._default_pose,  # 6
            noisy_joint_vel,  # 6
            info["last_act"],  # 6
            phase,
        ])

        accelerometer = self.get_accelerometer(data)
        global_angvel = self.get_global_angvel(data)
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
        root_height = data.qpos[2]

        privileged_state = jp.hstack([
            state,
            gyro,  # 3
            accelerometer,  # 3
            gravity,  # 3
            linvel,  # 3
            global_angvel,  # 3
            joint_angles - self._default_pose,
            joint_vel,
            root_height,  # 1
            data.actuator_force,  # 6
            contact,  # 2
            feet_vel,  # 2*3
            info["feet_air_time"],  # 2
            info["gait_freq"],
        ])

        # Get state history if available, otherwise return zeros
        state_history = info.get("state_history", jp.zeros((self._state_history_len, state.shape[0])))

        return {
            "state": state,
            "privileged_state": privileged_state,
            "state_history": state_history.flatten(),
        }

    def _get_reward(
            self,
            data: mjx.Data,
            action: jax.Array,
            info: dict[str, Any],
            metrics: dict[str, Any],
            done: jax.Array,
            first_contact: jax.Array,
            contact: jax.Array,
    ) -> dict[str, jax.Array]:
        del metrics  # Unused.
        return {
            # Tracking rewards.
            "tracking_lin_vel": self._reward_tracking_lin_vel(
                info["command"], self.get_local_linvel(data)
            ),
            "tracking_ang_vel": self._reward_tracking_ang_vel(
                info["command"], self.get_gyro(data)
            ),
            # "stay_still": self._reward_stay_still(
            #     info["command"], self.get_local_linvel(data), self.get_gyro(data)
            # ),
            # Base-related rewards.
            "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
            "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
            "orientation": self._cost_orientation(self.get_gravity(data)),
            "base_height": self._cost_base_height(data.qpos[2]),
            # Energy related rewards.
            "torques": self._cost_torques(data.actuator_force),
            "action_rate": self._cost_action_rate(
                action, info["last_act"], info["last_last_act"]
            ),
            "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
            "joint_acc": self._cost_joint_acc(data.qacc[6:]),
            # Feet related rewards.
            "feet_distance": self._cost_feet_distance(data),
            "feet_regulation": self._cost_feet_regulation(data),
            "feet_landing_vel": self._cost_feet_landing_vel(data, contact),
            "feet_slip": self._cost_feet_slip(data, contact, info),
            "feet_clearance": self._cost_feet_clearance(data, info),
            "feet_height": self._cost_feet_height(
                info["swing_peak"], first_contact, info
            ),
            "feet_air_time": self._reward_feet_air_time(
                info["feet_air_time"], first_contact, info["command"]
            ),
            "feet_phase": self._reward_feet_phase(
                data,
                info["phase"],
                self._config.reward_config.max_foot_height,
                info["command"],
            ),
            # Other rewards.
            "alive": self._reward_alive(),
            "termination": self._cost_termination(done),
            "stand_still": self._cost_stand_still(info["command"], data.qpos[7:]),
            # Pose related rewards.
            "joint_deviation_hip": self._cost_joint_deviation_hip(
                data.qpos[7:], info["command"]
            ),
            "joint_deviation_knee": self._cost_joint_deviation_knee(data.qpos[7:]),
            "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
            "pose": self._cost_pose(data.qpos[7:]),
        }

    # Tracking rewards.

    def _reward_tracking_lin_vel(
            self,
            commands: jax.Array,
            local_vel: jax.Array,
    ) -> jax.Array:
        lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
        return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

    def _reward_tracking_ang_vel(
            self,
            commands: jax.Array,
            ang_vel: jax.Array,
    ) -> jax.Array:
        ang_vel_error = jp.square(commands[2] - ang_vel[2])
        return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

    # def _reward_stay_still(
    #         self,
    #         commands: jax.Array,
    #         local_vel: jax.Array,
    #         ang_vel: jax.Array,
    # ) -> jax.Array:
    #     cmd_norm = jp.linalg.norm(commands)
    #
    #     # Only apply this reward when commands are near zero
    #     is_stationary_cmd = cmd_norm < 0.1
    #
    #     # Penalize any movement when commands are zero
    #     lin_vel_penalty = jp.sum(jp.square(local_vel[:2]))
    #     ang_vel_penalty = jp.square(ang_vel[2])
    #
    #     # Use a steep exponential to heavily penalize movement when stationary
    #     movement_penalty = lin_vel_penalty + ang_vel_penalty
    #     reward = jp.exp(-movement_penalty / 0.01)  # Very steep curve
    #
    #     # Only apply reward when commands are near zero
    #     return jp.where(is_stationary_cmd, reward, jp.array(0.0))

    # Base-related rewards.

    def _cost_lin_vel_z(self, global_linvel) -> jax.Array:
        return jp.square(global_linvel[2])

    def _cost_ang_vel_xy(self, global_angvel) -> jax.Array:
        return jp.sum(jp.square(global_angvel[:2]))

    def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
        return jp.sum(jp.square(torso_zaxis[:2]))

    def _cost_base_height(self, base_height: jax.Array) -> jax.Array:
        return jp.square(
            base_height - self._config.reward_config.base_height_target
        )

    # Energy related rewards.

    def _cost_torques(self, torques: jax.Array) -> jax.Array:
        return jp.sum(jp.abs(torques))

    def _cost_energy(
            self, qvel: jax.Array, qfrc_actuator: jax.Array
    ) -> jax.Array:
        return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

    def _cost_joint_acc(self, qcc: jax.Array) -> jax.Array:
        return jp.sum(jp.square(qcc))

    def _cost_action_rate(
            self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
    ) -> jax.Array:
        del last_last_act  # Unused.
        c1 = jp.sum(jp.square(act - last_act))
        return c1

    # Other rewards.

    def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
        out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
        out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
        return jp.sum(out_of_limits)

    def _cost_stand_still(
            self,
            commands: jax.Array,
            qpos: jax.Array,
    ) -> jax.Array:
        cmd_norm = jp.linalg.norm(commands)
        return jp.sum(jp.abs(qpos - self._default_pose)) * (cmd_norm < 0.1)

    def _cost_termination(self, done: jax.Array) -> jax.Array:
        return done

    def _reward_alive(self) -> jax.Array:
        return jp.array(1.0)

    # Pose-related rewards.

    def _cost_joint_deviation_hip(
            self, qpos: jax.Array, cmd: jax.Array
    ) -> jax.Array:
        cost = jp.sum(
            jp.abs(qpos[self._hip_indices] - self._default_pose[self._hip_indices])
        )
        cost *= jp.abs(cmd[1]) > 0.1
        return cost

    def _cost_joint_deviation_knee(self, qpos: jax.Array) -> jax.Array:
        return jp.sum(
            jp.abs(
                qpos[self._knee_indices] - self._default_pose[self._knee_indices]
            )
        )

    def _cost_pose(self, qpos: jax.Array) -> jax.Array:
        return jp.sum(jp.square(qpos - self._default_pose) * self._weights)

    # Feet related rewards.

    def _cost_feet_distance(self, data: mjx.Data) -> jax.Array:
        feet_pos = data.site_xpos[self._feet_site_id]
        feet_dist = jp.linalg.norm(feet_pos[0] - feet_pos[1])
        reward = jp.clip(self._config.reward_config.min_feet_distance - feet_dist, 0.0, 1.0)
        return reward

    def _cost_feet_regulation(self, data: mjx.Data) -> jax.Array:
        feet_height = self._config.reward_config.base_height_target * 0.001
        foot_heights = data.site_xpos[self._feet_site_id, 2]  # z-coordinates of feet
        foot_velocities = data.sensordata[self._foot_linvel_sensor_adr]  # foot linear velocities
        foot_velocities_xy = foot_velocities[:, :2]  # only x-y components

        cost = jp.sum(
            jp.exp(-foot_heights / feet_height)
            * jp.square(jp.linalg.norm(foot_velocities_xy, axis=-1))
        )
        return cost

    def _cost_feet_landing_vel(self, data: mjx.Data, contact: jax.Array) -> jax.Array:
        foot_velocities = data.sensordata[self._foot_linvel_sensor_adr]  # foot linear velocities
        z_vels = foot_velocities[:, 2]  # z-components of foot velocities
        foot_heights = data.site_xpos[self._feet_site_id, 2]  # z-coordinates of feet

        # Check if feet are about to land: low height, not in contact, and moving downward
        about_to_land = (
                (foot_heights < self._config.reward_config.about_landing_threshold)
                & (~contact)
                & (z_vels < 0.0)
        )

        # Only penalize downward velocities when about to land
        landing_z_vels = jp.where(about_to_land, z_vels, jp.zeros_like(z_vels))
        cost = jp.sum(jp.square(landing_z_vels))
        return cost

    def _cost_feet_slip(
            self, data: mjx.Data, contact: jax.Array, info: dict[str, Any]
    ) -> jax.Array:
        del info  # Unused.
        body_vel = self.get_global_linvel(data)[:2]
        reward = jp.sum(jp.linalg.norm(body_vel, axis=-1) * contact)
        return reward

    def _cost_feet_clearance(
            self, data: mjx.Data, info: dict[str, Any]
    ) -> jax.Array:
        del info  # Unused.
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
        vel_xy = feet_vel[..., :2]
        vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
        foot_pos = data.site_xpos[self._feet_site_id]
        foot_z = foot_pos[..., -1]
        delta = jp.abs(foot_z - self._config.reward_config.max_foot_height)
        return jp.sum(delta * vel_norm)

    def _cost_feet_height(
            self,
            swing_peak: jax.Array,
            first_contact: jax.Array,
            info: dict[str, Any],
    ) -> jax.Array:
        del info  # Unused.
        error = swing_peak / self._config.reward_config.max_foot_height - 1.0
        return jp.sum(jp.square(error) * first_contact)

    def _reward_feet_air_time(
            self,
            air_time: jax.Array,
            first_contact: jax.Array,
            commands: jax.Array,
            threshold_min: float = 0.2,
            threshold_max: float = 0.5,
    ) -> jax.Array:
        cmd_norm = jp.linalg.norm(commands)
        air_time = (air_time - threshold_min) * first_contact
        air_time = jp.clip(air_time, max=threshold_max - threshold_min)
        reward = jp.sum(air_time)
        reward *= cmd_norm > 0.1  # No reward for zero commands.
        return reward

    def _reward_feet_phase(
            self,
            data: mjx.Data,
            phase: jax.Array,
            foot_height: jax.Array,
            commands: jax.Array,
    ) -> jax.Array:
        # Reward for tracking the desired foot height.
        del commands  # Unused.
        foot_pos = data.site_xpos[self._feet_site_id]
        foot_z = foot_pos[..., -1]
        rz = gait.get_rz(phase, swing_height=foot_height)
        error = jp.sum(jp.square(foot_z - rz))
        reward = jp.exp(-error / 0.01)
        return reward

    def sample_command(self, rng: jax.Array) -> jax.Array:
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

        lin_vel_x = jax.random.uniform(
            rng1, minval=self._config.lin_vel_x[0], maxval=self._config.lin_vel_x[1]
        )
        lin_vel_y = jax.random.uniform(
            rng2, minval=self._config.lin_vel_y[0], maxval=self._config.lin_vel_y[1]
        )
        ang_vel_yaw = jax.random.uniform(
            rng3,
            minval=self._config.ang_vel_yaw[0],
            maxval=self._config.ang_vel_yaw[1],
        )

        # With 10% chance, set everything to zero.
        return jp.where(
            jax.random.bernoulli(rng4, p=0.1),
            jp.zeros(3),
            jp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw]),
        )
