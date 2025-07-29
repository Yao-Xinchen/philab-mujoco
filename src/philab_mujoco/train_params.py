from ml_collections import config_dict

from philab_mujoco import locomotion


def brax_ppo_config(env_name: str) -> config_dict.ConfigDict:
    env_config = locomotion.get_default_config(env_name)

    rl_config = config_dict.create(
        num_timesteps=100_000_000,
        num_evals=10,
        reward_scaling=1.0,
        episode_length=env_config.episode_length,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        num_envs=8192,
        batch_size=256,
        max_grad_norm=1.0,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(128, 128, 128, 128),
            value_hidden_layer_sizes=(256, 256, 256, 256, 256),
            policy_obs_key="state",
            value_obs_key="state",
        ),
    )

    if env_name in ("TronSfJoystick", "TronPfJoystick"):
        rl_config.num_timesteps = 500_000_000
        rl_config.num_evals = 25
        rl_config.clipping_epsilon = 0.2
        rl_config.num_resets_per_eval = 1
        rl_config.entropy_cost = 0.005
        rl_config.network_factory = config_dict.create(
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
            encoder_hidden_layer_sizes=(256, 64, 16),
            policy_obs_key="state",
            value_obs_key="privileged_state",
            encoder_obs_key="state_history",
        )
    else:
        raise ValueError(f"Unsupported env: {env_name}")

    return rl_config
