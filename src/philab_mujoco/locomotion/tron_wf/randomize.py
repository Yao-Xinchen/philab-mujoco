import jax
from mujoco import mjx

FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1


def tron_wf_randomize(model: mjx.Model, rng: jax.Array):
    @jax.vmap
    def rand_dynamics(_rng):
        # Floor friction: =U(0.4, 1.0).
        _rng, _key = jax.random.split(_rng)
        geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
            jax.random.uniform(_key, minval=0.4, maxval=1.0)
        )

        # Scale static friction: *U(0.8, 1.2).
        _rng, _key = jax.random.split(_rng)
        _frictionloss = model.dof_frictionloss[6:] * jax.random.uniform(
            _key, shape=(8,), minval=0.8, maxval=1.2
        )
        dof_frictionloss = model.dof_frictionloss.at[6:].set(_frictionloss)

        # Scale armature: *U(1.0, 1.05).
        _rng, _key = jax.random.split(_rng)
        _armature = model.dof_armature[6:] * jax.random.uniform(
            _key, shape=(8,), minval=1.0, maxval=1.05
        )
        dof_armature = model.dof_armature.at[6:].set(_armature)

        # Scale all link masses: *U(0.9, 1.1).
        _rng, _key = jax.random.split(_rng)
        _dmass = jax.random.uniform(
            _key, shape=(model.nbody,), minval=0.9, maxval=1.1
        )
        _body_mass = model.body_mass.at[:].set(model.body_mass * _dmass)

        # Add mass to torso: +U(-1.0, 1.0).
        _rng, _key = jax.random.split(_rng)
        _dmass = jax.random.uniform(_key, minval=-1.0, maxval=1.0)
        _body_mass = _body_mass.at[TORSO_BODY_ID].set(
            _body_mass[TORSO_BODY_ID] + _dmass
        )

        # Jitter qpos0: +U(-0.07, 0.07).
        _rng, _key = jax.random.split(_rng)
        _qpos0 = model.qpos0
        _qpos0 = _qpos0.at[7:].set(
            _qpos0[7:]
            + jax.random.uniform(_key, shape=(8,), minval=-0.07, maxval=0.07)
        )

        return (
            geom_friction,
            dof_frictionloss,
            dof_armature,
            _body_mass,
            _qpos0,
        )

    (
        friction,
        frictionloss,
        armature,
        body_mass,
        qpos0,
    ) = rand_dynamics(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({
        "geom_friction": 0,
        "dof_frictionloss": 0,
        "dof_armature": 0,
        "body_mass": 0,
        "qpos0": 0,
    })

    model = model.tree_replace({
        "geom_friction": friction,
        "dof_frictionloss": frictionloss,
        "dof_armature": armature,
        "body_mass": body_mass,
        "qpos0": qpos0,
    })

    return model, in_axes
