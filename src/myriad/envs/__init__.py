from .bio.ccas_ccar.rendering import render_ccas_ccar_frame
from .bio.ccas_ccar.tasks import control as ccas_ccar_control
from .classic.cartpole.rendering import render_cartpole_frame_from_obs
from .classic.cartpole.tasks import control as cartpole_control
from .classic.pendulum.rendering import render_pendulum_frame_from_obs
from .classic.pendulum.tasks import control as pendulum_control
from .environment import (
    Environment,
    EnvironmentConfig,
    EnvironmentParams,
    EnvironmentState,
)
from .registration import EnvInfo, get_env_info, list_envs, make_env, register_env

# Register built-in environments
register_env(
    "cartpole-control",
    cartpole_control.make_env,
    cartpole_control.ControlTaskConfig,
    render_frame_fn=render_cartpole_frame_from_obs,
)
register_env(
    "cartpole-sysid",
    cartpole_control.make_env,
    cartpole_control.ControlTaskConfig,
    render_frame_fn=render_cartpole_frame_from_obs,
)
register_env(
    "pendulum-control",
    pendulum_control.make_env,
    pendulum_control.ControlTaskConfig,
    render_frame_fn=render_pendulum_frame_from_obs,
)
register_env(
    "ccas-ccar-control",
    ccas_ccar_control.make_env,
    ccas_ccar_control.ControlTaskConfig,
    render_frame_fn=render_ccas_ccar_frame,
)
register_env(
    "ccas-ccar-sysid",
    ccas_ccar_control.make_env,
    ccas_ccar_control.ControlTaskConfig,
    render_frame_fn=render_ccas_ccar_frame,
)

__all__ = [
    "make_env",
    "Environment",
    "EnvironmentConfig",
    "EnvironmentParams",
    "EnvironmentState",
    "EnvInfo",
    "register_env",
    "get_env_info",
    "list_envs",
]
