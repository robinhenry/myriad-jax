from .bio.ccasr_gfp.rendering import render_frame as render_ccasr_gfp_frame
from .bio.ccasr_gfp.tasks import control as ccasr_gfp_control
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
    "pendulum-control",
    pendulum_control.make_env,
    pendulum_control.ControlTaskConfig,
    render_frame_fn=render_pendulum_frame_from_obs,
)
register_env(
    "ccasr-gfp-control",
    ccasr_gfp_control.make_env,
    ccasr_gfp_control.ControlTaskConfig,
    render_frame_fn=render_ccasr_gfp_frame,
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
