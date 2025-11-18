"""
Utility functions for cart-pendulum control experiments.

This module provides helper functions for:
- State/observation conversions
- Observation normalization (for fairness between RL and classical control)
- Plotting and visualization
- Metrics computation
"""

import math
from typing import Optional
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize


def state_to_obs(state: np.ndarray) -> np.ndarray:
    """
    Convert state representation to observation representation.

    The observation uses [sin(θ), cos(θ)] instead of raw angle θ to avoid
    discontinuity at ±π. This is critical for neural network policies.

    Args:
        state: State vector [θ, θ̇, x, ẋ]

    Returns:
        Observation vector [sin(θ), cos(θ), θ̇, x, ẋ]

    Example:
        >>> state = np.array([np.pi/4, 0.5, 0.1, -0.2])
        >>> obs = state_to_obs(state)
        >>> obs.shape
        (5,)
    """
    theta, theta_dot, x, x_dot = state
    return np.array(
        [math.sin(theta), math.cos(theta), theta_dot, x, x_dot],
        dtype=np.float32
    )


def obs_to_state(obs: np.ndarray) -> np.ndarray:
    """
    Convert observation representation back to state representation.

    Recovers angle θ from [sin(θ), cos(θ)] using atan2.

    Args:
        obs: Observation vector [sin(θ), cos(θ), θ̇, x, ẋ]

    Returns:
        State vector [θ, θ̇, x, ẋ]

    Note:
        The recovered angle θ will be in [-π, π].

    Example:
        >>> obs = np.array([0.707, 0.707, 0.5, 0.1, -0.2])
        >>> state = obs_to_state(obs)
        >>> state[0]  # angle in radians
        0.7853981633974483
    """
    sin_theta, cos_theta, theta_dot, x, x_dot = obs
    theta = math.atan2(sin_theta, cos_theta)
    return np.array([theta, theta_dot, x, x_dot], dtype=np.float32)


def normalize_obs_from_state(
    vec_env: VecNormalize,
    state: np.ndarray
) -> np.ndarray:
    """
    Convert state to normalized observation using VecNormalize statistics.

    This function is critical for fairness when comparing RL and classical control:
    - RL policies are trained on normalized observations
    - Classical controllers should also receive normalized observations for fair comparison

    Args:
        vec_env: VecNormalize wrapper with computed statistics
        state: State vector [θ, θ̇, x, ẋ]

    Returns:
        Normalized observation vector [sin(θ), cos(θ), θ̇, x, ẋ]
        with zero mean and unit variance (based on training statistics)

    Example:
        >>> from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
        >>> # Assuming vec_env is a trained VecNormalize wrapper
        >>> state = np.array([0.1, 0.0, 0.0, 0.0])
        >>> norm_obs = normalize_obs_from_state(vec_env, state)
    """
    obs = state_to_obs(state)
    # VecNormalize expects shape (n_envs, obs_dim)
    return vec_env.normalize_obs(np.array([obs], dtype=np.float32))


def normalize_obs_array(
    vec_env: VecNormalize,
    obs: np.ndarray
) -> np.ndarray:
    """
    Normalize an observation using VecNormalize statistics.

    Args:
        vec_env: VecNormalize wrapper with computed statistics
        obs: Observation vector [sin(θ), cos(θ), θ̇, x, ẋ]

    Returns:
        Normalized observation
    """
    return vec_env.normalize_obs(np.array([obs], dtype=np.float32))


def denormalize_obs(
    vec_env: VecNormalize,
    norm_obs: np.ndarray
) -> np.ndarray:
    """
    Denormalize an observation to original scale.

    Args:
        vec_env: VecNormalize wrapper with computed statistics
        norm_obs: Normalized observation vector

    Returns:
        Original-scale observation [sin(θ), cos(θ), θ̇, x, ẋ]
    """
    # Get running mean and variance
    obs_mean = vec_env.obs_rms.mean
    obs_var = vec_env.obs_rms.var
    obs_std = np.sqrt(obs_var + 1e-8)

    # Denormalize: obs = norm_obs * std + mean
    return norm_obs * obs_std + obs_mean


def compute_angle_from_obs(obs: np.ndarray) -> float:
    """
    Extract angle (in radians) from observation vector.

    Args:
        obs: Observation [sin(θ), cos(θ), θ̇, x, ẋ]

    Returns:
        Angle θ in radians, range [-π, π]
    """
    sin_theta, cos_theta = obs[0], obs[1]
    return math.atan2(sin_theta, cos_theta)


def wrap_angle(theta: float) -> float:
    """
    Wrap angle to [-π, π] range.

    Args:
        theta: Angle in radians

    Returns:
        Wrapped angle in [-π, π]
    """
    return math.atan2(math.sin(theta), math.cos(theta))


def compute_energy(state: np.ndarray, m: float = 0.1, l: float = 1.0, g: float = 9.81) -> float:
    """
    Compute total mechanical energy of the cart-pendulum system.

    Energy = KE_cart + KE_pendulum + PE_pendulum

    Args:
        state: State vector [θ, θ̇, x, ẋ]
        m: Pole mass (kg)
        l: Pole length (m)
        g: Gravitational acceleration (m/s²)

    Returns:
        Total energy (J)

    Note:
        For upright equilibrium (θ=0, all velocities=0), energy should be ≈ m·g·l
    """
    theta, theta_dot, x, x_dot = state

    # Kinetic energy of cart
    KE_cart = 0.5 * 1.0 * x_dot**2  # M=1.0 kg

    # Kinetic energy of pendulum
    KE_pendulum = 0.5 * m * l**2 * theta_dot**2

    # Potential energy of pendulum (reference at hanging down)
    # Height of center of mass: h = l * (1 - cos(θ))
    PE_pendulum = m * g * l * (1 - math.cos(theta))

    return KE_cart + KE_pendulum + PE_pendulum


def check_device_available(device: str) -> str:
    """
    Check if requested device is available and return valid device string.

    Args:
        device: Requested device ('cuda', 'cpu', or 'auto')

    Returns:
        Valid device string ('cuda' or 'cpu')

    Raises:
        ImportError: If torch is not installed
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for device checking. Install with: pip install torch")

    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    elif device == 'cuda':
        if not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
            return 'cpu'
        return 'cuda'
    else:
        return 'cpu'


def ensure_directory_exists(path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path
    """
    import os
    os.makedirs(path, exist_ok=True)


def format_time(seconds: float) -> str:
    """
    Format time duration in human-readable form.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "1h 23m 45s" or "2m 15s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


class NumpyEncoder:
    """Helper class for encoding numpy arrays to JSON-compatible format."""

    @staticmethod
    def default(obj):
        """Convert numpy types to Python native types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
