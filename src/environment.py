"""
Cart-Pendulum Gymnasium Environment (Standard RL Convention)

This module implements the cart-pendulum system with standard RL conventions.

CRITICAL CONVENTIONS (STANDARD RL):
    - θ ∈ (-π, π]  (angle wraps at ±π)
    - θ = 0 at TOP (upright, target state)
    - θ = ±π at BOTTOM (hanging down, wrapping point)
    - Positive θ is counter-clockwise rotation

Physics Model (Simplified, m << M approximation):
    θ̈ = -g/l·sin(θ) - 2ζ·θ̇ - (u/l)·cos(θ)
    ẍ = u

    With physical parameters:
        l = 1.0 m (pendulum length)
        g = 9.81 m/s² (gravity)
        ζ = friction coefficient (dimensionless)
        u = control force/mass (m/s²)

    State: [θ, θ̇, x, ẋ]
        - θ: angle from bottom (rad), wrapped to (-π,π]
        - θ̇: angular velocity (rad/s)
        - x: cart position (m), limit at ±2.4m
        - ẋ: cart velocity (m/s)

Key Features:
    - RK4 integration with 10 substeps per control step
    - Control timestep: dt = 0.02s (50 Hz)
    - Track limits: ±2.4m (hard termination)
    - Simplified dynamics matching reference exactly
"""

from typing import Tuple, Union, Optional
import math
import numpy as np
import gymnasium as gym


class CartPendulumEnv(gym.Env):
    """
    Cart-Pendulum environment with standard RL convention.

    CRITICAL: θ ∈ (-π, π], θ=0 at TOP (upright target), θ=±π at BOTTOM
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50
    }

    def __init__(
        self,
        curriculum_phase: str = "swingup",
        zeta: Union[float, Tuple[float, float]] = 0.01,  # friction coefficient
        rk4_substeps: int = 10,
        dt: float = 0.02,  # control timestep (seconds)
        xmax: float = 2.4,  # track limits at ±xmax (meters)
        umax: float = 20.0,  # control limit (m/s²)
        soft_wall_start: float = 2.0,
        soft_wall_k: float = 0.0,
        du_weight: float = 1e-4,
        seed: Optional[int] = None,
        stabilization_prob: float = 0.0,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize Cart-Pendulum environment (reference-matched).

        Args:
            curriculum_phase: "swingup" or "stabilization"
            zeta: Angular friction coefficient (dimensionless)
                - Reference uses ζ=0.01 for simulations
                - Can be tuple (min, max) for domain randomization
            rk4_substeps: Number of RK4 substeps per control step
            dt: Control timestep (seconds), default 0.02s = 50 Hz
            xmax: Track limit (meters), default 2.4m
            umax: Control acceleration limit (m/s²), default 20.0
            soft_wall_start: Where soft wall penalty begins
            soft_wall_k: Soft wall penalty coefficient
            du_weight: Action smoothness penalty
            seed: Random seed
            stabilization_prob: Probability of near-upright start in swingup
            render_mode: 'human' or 'rgb_array'
        """
        super().__init__()

        # Physical parameters
        self.l = 1.0  # pendulum length (m)
        self.g = 9.81  # gravity (m/s²)

        # Time parameters
        self.dt = float(dt)
        self.n_substeps = int(rk4_substeps)
        self.dt_int = self.dt / self.n_substeps

        # Limits
        self.xmax = float(xmax)
        self.umax = float(umax)

        # Friction (can be randomized)
        self.zeta_config = zeta
        self._zeta_ep = float(zeta) if not hasattr(zeta, "__len__") else None

        # Curriculum
        self.curriculum_phase = curriculum_phase
        self.stabilization_prob = float(stabilization_prob)

        # Reward shaping
        self.soft_wall_start = float(soft_wall_start)
        self.soft_wall_k = float(soft_wall_k)
        self.du_weight = float(du_weight)

        # Gymnasium spaces
        self.action_space = gym.spaces.Box(
            low=-self.umax, high=self.umax, shape=(1,), dtype=np.float32
        )

        # Observation: [sin(θ), cos(θ), θ̇, x, ẋ]
        obs_limit = np.array([1.0, 1.0, 15.0, self.xmax, 10.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-obs_limit, high=obs_limit, dtype=np.float32
        )

        # State variables
        self.state = None  # [theta, theta_dot, x, x_dot]
        self._last_u = 0.0

        # Random number generator
        if seed is not None:
            self.seed(seed)

        # Rendering
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

    def _sample_friction_for_episode(self):
        """Sample friction coefficient for this episode."""
        if hasattr(self.zeta_config, "__len__"):
            self._zeta_ep = float(
                self.np_random.uniform(self.zeta_config[0], self.zeta_config[1])
            )

    def _dyn(self, state: np.ndarray, u: float) -> np.ndarray:
        """
        Compute state derivatives (SIMPLIFIED DYNAMICS matching reference).

        Equations (physical units):
            θ̈ = -(g/l)·sin(θ) - 2ζ·θ̇ - (u/l)·cos(θ)
            ẍ = u

        This assumes m << M (light pendulum approximation).

        Args:
            state: [θ, θ̇, x, ẋ]
            u: control acceleration (m/s²)

        Returns:
            [θ̇, θ̈, ẋ, ẍ]
        """
        theta, theta_dot, x, x_dot = state
        zeta = self._zeta_ep if self._zeta_ep is not None else float(self.zeta_config)

        # Simplified pendulum equation (matching reference)
        theta_ddot = -(self.g / self.l) * math.sin(theta) - 2 * zeta * theta_dot - (u / self.l) * math.cos(theta)

        # Cart equation (direct acceleration)
        x_ddot = u

        return np.array([theta_dot, theta_ddot, x_dot, x_ddot], dtype=np.float64)

    def _rk4_step(self, state: np.ndarray, u: float, dt: float) -> np.ndarray:
        """Single RK4 integration step."""
        k1 = self._dyn(state, u)
        k2 = self._dyn(state + 0.5 * dt * k1, u)
        k3 = self._dyn(state + 0.5 * dt * k2, u)
        k4 = self._dyn(state + dt * k3, u)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _integrate_rk4(self, state: np.ndarray, u: float) -> np.ndarray:
        """Integrate over one control timestep."""
        s = state.copy()
        for _ in range(self.n_substeps):
            s = self._rk4_step(s, u, self.dt_int)
        return s

    def _get_obs(self, state: np.ndarray) -> np.ndarray:
        """
        Convert state to observation.

        Uses [sin(θ), cos(θ), θ̇, x, ẋ] to avoid angle wrapping discontinuities.

        CRITICAL: With standard convention (θ=0 at top):
            - sin(0)=0, cos(0)=+1 → upright (target)
            - sin(±π)=0, cos(±π)=-1 → hanging down
        """
        theta, theta_dot, x, x_dot = state
        return np.array(
            [math.sin(theta), math.cos(theta), theta_dot, x, x_dot],
            dtype=np.float32
        )

    def _compute_reward(self, state: np.ndarray, u: float) -> float:
        """
        Compute reward (STANDARD RL CONVENTION).

        Target state: θ=0 (upright), x=0

        Convention: θ ∈ (-π, π], θ=0 at TOP, θ=±π at BOTTOM

        Reward components:
            1. Angle: cos(θ) + 1 (0 at bottom ±π, +2 at top 0)
            2. Angular velocity: -0.05·θ̇²
            3. Position: -0.15·x²
            4. Control: -0.01·u²
            5. Smoothness: -du_weight·(u - u_prev)²
            6. Soft wall: -soft_wall_k·overshoot²
            7. Success bonus: +10.0 if near upright (θ≈0)
        """
        theta, theta_dot, x, x_dot = state

        # Angle cost: reward being near θ=0 (upright)
        # cos(θ) = +1 at θ=0 (upright), -1 at θ=±π (hanging)
        # So: cos(θ) + 1 ranges from 2 (upright) to 0 (hanging)
        reward = math.cos(theta) + 1.0

        # Angular velocity damping
        reward -= 0.05 * theta_dot**2

        # Position cost
        reward -= 0.15 * x**2

        # Control effort
        reward -= 0.01 * u**2

        # Action smoothness
        du = u - self._last_u
        reward -= self.du_weight * du**2

        # Soft wall penalty
        if self.soft_wall_k > 0.0:
            if abs(x) > self.soft_wall_start:
                overshoot = abs(x) - self.soft_wall_start
                reward -= self.soft_wall_k * overshoot**2

        # Success bonus: near upright (θ ≈ 0) and centered (x ≈ 0)
        theta_error = abs(theta)  # Distance from θ=0 (upright)
        if theta_error < 0.2 and abs(x) < 0.2:
            reward += 10.0

        return float(reward)

    def _is_terminated(self, state: np.ndarray) -> bool:
        """
        Check termination (cart hitting walls).

        Terminates when |x| > xmax.
        """
        x = state[2]
        return abs(x) > self.xmax

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset environment.

        Convention: θ ∈ (-π, π], θ=0 at top (upright target)

        Phase 1 (stabilization): Start near upright
            θ ∈ [-0.2, +0.2] (near θ=0)

        Phase 2 (swingup): Start anywhere
            θ ∈ (-π, π]
        """
        super().reset(seed=seed)

        # Sample friction
        self._sample_friction_for_episode()

        # Sample initial state based on curriculum
        if self.curriculum_phase == "stabilization":
            # Near upright (θ ≈ 0)
            theta = self.np_random.uniform(-0.2, 0.2)
            x = self.np_random.uniform(-0.3, 0.3)
        else:  # swingup
            if self.np_random.random() < self.stabilization_prob:
                # Occasionally near upright (prevent forgetting)
                theta = self.np_random.uniform(-0.2, 0.2)
                x = self.np_random.uniform(-0.3, 0.3)
            else:
                # Full random (anywhere on circle)
                theta = self.np_random.uniform(-math.pi, math.pi)
                x = self.np_random.uniform(-0.5, 0.5)

        # Wrap theta to (-π, π]
        theta = ((theta + math.pi) % (2 * math.pi)) - math.pi

        # Start from rest
        theta_dot = 0.0
        x_dot = 0.0

        self.state = np.array([theta, theta_dot, x, x_dot], dtype=np.float64)
        self._last_u = 0.0

        return self._get_obs(self.state), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one control step."""
        # Extract and clip control
        u = float(np.clip(action[0], -self.umax, self.umax))

        # Integrate dynamics
        self.state = self._integrate_rk4(self.state, u)

        # Wrap theta to (-π, π]
        self.state[0] = ((self.state[0] + math.pi) % (2 * math.pi)) - math.pi

        # Compute reward
        reward = self._compute_reward(self.state, u)

        # Check termination
        terminated = self._is_terminated(self.state)

        # Sparse failure penalty
        if terminated:
            reward -= 500.0

        # Update smoothness tracking
        self._last_u = u

        # Get observation
        obs = self._get_obs(self.state)

        return obs, reward, terminated, False, {}

    def get_state(self) -> np.ndarray:
        """Get current state."""
        return self.state.copy()

    def set_state(self, state: np.ndarray) -> None:
        """Set state (for testing)."""
        self.state = np.array(state, dtype=np.float64)
        self._last_u = 0.0

    def render(self):
        """Render the environment (standard RL: θ=0 at top, θ=±π at bottom)."""
        if self.render_mode is None:
            return None

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise gym.error.DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        screen_width = 600
        screen_height = 400
        world_width = 2 * self.xmax + 1.0
        scale = screen_width / world_width

        cart_width = 50.0
        cart_height = 30.0
        pole_width = 10.0
        pole_length = scale * 2.0  # Pole length in pixels

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
                pygame.display.set_caption("Cart-Pendulum (θ=0 at top)")
            else:
                self.screen = pygame.Surface((screen_width, screen_height))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Fill background
        self.screen.fill((255, 255, 255))

        # Get state
        theta, theta_dot, x, x_dot = self.state

        # Cart position
        cart_x = int(x * scale + screen_width / 2.0)
        cart_y = int(screen_height * 0.7)

        # Draw rail
        rail_y = cart_y + cart_height / 2
        rail_left = int(screen_width / 2 - self.xmax * scale)
        rail_right = int(screen_width / 2 + self.xmax * scale)
        pygame.draw.line(self.screen, (0, 0, 0),
                        (rail_left, int(rail_y)), (rail_right, int(rail_y)), 4)

        # Draw cart
        cart_rect = pygame.Rect(
            int(cart_x - cart_width / 2),
            int(cart_y - cart_height / 2),
            int(cart_width),
            int(cart_height)
        )
        gfxdraw.box(self.screen, cart_rect, (70, 130, 180))
        pygame.draw.rect(self.screen, (0, 0, 0), cart_rect, 2)

        # Draw pole (STANDARD: θ=0 at top, θ=±π at bottom, positive CCW)
        pole_pivot_x = cart_x
        pole_pivot_y = cart_y

        # θ=0 means upright (pointing up), θ=±π means hanging down
        pole_end_x = pole_pivot_x + pole_length * math.sin(theta)
        pole_end_y = pole_pivot_y - pole_length * math.cos(theta)  # -cos (θ=0 is up)

        pygame.draw.line(
            self.screen,
            (139, 0, 0),
            (int(pole_pivot_x), int(pole_pivot_y)),
            (int(pole_end_x), int(pole_end_y)),
            int(pole_width)
        )

        # Draw pole mass
        gfxdraw.filled_circle(
            self.screen,
            int(pole_end_x),
            int(pole_end_y),
            int(pole_width * 1.5),
            (180, 0, 0)
        )

        # State info
        font = pygame.font.Font(None, 24)
        angle_deg = math.degrees(theta)
        theta_from_top = abs(math.degrees(theta))  # Distance from θ=0 (upright)

        info_lines = [
            f"θ = {angle_deg:6.1f}°",
            f"deviation from upright = {theta_from_top:6.1f}°",
            f"θ̇ = {theta_dot:6.2f}",
            f"x = {x:6.2f}",
            f"ẋ = {x_dot:6.2f}",
        ]

        y_offset = 10
        for line in info_lines:
            text_surface = font.render(line, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 25

        # Upright indicator (θ ≈ 0 is upright)
        if abs(theta) < math.radians(10):
            status_text = "UPRIGHT ✓"
            status_color = (0, 150, 0)
        else:
            status_text = "SWINGING"
            status_color = (150, 0, 0)

        status_surface = font.render(status_text, True, status_color)
        self.screen.blit(status_surface, (screen_width - 150, 10))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        """Close rendering."""
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
