"""
Cart-Pendulum Gymnasium Environment (Reference-Matched Implementation)

This module implements the cart-pendulum system matching the reference FFFB code exactly.

CRITICAL CONVENTIONS (matching reference v6, JB May 2025):
    - θ = 0 at BOTTOM (hanging down)
    - θ = π at TOP (upright, target state)
    - Positive θ is counter-clockwise rotation
    - Time scaled to pendulum period: τ = 2π

Physics Model (Simplified):
    Assumes m << M (light pendulum approximation):
        θ̈ = -sin(θ) - 2ζθ̇ - u·cos(θ)
        ẍ = u

    Where:
        ζ = angular friction coefficient (dimensionless)
        u = control acceleration (dimensionless)

    State: [θ, θ̇, x, ẋ]
        - θ: angle from bottom (rad), θ∈[0,2π) or wrapped to (-π,π)
        - θ̇: angular velocity (rad/τ, dimensionless)
        - x: cart position (dimensionless, limit at ±xmax)
        - ẋ: cart velocity (dimensionless)

Key Features:
    - RK4 integration with configurable substeps
    - Dimensionless time (scaled to pendulum period)
    - Simplified dynamics (m << M approximation)
    - Curriculum learning (stabilization vs swingup)
    - Matches reference FFFB implementation exactly
"""

from typing import Tuple, Union, Optional, Dict
import math
import numpy as np
import gymnasium as gym


class CartPendulumEnv(gym.Env):
    """
    Cart-Pendulum environment matching reference implementation exactly.

    CRITICAL: θ=0 at BOTTOM, θ=π at TOP (upright target)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50
    }

    def __init__(
        self,
        curriculum_phase: str = "swingup",
        zeta: Union[float, Tuple[float, float]] = 0.01,  # "actual" friction for simulation
        rk4_substeps: int = 10,
        xmax: float = 2.0,  # track limits at ±xmax
        umax: float = 5.0,  # control limit (for clipping in practice)
        soft_wall_start: float = 1.8,
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
            zeta: Angular friction coefficient (or range for randomization)
                - Reference uses ζ=0.01 for simulations
                - Can be tuple (min, max) for domain randomization
            rk4_substeps: Number of RK4 substeps per control step
            xmax: Track limit (cart can move in ±xmax)
            umax: Control force limit (for safety clipping)
            soft_wall_start: Where soft wall penalty begins
            soft_wall_k: Soft wall penalty coefficient
            du_weight: Action smoothness penalty
            seed: Random seed
            stabilization_prob: Probability of near-upright start in swingup
            render_mode: 'human' or 'rgb_array'
        """
        super().__init__()

        # Time scaling (dimensionless, scaled to pendulum period)
        # τ = 2π = 1 pendulum period in scaled time
        # Physical dt = 0.02s maps to dimensionless dt based on actual pendulum
        # For now, keep dt in "scaled" units
        self.dt = 2 * np.pi / 100  # ~100 steps per pendulum period
        self.n_substeps = int(rk4_substeps)
        self.dt_int = self.dt / self.n_substeps

        # Physics parameters
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
        # Action: dimensionless acceleration
        self.action_space = gym.spaces.Box(
            low=-self.umax, high=self.umax, shape=(1,), dtype=np.float32
        )

        # Observation: [sin(θ), cos(θ), θ̇, x, ẋ]
        # θ̇ and ẋ are dimensionless (scaled to pendulum period)
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

        Equations (dimensionless):
            θ̈ = -sin(θ) - 2ζθ̇ - u·cos(θ)
            ẍ = u

        This assumes m << M (light pendulum approximation).

        Args:
            state: [θ, θ̇, x, ẋ]
            u: control acceleration (dimensionless)

        Returns:
            [θ̇, θ̈, ẋ, ẍ]
        """
        theta, theta_dot, x, x_dot = state
        zeta = self._zeta_ep if self._zeta_ep is not None else float(self.zeta_config)

        # Simplified pendulum equation (matching reference)
        theta_ddot = -math.sin(theta) - 2 * zeta * theta_dot - u * math.cos(theta)

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

        CRITICAL: With θ=0 at bottom:
            - sin(0)=0, cos(0)=1 → hanging down
            - sin(π)=0, cos(π)=-1 → upright (target)
        """
        theta, theta_dot, x, x_dot = state
        return np.array(
            [math.sin(theta), math.cos(theta), theta_dot, x, x_dot],
            dtype=np.float32
        )

    def _compute_reward(self, state: np.ndarray, u: float) -> float:
        """
        Compute reward (θ=0 at bottom, θ=π at top).

        Target state: θ=π (upright), x=0

        Reward components:
            1. Angle cost: -(π - θ)² penalizes deviation from upright
               OR: cos(θ) + 1 (ranges from 0 at bottom to +2 at top)
            2. Angular velocity: -0.05·θ̇²
            3. Position: -0.15·x²
            4. Control: -0.01·u²
            5. Smoothness: -1e-4·(u - u_prev)²
            6. Soft wall: -soft_wall_k·overshoot²
            7. Success bonus: +10.0 if near upright
        """
        theta, theta_dot, x, x_dot = state

        # Angle cost: reward being near θ=π (upright)
        # cos(θ) = -1 at θ=π (upright), +1 at θ=0 (bottom)
        # So: cos(θ) + 1 ranges from 0 (bottom) to 2 (top)
        reward = math.cos(theta) + 1.0  # 0 at bottom, +2 at top

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

        # Success bonus: near upright (θ ≈ π) and centered (x ≈ 0)
        # θ ∈ [π-0.2, π+0.2] is roughly ±11° from upright
        theta_error = abs(theta - math.pi)  # Distance from upright
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

        Convention: θ=0 at bottom, θ=π at top (upright target)

        Phase 1 (stabilization): Start near upright
            θ ∈ [π-0.2, π+0.2]

        Phase 2 (swingup): Start anywhere
            θ ∈ [0, 2π] or equivalently [-π, π]
        """
        super().reset(seed=seed)

        # Sample friction
        self._sample_friction_for_episode()

        # Sample initial state based on curriculum
        if self.curriculum_phase == "stabilization":
            # Near upright (θ ≈ π)
            theta = self.np_random.uniform(math.pi - 0.2, math.pi + 0.2)
            x = self.np_random.uniform(-0.3, 0.3)
        else:  # swingup
            if self.np_random.random() < self.stabilization_prob:
                # Occasionally near upright (prevent forgetting)
                theta = self.np_random.uniform(math.pi - 0.2, math.pi + 0.2)
                x = self.np_random.uniform(-0.3, 0.3)
            else:
                # Full random (anywhere on circle)
                # Reference starts at θ=0 (bottom)
                # For training variety, sample full range
                theta = self.np_random.uniform(0, 2 * math.pi)
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
        """Render the environment (θ=0 at bottom convention)."""
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
                pygame.display.set_caption("Cart-Pendulum (θ=0 at bottom)")
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

        # Draw pole (θ=0 at bottom, positive CCW)
        pole_pivot_x = cart_x
        pole_pivot_y = cart_y

        # θ=0 means hanging down, θ=π means up
        pole_end_x = pole_pivot_x + pole_length * math.sin(theta)
        pole_end_y = pole_pivot_y + pole_length * math.cos(theta)  # Note: +cos (θ=0 is down)

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
        theta_from_top = abs(math.degrees(theta - math.pi))  # Deviation from upright

        info_lines = [
            f"θ = {angle_deg:6.1f}° (from bottom)",
            f"θ from top = {theta_from_top:6.1f}°",
            f"θ̇ = {theta_dot:6.2f}",
            f"x = {x:6.2f}",
            f"ẋ = {x_dot:6.2f}",
        ]

        y_offset = 10
        for line in info_lines:
            text_surface = font.render(line, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 25

        # Upright indicator
        if abs(theta - math.pi) < math.radians(10):
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
