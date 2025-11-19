"""
Cart-Pendulum Gymnasium Environment

This module implements a high-fidelity cart-pendulum system using RK4 numerical integration.

Physics Model:
    The system consists of a cart of mass M on a frictionless rail with a pendulum of mass m
    and length l attached to it. The state is [theta, theta_dot, x, x_dot] where:
    - theta: pole angle from upward vertical (rad), positive counter-clockwise
    - theta_dot: angular velocity (rad/s)
    - x: cart position (m), positive to the right
    - x_dot: cart velocity (m/s)

    Equations of motion (derived from Lagrangian mechanics):
        (M + m)ẍ + ml(θ̈cos(θ) - θ̇²sin(θ)) = u - c_x·ẋ
        ml²θ̈ + mlẍcos(θ) = mgl·sin(θ) - c_θ·θ̇

    Where:
        M = 1.0 kg (cart mass)
        m = 0.1 kg (pole mass)
        l = 1.0 m (pole length from pivot to center of mass)
        g = 9.81 m/s² (gravity)
        c_theta = angular friction coefficient (N·m·s)
        c_x = linear friction coefficient (N·s/m)
        u = control force applied to cart (N)

Key Features:
    - RK4 integration with configurable substeps for accuracy
    - Domain randomization via friction parameter ranges
    - Curriculum learning support (stabilization vs swingup)
    - Soft wall penalties instead of hard termination
    - Action smoothness penalties for real-world deployment
    - Observation normalization compatible with VecNormalize
"""

from typing import Tuple, Union, Optional, Dict
import math
import numpy as np
import gymnasium as gym


class CartPendulumEnv(gym.Env):
    """
    Cart-Pendulum environment with RK4 integration and configurable physics.

    This environment supports both swing-up and stabilization tasks with
    domain randomization for robust policy learning.

    Attributes:
        metadata (dict): Gymnasium metadata
        dt (float): Control timestep (0.02s = 50Hz)
        n_substeps (int): Number of RK4 integration substeps per control step
        curriculum_phase (str): Either "swingup" or "stabilization"

    Physics Parameters:
        M (float): Cart mass (1.0 kg)
        m (float): Pole mass (0.1 kg)
        l (float): Pole length to center of mass (1.0 m)
        g (float): Gravitational acceleration (9.81 m/s²)
        c_theta (float or tuple): Angular friction coefficient or range
        c_x (float or tuple): Linear friction coefficient or range

    Spaces:
        Action: Box([-10.0, 10.0], shape=(1,)) - force in Newtons
        Observation: Box([-inf, inf], shape=(5,)) - [sin(θ), cos(θ), θ̇, x, ẋ]

    Termination Conditions:
        - No hard termination on cart position
        - Episodes end via TimeLimit wrapper (typically 1000 steps)
        - Allows controllers to demonstrate recovery from wall hits

    Reward Function:
        Designed to encourage upright stabilization with minimal cart deviation:
        - Angle term: penalizes deviation from upright (θ=0)
        - Position term: penalizes cart deviation from center (x=0)
        - Velocity terms: penalizes excessive motion
        - Control cost: penalizes large control efforts
        - Action smoothness: penalizes rapid control changes (du_weight)
        - Soft wall: penalizes approaching rail limits (soft_wall_k)

    Example:
        >>> env = CartPendulumEnv(curriculum_phase="swingup", rk4_substeps=10)
        >>> obs, info = env.reset()
        >>> obs.shape
        (5,)
        >>> action = env.action_space.sample()
        >>> obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50
    }

    # Physical constants
    M = 1.0      # Cart mass (kg)
    m = 0.1      # Pole mass (kg)
    l = 1.0      # Pole length to center of mass (m)
    g = 9.81     # Gravity (m/s²)

    def __init__(
        self,
        curriculum_phase: str = "swingup",
        c_theta: Union[float, Tuple[float, float]] = 0.0,
        c_x: Union[float, Tuple[float, float]] = 0.0,
        rk4_substeps: int = 10,
        soft_wall_start: float = 1.8,
        soft_wall_k: float = 0.0,
        du_weight: float = 1e-3,
        seed: Optional[int] = None,
        stabilization_prob: float = 0.0,
        reward_weights: Optional[Dict[str, float]] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the Cart-Pendulum environment.

        Args:
            curriculum_phase: "swingup" or "stabilization"
                - "swingup": Episodes start from random angles (full problem)
                - "stabilization": Episodes start near upright (easier subproblem)

            c_theta: Angular friction coefficient (N·m·s)
                - float: Fixed friction for all episodes
                - tuple: (min, max) for uniform random sampling per episode

            c_x: Linear friction coefficient (N·s/m)
                - float: Fixed friction for all episodes
                - tuple: (min, max) for uniform random sampling per episode

            rk4_substeps: Number of RK4 integration steps per control step
                - Higher values = more accurate but slower
                - Recommended: 6-10 for training, 10+ for evaluation

            soft_wall_start: Distance (m) at which soft wall penalty begins
                - Default 1.8m (rail limit is 2.4m)

            soft_wall_k: Soft wall penalty coefficient
                - 0.0 = no penalty for approaching walls (evaluation mode)
                - >0.0 = smooth penalty that increases as cart approaches limit
                - Used during training to discourage wall hits

            du_weight: Action smoothness penalty weight
                - Penalizes |u_t - u_{t-1}| to encourage smooth control
                - Important for real-world actuators
                - Default: 1e-3 (tuned for smooth swing-up)

            seed: Random seed for reproducibility

            stabilization_prob: Probability of starting near upright in swingup mode
                - 0.0 = always random (full swingup)
                - 1.0 = always near upright (stabilization only)

            reward_weights: Optional dictionary to customize reward function weights
                - 'theta': Weight for angle error (default: 1.0)
                - 'theta_dot': Weight for angular velocity (default: 0.05)
                - 'x': Weight for position error (default: 0.25)
                - 'x_dot': Weight for linear velocity (default: 0.02)
                - 'u': Weight for control effort (default: 0.01)
                If not provided, uses default weights above (tuned for robust swing-up)
        """
        super().__init__()

        # Timing and integration
        self.dt = 0.02  # Control frequency: 50 Hz
        self.n_substeps = int(rk4_substeps)
        assert self.n_substeps >= 1, "Need at least 1 RK4 substep"
        self.dt_int = self.dt / self.n_substeps  # Integration timestep

        # Curriculum and randomization
        self.curriculum_phase = curriculum_phase
        self.stabilization_prob = float(stabilization_prob)

        # Friction configuration (supports domain randomization)
        self.c_theta = c_theta
        self.c_x = c_x
        self._c_theta_ep = float(c_theta) if not hasattr(c_theta, "__len__") else None
        self._c_x_ep = float(c_x) if not hasattr(c_x, "__len__") else None

        # Reward shaping parameters
        self.soft_wall_start = float(soft_wall_start)
        self.soft_wall_k = float(soft_wall_k)
        self.du_weight = float(du_weight)

        # Reward weights (tuned for robust swing-up and smooth control)
        self.reward_weights = {
            'theta': 1.0,          # Angle cost (prioritize upright)
            'theta_dot': 0.05,     # Angular velocity cost (moderate damping)
            'x': 0.25,             # Position cost (keep cart centered)
            'x_dot': 0.02,         # Linear velocity cost (cart damping)
            'u': 0.01,             # Control effort cost (penalize large forces)
        }
        # Override with user-provided weights
        if reward_weights is not None:
            self.reward_weights.update(reward_weights)

        # Gymnasium spaces
        # Action: force applied to cart
        self.action_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(1,), dtype=np.float32
        )

        # Observation: [sin(θ), cos(θ), θ̇, x, ẋ]
        # Using sin/cos instead of raw angle avoids discontinuity at ±π
        obs_limit = np.array([1.0, 1.0, 15.0, 2.4, 10.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-obs_limit, high=obs_limit, dtype=np.float32
        )

        # State variables
        self.state = None  # [theta, theta_dot, x, x_dot]
        self._last_u = 0.0  # For action smoothness penalty

        # Random number generator
        if seed is not None:
            self.seed(seed)

        # Rendering
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

    def _sample_friction_for_episode(self):
        """
        Sample episode-specific friction coefficients from configured ranges.

        Called at the start of each episode to implement domain randomization.
        If friction is specified as a tuple (min, max), samples uniformly.
        Otherwise uses the fixed value.
        """
        if hasattr(self.c_theta, "__len__"):
            self._c_theta_ep = float(
                self.np_random.uniform(self.c_theta[0], self.c_theta[1])
            )
        if hasattr(self.c_x, "__len__"):
            self._c_x_ep = float(
                self.np_random.uniform(self.c_x[0], self.c_x[1])
            )

    def _dyn(self, state: np.ndarray, u: float) -> np.ndarray:
        """
        Compute state derivatives from current state and control input.

        Implements the equations of motion derived from Lagrangian mechanics:
            (M + m)ẍ + ml(θ̈cos(θ) - θ̇²sin(θ)) = u - c_x·ẋ
            ml²θ̈ + mlẍcos(θ) = mgl·sin(θ) - c_θ·θ̇

        Solving for accelerations:
            θ̈ = [mgl·sin(θ) - c_θ·θ̇ - ml·cos(θ)·(u - c_x·ẋ + ml·θ̇²·sin(θ))/(M+m)] / [ml² - m²l²cos²(θ)/(M+m)]
            ẍ = [u - c_x·ẋ + ml(θ̇²sin(θ) - θ̈cos(θ))] / (M + m)

        Args:
            state: Current state [θ, θ̇, x, ẋ]
            u: Control force (N)

        Returns:
            State derivative [θ̇, θ̈, ẋ, ẍ]
        """
        theta, theta_dot, x, x_dot = state
        c_th = self._c_theta_ep
        c_lin = self._c_x_ep

        s = math.sin(theta)
        c = math.cos(theta)

        # Compute accelerations by solving coupled equations
        # Numerator for theta acceleration
        num_theta = (
            self.m * self.g * self.l * s
            - c_th * theta_dot
            - self.m * self.l * c * (u - c_lin * x_dot + self.m * self.l * theta_dot**2 * s) / (self.M + self.m)
        )

        # Denominator for theta acceleration
        denom_theta = self.m * self.l**2 - (self.m**2 * self.l**2 * c**2) / (self.M + self.m)

        theta_ddot = num_theta / denom_theta

        # Cart acceleration
        x_ddot = (u - c_lin * x_dot + self.m * self.l * (theta_dot**2 * s - theta_ddot * c)) / (self.M + self.m)

        return np.array([theta_dot, theta_ddot, x_dot, x_ddot], dtype=np.float64)

    def _rk4_step(self, state: np.ndarray, u: float, dt: float) -> np.ndarray:
        """
        Perform one Runge-Kutta 4th order integration step.

        RK4 is a high-accuracy numerical integration method with error O(dt^5).
        It computes four derivative estimates (k1, k2, k3, k4) and combines them
        for the state update.

        Args:
            state: Current state [θ, θ̇, x, ẋ]
            u: Control force (N), held constant during this step
            dt: Integration timestep

        Returns:
            Updated state after one RK4 step
        """
        k1 = self._dyn(state, u)
        k2 = self._dyn(state + 0.5 * dt * k1, u)
        k3 = self._dyn(state + 0.5 * dt * k2, u)
        k4 = self._dyn(state + dt * k3, u)

        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _integrate_rk4(self, state: np.ndarray, u: float) -> np.ndarray:
        """
        Integrate dynamics over one control timestep using RK4 substeps.

        Args:
            state: Current state [θ, θ̇, x, ẋ]
            u: Control force (N), held constant over all substeps

        Returns:
            State after dt seconds (n_substeps RK4 steps)
        """
        s = state.copy()
        for _ in range(self.n_substeps):
            s = self._rk4_step(s, u, self.dt_int)
        return s

    def _get_obs(self, state: np.ndarray) -> np.ndarray:
        """
        Convert state to observation.

        Uses [sin(θ), cos(θ), θ̇, x, ẋ] representation to avoid angle wrapping
        discontinuities at ±π. This is critical for neural network policies.

        Args:
            state: [θ, θ̇, x, ẋ]

        Returns:
            Observation [sin(θ), cos(θ), θ̇, x, ẋ]
        """
        theta, theta_dot, x, x_dot = state
        return np.array(
            [math.sin(theta), math.cos(theta), theta_dot, x, x_dot],
            dtype=np.float32
        )

    def _compute_reward(self, state: np.ndarray, u: float) -> float:
        """
        Compute reward for current state and action.

        Reward components (with tuned default weights):
            1. Angle cost: -1.0·θ² (encourage upright, θ=0)
            2. Angular velocity cost: -0.05·θ̇² (moderate damping, avoid overshoot)
            3. Position cost: -0.25·x² (keep cart centered)
            4. Linear velocity cost: -0.02·ẋ² (cart motion damping)
            5. Control cost: -0.01·u² (penalize large control efforts)
            6. Action smoothness: -1e-3·(u - u_prev)² (smooth, realistic control)
            7. Soft wall penalty: -soft_wall_k·overshoot² (prevent rail violations)

        These weights are tuned for robust swing-up with smooth control suitable
        for real hardware deployment.

        Args:
            state: Current state [θ, θ̇, x, ẋ]
            u: Control force (N)

        Returns:
            Scalar reward
        """
        theta, theta_dot, x, x_dot = state

        # Basic state costs (using configurable weights)
        reward = 0.0
        reward -= self.reward_weights['theta'] * theta**2  # Angle cost
        reward -= self.reward_weights['theta_dot'] * theta_dot**2  # Angular velocity cost
        reward -= self.reward_weights['x'] * x**2  # Position cost
        reward -= self.reward_weights['x_dot'] * x_dot**2  # Linear velocity cost
        reward -= self.reward_weights['u'] * u**2  # Control effort cost

        # Action smoothness penalty
        du = u - self._last_u
        reward -= self.du_weight * du**2

        # Soft wall penalty (exponential growth as cart approaches limits)
        if self.soft_wall_k > 0.0:
            if abs(x) > self.soft_wall_start:
                overshoot = abs(x) - self.soft_wall_start
                reward -= self.soft_wall_k * overshoot**2

        return float(reward)

    def _is_terminated(self, state: np.ndarray) -> bool:
        """
        Check if episode should terminate due to failure condition.

        Episodes terminate when the cart exceeds the rail limits at x = ±2.4m.
        This hard termination prevents unrealistic behavior and matches the
        consolidated study implementation.

        The soft wall penalty (when enabled during training) provides a smooth
        gradient starting at soft_wall_start (typically 1.8m) to discourage
        the agent from hitting the hard limit.

        Args:
            state: Current state [θ, θ̇, x, ẋ]

        Returns:
            True if |x| > 2.4m (cart hit wall), False otherwise
        """
        x = state[2]
        return abs(x) > 2.4

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset environment to initial state.

        Initial state depends on curriculum_phase:
            - "stabilization": Start near upright (θ ∈ [-0.2, 0.2], x ∈ [-0.3, 0.3])
            - "swingup": Start at random angle (θ ∈ [-π, π], x ∈ [-0.5, 0.5])

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)

        Returns:
            observation: Initial observation [sin(θ), cos(θ), θ̇, x, ẋ]
            info: Additional information (empty dict)
        """
        super().reset(seed=seed)

        # Sample friction for this episode
        self._sample_friction_for_episode()

        # Sample initial state based on curriculum
        if self.curriculum_phase == "stabilization":
            # Start near upright
            theta = self.np_random.uniform(-0.2, 0.2)
            x = self.np_random.uniform(-0.3, 0.3)
        else:  # swingup
            # Check if we should occasionally start near upright
            if self.np_random.random() < self.stabilization_prob:
                theta = self.np_random.uniform(-0.2, 0.2)
                x = self.np_random.uniform(-0.3, 0.3)
            else:
                # Full random angle (swing-up required)
                theta = self.np_random.uniform(-math.pi, math.pi)
                x = self.np_random.uniform(-0.5, 0.5)

        # Start with zero velocities
        theta_dot = 0.0
        x_dot = 0.0

        self.state = np.array([theta, theta_dot, x, x_dot], dtype=np.float64)
        self._last_u = 0.0

        return self._get_obs(self.state), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one control step in the environment.

        Args:
            action: Control force [u] where u ∈ [-10, 10] N

        Returns:
            observation: New observation [sin(θ), cos(θ), θ̇, x, ẋ]
            reward: Scalar reward for this transition
            terminated: True if episode ended due to failure
            truncated: True if episode ended due to time limit (handled by wrapper)
            info: Additional information dict
        """
        # Extract control input
        u = float(np.clip(action[0], -10.0, 10.0))

        # Integrate dynamics over one control timestep
        self.state = self._integrate_rk4(self.state, u)

        # Compute reward
        reward = self._compute_reward(self.state, u)

        # Check termination
        terminated = self._is_terminated(self.state)

        # Update for action smoothness calculation
        self._last_u = u

        # Get observation
        obs = self._get_obs(self.state)

        return obs, reward, terminated, False, {}

    def get_state(self) -> np.ndarray:
        """
        Get the current true state (for analysis/debugging).

        Returns:
            State [θ, θ̇, x, ẋ]
        """
        return self.state.copy()

    def set_state(self, state: np.ndarray) -> None:
        """
        Set the environment to a specific state (for testing/analysis).

        Args:
            state: [θ, θ̇, x, ẋ]
        """
        self.state = np.array(state, dtype=np.float64)
        self._last_u = 0.0

    def render(self):
        """
        Render the cart-pendulum system using pygame.

        Rendering code adapted from Gymnasium's CartPole-v1 (MIT License).
        https://github.com/Farama-Foundation/Gymnasium

        Returns:
            If render_mode is "rgb_array", returns numpy array of shape (H, W, 3)
            representing the rendered frame. Otherwise returns None.
        """
        if self.render_mode is None:
            return None

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise gym.error.DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        # Screen dimensions
        screen_width = 600
        screen_height = 400

        # World coordinates
        world_width = 5.0  # Show ±2.5m
        scale = screen_width / world_width

        # Cart dimensions (in pixels)
        cart_width = 50.0
        cart_height = 30.0

        # Pole dimensions
        pole_width = 10.0
        pole_length = scale * 2.0 * self.l  # pole goes from pivot to end (2*l total)

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
                pygame.display.set_caption("Cart-Pendulum")
            else:  # rgb_array
                self.screen = pygame.Surface((screen_width, screen_height))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Fill background
        self.screen.fill((255, 255, 255))

        # Get current state
        theta, theta_dot, x, x_dot = self.state

        # Convert world coordinates to screen coordinates
        cart_x = int(x * scale + screen_width / 2.0)  # x=0 is center
        cart_y = int(screen_height * 0.7)  # Cart height from top

        # Draw rail
        rail_y = cart_y + cart_height / 2
        rail_left = int(screen_width / 2 - 2.4 * scale)
        rail_right = int(screen_width / 2 + 2.4 * scale)
        pygame.draw.line(
            self.screen,
            (0, 0, 0),  # Black
            (rail_left, int(rail_y)),
            (rail_right, int(rail_y)),
            4
        )

        # Draw rail limits (red markers)
        limit_height = 20
        pygame.draw.line(
            self.screen,
            (200, 0, 0),  # Red
            (rail_left, int(rail_y - limit_height)),
            (rail_left, int(rail_y + limit_height)),
            3
        )
        pygame.draw.line(
            self.screen,
            (200, 0, 0),  # Red
            (rail_right, int(rail_y - limit_height)),
            (rail_right, int(rail_y + limit_height)),
            3
        )

        # Draw cart (blue rectangle)
        cart_rect = pygame.Rect(
            int(cart_x - cart_width / 2),
            int(cart_y - cart_height / 2),
            int(cart_width),
            int(cart_height)
        )
        gfxdraw.box(self.screen, cart_rect, (70, 130, 180))  # Steel blue
        pygame.draw.rect(self.screen, (0, 0, 0), cart_rect, 2)  # Black border

        # Draw pole (red line with circle at end)
        # Pole rotates from cart center, theta=0 is up, positive is CCW
        pole_pivot_x = cart_x
        pole_pivot_y = cart_y

        pole_end_x = pole_pivot_x + pole_length * math.sin(theta)
        pole_end_y = pole_pivot_y - pole_length * math.cos(theta)  # y is down in pygame

        # Draw pole line
        pygame.draw.line(
            self.screen,
            (139, 0, 0),  # Dark red
            (int(pole_pivot_x), int(pole_pivot_y)),
            (int(pole_end_x), int(pole_end_y)),
            int(pole_width)
        )

        # Draw pole mass (circle at end)
        gfxdraw.filled_circle(
            self.screen,
            int(pole_end_x),
            int(pole_end_y),
            int(pole_width * 1.5),
            (180, 0, 0)  # Brighter red
        )
        gfxdraw.aacircle(
            self.screen,
            int(pole_end_x),
            int(pole_end_y),
            int(pole_width * 1.5),
            (0, 0, 0)  # Black outline
        )

        # Draw state information
        font = pygame.font.Font(None, 24)
        angle_deg = math.degrees(theta)

        info_lines = [
            f"θ = {angle_deg:6.1f}°",
            f"θ̇ = {theta_dot:6.2f} rad/s",
            f"x = {x:6.2f} m",
            f"ẋ = {x_dot:6.2f} m/s",
        ]

        y_offset = 10
        for line in info_lines:
            text_surface = font.render(line, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 25

        # Add upright indicator
        if abs(theta) < math.radians(10):
            status_text = "UPRIGHT ✓"
            status_color = (0, 150, 0)  # Green
        else:
            status_text = "BALANCING"
            status_color = (150, 0, 0)  # Red

        status_surface = font.render(status_text, True, status_color)
        self.screen.blit(status_surface, (screen_width - 150, 10))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        """
        Clean up rendering resources.

        Closes pygame display and quits pygame if it was initialized.
        """
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
