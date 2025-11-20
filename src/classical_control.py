"""
Classical Control Baselines for Cart-Pendulum (Reference-Matched)

This module implements optimal control baselines matching the reference FFFB code (v6, JB May 2025).

CRITICAL CONVENTIONS:
    - θ = 0 at BOTTOM (hanging down)
    - θ = π at TOP (upright, target state)
    - Simplified dynamics: θ̈ = -(g/l)·sin(θ) - 2ζ·θ̇ - (u/l)·cos(θ), ẍ = u
    - Physical units: l=1.0m, g=9.81m/s², dt=0.02s, umax=20.0m/s²
    - Friction: IGNORED in FF planning, INCLUDED in prediction/simulation

Control Architecture:
1. Trajectory optimization via Boundary Value Problem (BVP) solver
2. Linear Quadratic Regulator (LQR) for tracking and stabilization
3. Feedforward + Feedback (FF+FB) control

Mathematical Foundation:
    Optimal control problem (frictionless for planning):
        minimize ∫[x^T Q x + u^T R u] dt
        subject to: θ̈ = -(g/l)·sin(θ) - (u/l)·cos(θ), ẍ = u

    BVP formulation uses Pontryagin's Maximum Principle:
        - State equations: ẋ = f(x, u)
        - Costate equations: λ̇ = -∂H/∂x
        - Optimality: ∂H/∂u = 0 => u = -λ_ẋ + (λ_θ̇/l)·cos(θ)

    LQR formulation:
        - Riccati equation: Ṡ = -A^T S - S A - Q + S B R^{-1} B^T S
        - Feedback gain: K(t) = R^{-1} B^T S(t)
"""

from typing import Optional, Tuple
import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
from scipy.linalg import solve_continuous_are


class TrajectoryPlanner:
    """
    Optimal trajectory planner using BVP and time-varying LQR.

    MATCHES REFERENCE IMPLEMENTATION (v6, JB May 2025):
        - θ=0 at bottom, θ=π at top
        - Simplified dynamics (m << M)
        - Friction ignored in FF, included in prediction
        - Direction selection (CW vs CCW)
        - State prediction for planning delay

    Attributes:
        Q (np.ndarray): State cost matrix (4x4)
        R (np.ndarray): Control cost matrix (1x1)
        umax (float): Maximum control acceleration (dimensionless)
        zeta (float): Friction coefficient for prediction (dimensionless)
        prediction_time (float): Planning delay compensation
        plan (bool): Whether a valid plan exists
        FFsol: Feedforward trajectory solution
        Ssol: Riccati solution for time-varying gains
        Kend: Terminal LQR gain for stabilization
    """

    def __init__(
        self,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        umax: float = 20.0,
        zeta: float = 0.01,
        prediction_time: float = 0.0,
        l: float = 1.0,
        g: float = 9.81
    ):
        """
        Initialize the trajectory planner.

        Args:
            Q: State cost matrix (4x4). Default: diag([1, 10, 1, 1]) (reference v6)
            R: Control cost matrix (1x1). Default: [[10.0]] (reference v6)
            umax: Maximum control acceleration (m/s²), default 20.0
            zeta: Angular friction coefficient (dimensionless, for prediction only)
                - Reference uses ζ=0.01 for simulations
                - Used ONLY for state prediction, NOT in FF planning
            prediction_time: Planning delay compensation (seconds)
            l: Pendulum length (m), default 1.0
            g: Gravity (m/s²), default 9.81
        """
        if Q is None:
            Q = np.diag([1.0, 10.0, 1.0, 1.0])  # Reference v6 values
        if R is None:
            R = np.array([[10.0]])  # Reference v6 value

        self.Q = Q
        self.R = R
        self.umax = float(umax)
        self.zeta = float(zeta)
        self.prediction_time = float(prediction_time)
        self.l = float(l)
        self.g = float(g)
        self.plan = None
        self.FFsol = None
        self.Ssol = None
        self.Kend = None
        self.τ = None  # Trajectory duration
        self.xstate_init = None
        self.xstate_end = None

    def _predict_state(self, s: np.ndarray, dt: float) -> np.ndarray:
        """
        Predict future state with zero control (includes friction).

        Compensates for planning computation time by forward-simulating
        with u=0 and friction.

        Args:
            s: Current state [θ, θ̇, x, ẋ]
            dt: Prediction time horizon (seconds)

        Returns:
            Predicted state after dt
        """
        if dt <= 0.0:
            return s

        def dynamics(t, state):
            """Simplified dynamics with friction, zero control."""
            θ, θdot, x, xdot = state
            return np.array([
                θdot,
                -(self.g/self.l) * np.sin(θ) - 2.0 * self.zeta * θdot,  # Friction in prediction
                xdot,
                0.0  # u=0
            ])

        result = solve_ivp(dynamics, (0, dt), s, dense_output=True)
        return result.sol(dt)

    def plan_from(self, s: np.ndarray) -> bool:
        """
        Plan optimal trajectory from initial state to upright (θ=π).

        Features matching reference:
        1. State prediction: Compensates for planning delay
        2. Direction selection: Tries CW and CCW, picks lower cost
        3. Multiple durations: Tries [3.5, 4.0, 5.0] until success

        Args:
            s: Initial state [θ, θ̇, x, ẋ] (θ=0 is bottom)

        Returns:
            True if planning succeeded, False otherwise
        """
        # Predict future state to compensate for planning delay
        s_pred = self._predict_state(s, self.prediction_time)

        # Determine target angles for CW and CCW swingup
        # Convention: θ=0 at bottom, θ=π at top (upright)
        θ0 = s_pred[0]

        # Wrap θ0 to (-π, π]
        θ0_wrapped = ((θ0 + np.pi) % (2 * np.pi)) - np.pi

        # CCW swingup: go to nearest upright position counter-clockwise
        # θ=π is upright, so we go to nearest (2k+1)π
        if θ0_wrapped <= 0:
            θ_target_ccw = np.pi  # Go up to π
            θ_target_cw = -np.pi  # Go down to -π (same as π, but through bottom)
        else:
            θ_target_ccw = np.pi  # Stay at π
            θ_target_cw = -np.pi  # Wrap around

        # Actually, let's be more systematic: find the two nearest uprights
        # Upright positions are at θ = (2k+1)π for integer k
        # The two nearest are at ..., -π, π, 3π, ...
        k_nearest = np.round((θ0_wrapped - np.pi) / (2 * np.pi))
        θ_target_1 = (2 * k_nearest + 1) * np.pi  # One candidate
        θ_target_2 = (2 * (k_nearest + 1) + 1) * np.pi  # Other candidate

        # Simplify: just use ±π as the two targets
        θ_target_ccw = np.pi
        θ_target_cw = -np.pi

        best_cost = float('inf')
        best_plan = None

        # Compute natural pendulum period (matching reference approach)
        # Reference uses τ = 2π in dimensionless time
        # Physical time: T = 2π / √(g/l) ≈ 2.0 seconds
        T_period = 2 * np.pi / np.sqrt(self.g / self.l)

        # Try both directions with multiple durations (reference uses 1 period)
        # We try: 1, 1.5, and 2 periods for robustness
        for θ_target in [θ_target_cw, θ_target_ccw]:
            for duration in [T_period, 1.5*T_period, 2.0*T_period]:
                cost = self._plan_maneuver(s_pred, θ_target, duration)
                if cost is not None and cost < best_cost:
                    best_cost = cost
                    best_plan = {
                        'θ_target': θ_target,
                        'duration': duration,
                        'FFsol': self.FFsol,
                        'Ssol': self.Ssol,
                        'Kend': self.Kend,
                        'τ': self.τ
                    }

        if best_plan is not None:
            # Restore best plan
            self.FFsol = best_plan['FFsol']
            self.Ssol = best_plan['Ssol']
            self.Kend = best_plan['Kend']
            self.τ = best_plan['τ']
            self.xstate_init = s_pred
            self.plan = True
            return True

        self.plan = False
        return False

    def _plan_maneuver(self, s: np.ndarray, θ_target: float, duration: float) -> Optional[float]:
        """
        Plan maneuver with specific duration and target (MATCHES REFERENCE).

        Uses Pontryagin's Maximum Principle with SIMPLIFIED DYNAMICS (frictionless):
            θ̈ = -sin(θ) - u·cos(θ)
            ẍ = u

        Args:
            s: Initial state [θ, θ̇, x, ẋ]
            θ_target: Target angle (π or -π for upright)
            duration: Trajectory duration (dimensionless)

        Returns:
            Trajectory cost if successful, None if BVP fails
        """
        self.τ = float(duration)
        self.xstate_init = s
        self.xstate_end = np.array([θ_target, 0.0, 0.0, 0.0])

        # BVP function: augmented dynamics [state, costate]
        def bvpfcn(t, X):
            """
            Augmented dynamics (FRICTIONLESS, matching reference).

            State: X = [θ, θ̇, x, ẋ, λ_θ, λ_θ̇, λ_x, λ_ẋ]

            Simplified dynamics (physical units):
                θ̈ = -(g/l)·sin(θ) - (u/l)·cos(θ)
                ẍ = u

            Control from optimality:
                u* = -λ_ẋ + (λ_θ̇/l)·cos(θ)

            CRITICAL: scipy's solve_bvp can pass X as 1D or 2D!
                - 1D: X.shape = (8,) for single evaluation point
                - 2D: X.shape = (8, n) for n evaluation points
            Reference implementation handles both cases explicitly.
            """
            # Handle both 1D and 2D cases (CRITICAL FIX matching reference)
            if X.ndim == 1:
                X_reshaped = X.reshape(-1, 1)
            else:
                X_reshaped = X

            θ, θdot, x, xdot, λθ, λθdot, λx, λxdot = X_reshaped

            # Optimal control (from ∂H/∂u = 0)
            # For the system with dynamics θ̈ = -(g/l)sin(θ) - (u/l)cos(θ), ẍ = u
            # Hamiltonian derivatives give: u* = -λ_ẋ + (λ_θ̇/l)·cos(θ)
            u = -λxdot + (λθdot / self.l) * np.cos(θ)

            # State dynamics (SIMPLIFIED, FRICTIONLESS)
            dX = np.zeros((8, X_reshaped.shape[1]))
            dX[0] = θdot
            dX[1] = -(self.g / self.l) * np.sin(θ) - (u / self.l) * np.cos(θ)  # No friction
            dX[2] = xdot
            dX[3] = u

            # Costate dynamics: λ̇ = -∂H/∂x
            # For θ̈ = -(g/l)sin(θ) - (u/l)cos(θ):
            # ∂H/∂θ = -λ_θ̇·[-(g/l)cos(θ) + (u/l)sin(θ)]
            # ∂H/∂θ̇ = -λ_θ (no friction term)
            # ∂H/∂x = 0
            # ∂H/∂ẋ = -λ_x
            dX[4] = λθdot * ((self.g / self.l) * np.cos(θ) - (u / self.l) * np.sin(θ))
            dX[5] = -λθ  # No friction damping term
            dX[6] = np.zeros_like(λx)
            dX[7] = -λx

            return dX

        # Boundary conditions
        def bc(x0, xτ):
            """BC: x(0) = s, x(τ) = target"""
            return np.hstack((x0[:4] - self.xstate_init, xτ[:4] - self.xstate_end))

        # Initial guess
        tvec = np.linspace(0.0, self.τ, 25)
        xλguess = np.zeros((8, tvec.size))
        xλguess[0, :] = np.linspace(self.xstate_init[0], self.xstate_end[0], tvec.size)
        xλguess[2, :] = np.linspace(self.xstate_init[2], self.xstate_end[2], tvec.size)

        # Solve BVP
        res = solve_bvp(bvpfcn, bc, tvec, xλguess, tol=1e-5, max_nodes=3000)
        if not res.success:
            return None

        self.FFsol = res.sol

        # Compute trajectory cost
        n_samples = 100
        t_samples = np.linspace(0, self.τ, n_samples)
        u_samples = np.array([self._uff(t) for t in t_samples])
        trajectory_cost = 0.5 * np.sum(u_samples**2) * (self.τ / n_samples)

        # Compute time-varying LQR gains (Riccati equation)
        def A(t):
            """State matrix from linearization (FRICTIONLESS)."""
            θt = self.FFsol(t)[0]
            _, _, _, _, _, λθdot, _, λxdot = self.FFsol(t)
            u_ff = -λxdot + (λθdot / self.l) * np.cos(θt)

            return np.array([
                [0, 1, 0, 0],
                [-(self.g / self.l) * np.cos(θt) + (u_ff / self.l) * np.sin(θt), 0, 0, 0],  # No friction
                [0, 0, 0, 1],
                [0, 0, 0, 0]
            ])

        def B(t):
            """Control matrix from linearization."""
            θt = self.FFsol(t)[0]
            return np.array([[0.0], [-(1.0 / self.l) * np.cos(θt)], [0.0], [1.0]])

        def Riccati(t, Svec):
            """Riccati ODE (backwards integration)."""
            S = Svec.reshape((4, 4))
            At = A(t)
            Bt = B(t)
            dS = -At.T @ S - S @ At - self.Q + S @ (Bt @ Bt.T) @ S / self.R[0, 0]
            return dS.flatten()

        # Terminal LQR gain (steady-state at upright)
        Aend = A(self.τ)
        Bend = B(self.τ)
        Send = solve_continuous_are(Aend, Bend, self.Q, self.R)
        self.Kend = Bend.T @ Send / self.R[0, 0]

        # Integrate Riccati backwards
        self.Ssol = solve_ivp(
            Riccati,
            (self.τ, 0.0),
            Send.flatten(),
            dense_output=True,
            rtol=1e-6,
            atol=1e-8
        ).sol

        return trajectory_cost

    def _uff(self, t: float) -> float:
        """
        Feedforward control at time t.

        Args:
            t: Time (seconds)

        Returns:
            Feedforward control acceleration (m/s²)
        """
        θ, _, _, _, _, λθdot, _, λxdot = self.FFsol(t)
        return -λxdot + (λθdot / self.l) * np.cos(θ)

    def _K(self, t: float) -> np.ndarray:
        """
        Time-varying feedback gain K(t) = R^{-1} B^T S(t).

        Args:
            t: Time (seconds)

        Returns:
            Feedback gain vector (1x4)
        """
        S = self.Ssol(t).reshape((4, 4))
        θt = self.FFsol(t)[0]
        Bt = np.array([[0.0], [-(1.0 / self.l) * np.cos(θt)], [0.0], [1.0]])
        return (Bt.T @ S / self.R[0, 0]).ravel()

    def get_action(self, s: np.ndarray, t: float) -> float:
        """
        Compute control action (FF+FB architecture).

        Control law:
            - During trajectory (t < τ): u = u_ff(t) - K(t)·[s - s*(t)]
            - After trajectory (t >= τ): u = -K_end·[s - s_target]

        Args:
            s: Current state [θ, θ̇, x, ẋ] (θ=0 is bottom, θ=π is top)
            t: Time since start of maneuver (seconds)

        Returns:
            Control acceleration (m/s²), clipped to [-umax, umax]
        """
        if self.plan is None or not self.plan:
            # No valid plan, return zero control
            return 0.0

        if t < self.τ:
            # During trajectory: FF + FB tracking
            ref = self.FFsol(t)[:4]
            dev = s - ref
            u = self._uff(t) - self._K(t) @ dev
        else:
            # After trajectory: LQR stabilization around upright
            # Target: θ=π (or -π), x=0
            target = np.array([np.pi, 0.0, 0.0, 0.0])

            # Handle angle wrapping: compute deviation in wrapped space
            dev = s - target
            dev[0] = ((dev[0] + np.pi) % (2 * np.pi)) - np.pi  # Wrap to (-π, π]

            u = -self.Kend @ dev

        # Ensure scalar and clip
        u_scalar = float(np.asarray(u).squeeze())
        return float(np.clip(u_scalar, -self.umax, self.umax))

    def reset(self):
        """Reset the planner (clear current plan)."""
        self.plan = None
        self.FFsol = None
        self.Ssol = None
        self.Kend = None
        self.τ = None
        self.xstate_init = None
        self.xstate_end = None


def compute_lqr_gain(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray
) -> np.ndarray:
    """
    Compute steady-state LQR gain for continuous-time system.

    Solves: A^T S + S A - S B R^{-1} B^T S + Q = 0
    Returns: K = R^{-1} B^T S

    Args:
        A: State matrix (n×n)
        B: Control matrix (n×m)
        Q: State cost matrix (n×n), positive semi-definite
        R: Control cost matrix (m×m), positive definite

    Returns:
        Optimal feedback gain K (m×n)

    Example:
        >>> # Linearized cart-pendulum around upright (θ=π)
        >>> A = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        >>> B = np.array([[0], [1], [0], [1]])  # Note: simplified dynamics
        >>> Q = np.diag([10, 1, 10, 1])
        >>> R = np.array([[1]])
        >>> K = compute_lqr_gain(A, B, Q, R)
    """
    S = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ S)
    return K
