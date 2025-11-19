"""
Classical Control Baselines for Cart-Pendulum

This module implements optimal control baselines using:
1. Trajectory optimization via Boundary Value Problem (BVP) solver
2. Linear Quadratic Regulator (LQR) for tracking and stabilization
3. Feedforward + Feedback (FF+FB) control architecture

The approach:
- For swing-up: Solve a BVP to find an optimal trajectory from initial state to upright
- For tracking: Use time-varying LQR to follow the trajectory with feedback
- For stabilization: Use steady-state LQR around upright equilibrium

Mathematical Foundation:
    The system dynamics in non-dimensional form:
        θ̈ = -sin(θ) - u·cos(θ)  [normalized pendulum equation]
        ẍ = u                     [normalized cart equation]

    Optimal control problem:
        minimize ∫[x^T Q x + u^T R u] dt
        subject to dynamics

    BVP formulation uses Pontryagin's Maximum Principle:
        - State equations: ẋ = f(x, u)
        - Costate equations: λ̇ = -∂H/∂x
        - Optimality: ∂H/∂u = 0 => u = R^{-1} B^T λ

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

    This class implements a two-phase control strategy:
    1. Feedforward (FF): Optimal open-loop trajectory from BVP solver
    2. Feedback (FB): Time-varying LQR for disturbance rejection

    The planner attempts multiple trajectory durations to find a feasible solution,
    then computes the Riccati solution backwards in time for optimal tracking gains.

    Attributes:
        Q (np.ndarray): State cost matrix (4x4), penalizes state deviations
        R (np.ndarray): Control cost matrix (1x1), penalizes control effort
        umax (float): Maximum control force (N)
        plan (bool): Whether a valid plan exists
        FFsol: Feedforward trajectory solution (scipy BVP solution object)
        Ssol: Riccati solution for time-varying gains
        Kend: Terminal LQR gain for stabilization after trajectory

    Example:
        >>> planner = TrajectoryPlanner(umax=10.0)
        >>> initial_state = np.array([np.pi, 0.0, 0.0, 0.0])  # Hanging down
        >>> success = planner.plan_from(initial_state)
        >>> if success:
        ...     for t in np.arange(0, 5, 0.02):
        ...         action = planner.get_action(current_state, t)
    """

    def __init__(
        self,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        umax: float = 10.0,
        c_theta: float = 0.0,
        prediction_time: float = 0.0
    ):
        """
        Initialize the trajectory planner.

        Args:
            Q: State cost matrix (4x4). Default: diag([10, 4, 10, 2])
                - Penalizes [θ, θ̇, x, ẋ] deviations
                - Higher values = tighter tracking of that state component

            R: Control cost matrix (1x1). Default: [[2.0]]
                - Penalizes control effort
                - Higher values = smoother, less aggressive control

            umax: Maximum absolute control force (N)
                - Actions are clipped to [-umax, umax]

            c_theta: Angular friction coefficient (N·m·s)
                - Models pendulum damping: τ_friction = -c_theta * θ̇
                - Should match environment friction for accurate tracking
                - Default: 0.0 (no friction)

            prediction_time: Planning delay compensation (seconds)
                - Predicts future state after this time delay
                - Compensates for BVP solver computation time
                - Default: 0.0 (no prediction)
        """
        if Q is None:
            Q = np.diag([10.0, 4.0, 10.0, 2.0])
        if R is None:
            R = np.array([[2.0]])

        self.Q = Q
        self.R = R
        self.umax = float(umax)
        self.c_theta = float(c_theta)
        self.prediction_time = float(prediction_time)
        self.plan = None
        self.FFsol = None
        self.Ssol = None
        self.Kend = None
        self.τ = None  # Trajectory duration
        self.plan_start_time = 0.0  # Time when planning started

    def _predict_state(self, s: np.ndarray, dt: float) -> np.ndarray:
        """
        Predict future state after time dt with zero control.

        Compensates for planning computation time by forward-simulating
        the system dynamics with u=0.

        Args:
            s: Current state [θ, θ̇, x, ẋ]
            dt: Prediction time horizon (seconds)

        Returns:
            Predicted state after dt seconds
        """
        if dt <= 0.0:
            return s

        def dynamics(t, state):
            """Dynamics with zero control and friction."""
            θ, θdot, x, xdot = state
            return np.array([
                θdot,
                -np.sin(θ) - 2.0 * self.c_theta * θdot,  # Pendulum with friction
                xdot,
                0.0  # Cart: no control, no friction
            ])

        result = solve_ivp(dynamics, (0, dt), s, dense_output=True)
        return result.sol(dt)

    def plan_from(self, s: np.ndarray) -> bool:
        """
        Plan an optimal trajectory from given initial state to upright.

        Features:
        1. State prediction: Compensates for planning time by predicting future state
        2. Direction selection: Tries both CW and CCW swingup, picks lower cost
        3. Multiple durations: Tries [3.5, 4.0, 5.0] seconds until one succeeds

        For each configuration, solves the two-point BVP:
            - Start: s_pred (predicted initial state)
            - End: target angle (either CW or CCW swingup to upright)

        Args:
            s: Initial state [θ, θ̇, x, ẋ]

        Returns:
            True if planning succeeded, False otherwise
        """
        # Predict future state to compensate for planning delay
        s_pred = self._predict_state(s, self.prediction_time)

        # Determine target angles for CW and CCW swingup
        # Convention: θ=0 is upright, so we swing to nearest multiple of 2π
        θ0 = s_pred[0]

        # CCW swingup: go to nearest upright position counter-clockwise
        θ_target_ccw = 2.0 * np.pi * np.ceil(θ0 / (2.0 * np.pi) + 0.5) - np.pi
        # Simplify: round to nearest 2π, then subtract π (since we measure from top)
        # Actually for θ=0 at top: CCW means increasing θ to next 2πk
        θ_target_ccw = 2.0 * np.pi * np.ceil(θ0 / (2.0 * np.pi))

        # CW swingup: go to nearest upright position clockwise
        θ_target_cw = θ_target_ccw - 2.0 * np.pi

        best_cost = float('inf')
        best_plan = None

        # Try both directions with multiple durations
        for θ_target in [θ_target_cw, θ_target_ccw]:
            for duration in [3.5, 4.0, 5.0]:
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
        Internal method to plan a maneuver with specific duration and target angle.

        Uses Pontryagin's Maximum Principle to formulate as BVP:
            - State equations: ẋ = f(x, u)
            - Costate equations: λ̇ = -∂H/∂x where H = L + λ^T f
            - Optimality: u* = R^{-1} B^T λ

        After finding the open-loop trajectory, computes time-varying LQR
        gains by integrating the Riccati equation backwards in time.

        Args:
            s: Initial state [θ, θ̇, x, ẋ]
            θ_target: Target angle (for CW vs CCW swingup)
            duration: Trajectory duration (seconds)

        Returns:
            Trajectory cost (∫u² dt) if successful, None if BVP fails
        """
        self.τ = float(duration)
        self.xstate_init = s
        self.xstate_end = np.array([θ_target, 0.0, 0.0, 0.0])  # Target: specified angle, centered

        # Define the augmented dynamics: [state, costate]
        def bvpfcn(t, X):
            """
            Augmented dynamics for BVP solver with friction.

            State: X = [θ, θ̇, x, ẋ, λ_θ, λ_{θ̇}, λ_x, λ_{ẋ}]

            From Hamiltonian H = L + λ^T f:
                State eqns: ẋ = ∂H/∂λ = f(x, u*)
                Costate eqns: λ̇ = -∂H/∂x
                Control: u* = -λ_{ẋ} + λ_{θ̇}·cos(θ)  [from ∂H/∂u = 0]
            """
            θ, θdot, x, xdot, λθ, λθdot, λx, λxdot = X

            # Optimal control from costate
            u = -λxdot + λθdot * np.cos(θ)

            # State dynamics (normalized, with friction)
            dX = np.zeros_like(X)
            dX[0, :] = θdot
            dX[1, :] = -np.sin(θ) - 2.0 * self.c_theta * θdot - u * np.cos(θ)  # Added friction
            dX[2, :] = xdot
            dX[3, :] = u

            # Costate dynamics: λ̇ = -∂H/∂x
            # With friction: ∂f_θdot/∂θdot = -2*c_theta
            # ∂H/∂θ = -λ_{θ̇}·(cos(θ) - u·sin(θ))
            # ∂H/∂{θ̇} = -λ_θ - λ_{θ̇}·(-2*c_theta) = -λ_θ + 2*c_theta*λ_{θ̇}
            # ∂H/∂x = 0 (no direct x dependence in dynamics)
            # ∂H/∂{ẋ} = -λ_x
            costate = np.vstack([
                λθdot * (np.cos(θ) - u * np.sin(θ)),  # dλ_θ/dt
                -λθ + 2.0 * self.c_theta * λθdot,      # dλ_{θ̇}/dt (with friction term)
                np.zeros_like(λx),                       # dλ_x/dt
                -λx                                      # dλ_{ẋ}/dt
            ])
            dX[4:, :] = costate

            return dX

        # Boundary conditions: match initial and final states
        def bc(x0, xτ):
            """Boundary conditions: x(0) = s, x(τ) = 0"""
            return np.hstack((x0[:4] - self.xstate_init, xτ[:4] - self.xstate_end))

        # Initial guess: linear interpolation for states, zero costates
        tvec = np.linspace(0.0, self.τ, 25)
        xλguess = np.zeros((8, tvec.size))
        xλguess[0, :] = np.linspace(self.xstate_init[0], self.xstate_end[0], tvec.size)
        xλguess[2, :] = np.linspace(self.xstate_init[2], self.xstate_end[2], tvec.size)

        # Solve BVP
        res = solve_bvp(bvpfcn, bc, tvec, xλguess, tol=1e-5, max_nodes=3000)
        if not res.success:
            return None

        self.FFsol = res.sol

        # Compute trajectory cost for direction selection
        # Cost = ∫ u² dt (integrated control effort)
        n_samples = 100
        t_samples = np.linspace(0, self.τ, n_samples)
        u_samples = np.array([self._uff(t) for t in t_samples])
        trajectory_cost = 0.5 * np.sum(u_samples**2) * (self.τ / n_samples)

        # Now compute time-varying LQR gains via Riccati equation
        # Linearize around the optimal trajectory
        def A(t):
            """
            State matrix A(t) from linearization around trajectory with friction.

            ∂f/∂x evaluated at x*(t), u*(t)
            """
            θt = self.FFsol(t)[0]
            _, _, _, _, _, λθdot, _, λxdot = self.FFsol(t)
            u_ff = -λxdot + λθdot * np.cos(θt)

            return np.array([
                [0, 1, 0, 0],
                [-np.cos(θt) + u_ff * np.sin(θt), -2.0 * self.c_theta, 0, 0],  # Added friction term
                [0, 0, 0, 1],
                [0, 0, 0, 0]
            ])

        def B(t):
            """Control matrix B(t) from linearization."""
            θt = self.FFsol(t)[0]
            return np.array([[0.0], [-np.cos(θt)], [0.0], [1.0]])

        def Riccati(t, Svec):
            """
            Riccati ODE: Ṡ = -A^T S - S A - Q + S B R^{-1} B^T S

            Integrated backwards from terminal condition.
            """
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

        # Integrate Riccati equation backwards
        self.Ssol = solve_ivp(
            Riccati,
            (self.τ, 0.0),  # Backwards integration
            Send.flatten(),
            dense_output=True,
            rtol=1e-6,
            atol=1e-8
        ).sol

        return trajectory_cost  # Return cost for direction selection

    def _uff(self, t: float) -> float:
        """
        Feedforward control at time t from optimal trajectory.

        Args:
            t: Time (seconds)

        Returns:
            Feedforward control force (N)
        """
        θ, _, _, _, _, λθdot, _, λxdot = self.FFsol(t)
        return -λxdot + λθdot * np.cos(θ)

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
        Bt = np.array([[0.0], [-np.cos(θt)], [0.0], [1.0]])
        return (Bt.T @ S / self.R[0, 0]).ravel()

    def get_action(self, s: np.ndarray, t: float) -> float:
        """
        Compute control action for current state and time.

        Control law:
            - During trajectory (t < τ):
                u(t) = u_ff(t) - K(t)·[s - s*(t)]
                (feedforward + feedback on deviation from plan)

            - After trajectory (t >= τ):
                u(t) = -K_end·s
                (LQR stabilization around upright)

        Args:
            s: Current state [θ, θ̇, x, ẋ]
            t: Time since start of maneuver (seconds)

        Returns:
            Control force (N), clipped to [-umax, umax]
        """
        # Plan if not already done
        if self.plan is None:
            self.plan = self.plan_from(s)

        if not self.plan:
            # Planning failed, return zero control
            return 0.0

        if t < self.τ:
            # During trajectory: FF + FB tracking
            ref = self.FFsol(t)[:4]  # Reference state
            dev = s - ref  # Deviation from plan
            u = self._uff(t) - self._K(t) @ dev
        else:
            # After trajectory: LQR stabilization
            dev = s - np.zeros(4)
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


def compute_lqr_gain(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray
) -> np.ndarray:
    """
    Compute steady-state LQR gain for continuous-time system.

    Solves the continuous-time algebraic Riccati equation:
        A^T S + S A - S B R^{-1} B^T S + Q = 0

    Then computes the optimal feedback gain:
        K = R^{-1} B^T S

    Args:
        A: State matrix (n×n)
        B: Control matrix (n×m)
        Q: State cost matrix (n×n), must be positive semi-definite
        R: Control cost matrix (m×m), must be positive definite

    Returns:
        Optimal feedback gain K (m×n) such that u = -K·x minimizes cost

    Example:
        >>> # Linearized cart-pendulum around upright
        >>> A = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        >>> B = np.array([[0], [-1], [0], [1]])
        >>> Q = np.diag([10, 1, 10, 1])
        >>> R = np.array([[1]])
        >>> K = compute_lqr_gain(A, B, Q, R)
    """
    S = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ S)
    return K
