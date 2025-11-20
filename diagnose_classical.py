#!/usr/bin/env python3
"""
Diagnostic script to understand classical controller behavior.
"""
import sys
sys.path.append('.')

import numpy as np
from src.classical_control import TrajectoryPlanner
from src.environment import CartPendulumEnv

def test_simple_case():
    """Test the simplest case: hanging down, swing to upright."""
    print("="*80)
    print("DIAGNOSTIC: Classical Controller Behavior")
    print("="*80)
    print()

    # Initial state: hanging down (θ=0), at rest, centered
    theta0_deg = 0.0  # Bottom position
    initial_state = np.array([np.deg2rad(theta0_deg), 0.0, 0.0, 0.0])

    print(f"Initial state:")
    print(f"  θ = {theta0_deg:.1f}° (0° = bottom, ±180° = upright)")
    print(f"  θ̇ = 0.0 rad/s")
    print(f"  x = 0.0 m")
    print(f"  ẋ = 0.0 m/s")
    print()

    # Create planner
    print("Creating TrajectoryPlanner with reference parameters...")
    planner = TrajectoryPlanner(
        Q=np.diag([1.0, 10.0, 1.0, 1.0]),  # Reference Q matrix
        R=np.array([[10.0]]),               # Reference R value
        umax=20.0,
        zeta=0.01,
        l=1.0,
        g=9.81
    )
    print()

    # Try planning
    print("Attempting to plan trajectory...")
    success = planner.plan_from(initial_state)

    if not success:
        print("✗ PLANNING FAILED!")
        print()
        print("This means the BVP solver could not find a trajectory.")
        print("Possible causes:")
        print("  1. Cost matrices Q, R are inappropriate")
        print("  2. Duration range [3.5, 4.0, 5.0] is inappropriate")
        print("  3. Boundary conditions are incompatible")
        print("  4. Numerical issues in BVP solver")
        return False

    print(f"✓ Planning succeeded!")
    print(f"  Trajectory duration: {planner.τ:.2f}s")
    print(f"  Target angle: {np.rad2deg(planner.xstate_end[0]):.1f}°")
    print()

    # Simulate the trajectory
    print("Simulating trajectory execution...")
    env = CartPendulumEnv(zeta=0.01, dt=0.02, umax=20.0, xmax=2.4)
    env.reset()
    env.set_state(initial_state)

    max_steps = int(10.0 / env.dt)  # 10 seconds
    states = []
    actions = []
    times = []

    for step in range(max_steps):
        t = step * env.dt
        times.append(t)

        # Get control action
        action = planner.get_action(env.state, t)
        actions.append(action)

        # Record state BEFORE stepping (for clarity)
        states.append(env.state.copy())

        # Step environment
        obs, reward, terminated, truncated, info = env.step(np.array([action]))

        if terminated:
            print(f"✗ Episode terminated at t={t:.2f}s (cart hit wall)")
            print(f"  Cart position: {env.state[2]:.2f}m (limit: ±2.4m)")
            break

    # Analyze final state
    final_state = env.state
    final_theta_rad = final_state[0]
    final_theta_deg = np.rad2deg(final_theta_rad)

    print()
    print("Final state:")
    print(f"  θ = {final_theta_deg:+.2f}° (target: ±180°)")
    print(f"  θ̇ = {final_state[1]:+.3f} rad/s")
    print(f"  x = {final_state[2]:+.3f} m")
    print(f"  ẋ = {final_state[3]:+.3f} m/s")
    print()

    # Check success (using same criterion as evaluation script)
    theta_error_rad = abs(abs(final_theta_rad) - np.pi)
    theta_error_deg = np.rad2deg(theta_error_rad)
    success = theta_error_rad < np.deg2rad(10)

    print(f"Success check:")
    print(f"  Angle error from upright: {theta_error_deg:.2f}°")
    print(f"  Success threshold: 10°")
    print(f"  Result: {'✓ SUCCESS' if success else '✗ FAILURE'}")
    print()

    # Analyze trajectory
    states_array = np.array(states)
    actions_array = np.array(actions)
    times_array = np.array(times)

    print("Trajectory statistics:")
    print(f"  Duration: {times_array[-1]:.2f}s ({len(states)} steps)")
    print(f"  Max |θ|: {np.rad2deg(np.max(np.abs(states_array[:, 0]))):.1f}°")
    print(f"  Max |x|: {np.max(np.abs(states_array[:, 2])):.3f}m")
    print(f"  Max |u|: {np.max(np.abs(actions_array)):.2f} m/s²")
    print()

    # Check if pendulum actually reached upright during trajectory
    upright_mask = np.abs(np.abs(states_array[:, 0]) - np.pi) < np.deg2rad(30)
    if np.any(upright_mask):
        first_upright_idx = np.argmax(upright_mask)
        first_upright_time = times_array[first_upright_idx]
        print(f"  First time near upright (±30°): t={first_upright_time:.2f}s")
    else:
        print(f"  ⚠ WARNING: Pendulum never got within 30° of upright!")
    print()

    # Diagnose if result seems wrong
    if success and final_state[1]**2 > 1.0:
        print("⚠ WARNING: Success detected but high angular velocity!")
        print("  This suggests the pendulum is swinging through upright, not balanced.")
        print()

    if not success and theta_error_deg < 30:
        print("⚠ NOTE: Failed but got close to upright.")
        print("  Controller might be working but needs more time or better tuning.")
        print()

    return success

if __name__ == "__main__":
    try:
        success = test_simple_case()
        print("="*80)
        if success:
            print("✓✓✓ DIAGNOSTIC COMPLETE: Controller appears to work ✓✓✓")
        else:
            print("✗✗✗ DIAGNOSTIC COMPLETE: Controller has issues ✗✗✗")
        print("="*80)
        print()
        sys.exit(0 if success else 1)
    except Exception as e:
        print()
        print("="*80)
        print(f"✗✗✗ DIAGNOSTIC FAILED WITH ERROR ✗✗✗")
        print(f"Error: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        sys.exit(2)
