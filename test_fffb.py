#!/usr/bin/env python3
"""
Quick test to verify FFFB algorithm works with new parameters.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.abspath('.'))

from src import CartPendulumEnv, TrajectoryPlanner

def test_fffb_simple():
    """Test FFFB on simple initial condition: hanging down."""
    print("="*80)
    print("Testing FFFB Algorithm")
    print("="*80)

    # Create environment with new parameters
    env = CartPendulumEnv(
        curriculum_phase="swingup",
        zeta=0.01,
        dt=0.02,
        xmax=2.4,
        umax=20.0
    )

    # Create FFFB planner with matching parameters
    planner = TrajectoryPlanner(
        umax=20.0,  # Match environment
        zeta=0.01,
        l=1.0,
        g=9.81
    )

    # Test case 1: Hanging down (θ=0)
    print("\n### Test 1: Hanging down (θ=0)")
    initial_state = np.array([0.0, 0.0, 0.0, 0.0])  # θ=0 is bottom

    print(f"Initial state: θ={np.rad2deg(initial_state[0]):.1f}°, x={initial_state[2]:.2f}m")
    print("Planning trajectory...")

    success = planner.plan_from(initial_state)
    if success:
        print(f"✓ Planning succeeded! Duration: {planner.τ:.2f}s")
    else:
        print("✗ Planning FAILED!")
        return False

    # Simulate
    print("Simulating...")
    env.set_state(initial_state)

    max_steps = int(20.0 / env.dt)  # 20 seconds
    states = []
    actions = []

    for step in range(max_steps):
        t = step * env.dt

        # Get FFFB action
        action = planner.get_action(env.state, t)
        actions.append(action)
        states.append(env.state.copy())

        # Step environment
        obs, reward, terminated, truncated, info = env.step(np.array([action]))

        if terminated:
            print(f"✗ Episode terminated at step {step} (t={t:.2f}s)")
            print(f"   Final state: θ={np.rad2deg(env.state[0]):.1f}°, x={env.state[2]:.2f}m")
            return False

        # Check success
        theta_error = abs(env.state[0] - np.pi)
        if theta_error < 0.1 and abs(env.state[2]) < 0.2:
            print(f"✓ Success at step {step} (t={t:.2f}s)")
            print(f"   Final state: θ={np.rad2deg(env.state[0]):.1f}°, x={env.state[2]:.2f}m")
            print(f"   Max |x|: {max(abs(s[2]) for s in states):.3f}m (limit: 2.4m)")
            print(f"   Max |u|: {max(abs(a) for a in actions):.2f} m/s² (limit: 20.0)")
            return True

    print(f"✗ Did not reach upright within {max_steps} steps")
    print(f"   Final state: θ={np.rad2deg(env.state[0]):.1f}°, x={env.state[2]:.2f}m")
    return False


if __name__ == "__main__":
    success = test_fffb_simple()
    print("\n" + "="*80)
    if success:
        print("✓✓✓ FFFB TEST PASSED ✓✓✓")
    else:
        print("✗✗✗ FFFB TEST FAILED ✗✗✗")
    print("="*80 + "\n")

    sys.exit(0 if success else 1)
