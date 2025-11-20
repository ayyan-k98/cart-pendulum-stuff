# CRITICAL BUG: Angle Convention Mismatch

## The Problem

The codebase has **INCONSISTENT angle conventions** between different components!

### Standard RL CartPole Convention (Correct)
- θ ∈ (-π, π]  (angle wraps at ±π)
- **θ = 0** at TOP (upright, goal state)
- **θ = ±π** at BOTTOM (hanging down, wrapping point)
- This is the standard used by OpenAI Gym, most RL papers, etc.

### What the Code Actually Does

#### 1. Reward Function (environment.py:224) ✓ CORRECT
```python
reward = math.cos(theta) + 1.0
```

Analysis:
- θ=0°:   reward = cos(0) + 1 = **2.0** ← HIGH reward (upright)
- θ=90°:  reward = cos(π/2) + 1 = 1.0
- θ=180°: reward = cos(π) + 1 = **0.0** ← LOW reward (hanging)

**This uses the STANDARD convention: θ=0 is the goal!**

#### 2. Success Detection (environment.py:246) ✗ WRONG!
```python
# Success bonus: near upright (θ ≈ π) and centered (x ≈ 0)
theta_error = abs(theta - math.pi)
if theta_error < 0.2 and abs(x) < 0.2:
    reward += 10.0
```

**This checks for θ≈π as success, contradicting the reward function!**

#### 3. Classical Controller Target (classical_control.py:178-179) ✗ WRONG!
```python
θ_target_ccw = np.pi
θ_target_cw = -np.pi
```

**The classical controller plans trajectories to θ=±π, thinking that's upright!**

#### 4. Evaluation Success (evaluation.py:538) ✗ WRONG!
```python
# Success: final angle within 10 degrees of upright (θ ≈ ±π)
final_theta = df['theta'].iloc[-1]
final_angle_error = np.abs(np.abs(final_theta) - np.pi)  # Distance from ±π
success = final_angle_error < np.deg2rad(10)
```

**Checks for θ≈±π as success!**

#### 5. Comments Throughout ✗ WRONG!
```python
# CRITICAL CONVENTIONS:
#     - θ = 0 at BOTTOM (hanging down)  ← WRONG!
#     - θ = π at TOP (upright, target state)  ← WRONG!
```

These comments are **backwards** from the actual standard convention!

---

## Why This Causes "Nonsensical Results"

1. **RL Training**: The reward function `cos(θ)+1` gives maximum reward at θ=0
   - So the RL policy learned to balance at **θ=0** (correct!)

2. **Classical Controller**: Plans trajectories to **θ=±π**
   - Thinks θ=±π is upright (wrong!)
   - Swings pendulum to hanging position instead of upright!

3. **Success Detection**: Checks for **θ≈±π**
   - Classical controller "succeeds" by reaching θ=-180° (hanging down!)
   - RL controller "fails" even when at θ=0° (actually upright!)

4. **Result**: Completely backwards comparison!
   - RL reaches θ=0° (upright) → marked as "failure"
   - Classical reaches θ=-180° (hanging) → marked as "success"

---

## The Source of Confusion

The reference FFFB implementation (v6, JB May 2025) uses a **NON-STANDARD convention**:

```python
# v5, JB (Jan. 1, 2025)
#    - revert to θ=0 at bottom (more symmetric if swinging both ways)
```

This was a deliberate choice for the physics simulation, but it's **NOT** the standard RL convention!

When copying the reference code, we:
1. ✓ Correctly used standard convention in reward function (cos(θ)+1)
2. ✗ Incorrectly copied the non-standard convention to success detection
3. ✗ Incorrectly used non-standard convention in classical controller
4. ✗ Added wrong comments throughout

---

## What Needs to Be Fixed

### Priority 1: Classical Controller Target
```python
# classical_control.py lines 178-179
# CHANGE FROM:
θ_target_ccw = np.pi   # ← WRONG! This is bottom!
θ_target_cw = -np.pi   # ← WRONG! This is bottom!

# CHANGE TO:
θ_target = 0.0  # ← CORRECT! This is top (upright)
```

The classical controller should swing up to **θ=0**, not θ=±π!

### Priority 2: Success Detection in Environment
```python
# environment.py line 246
# CHANGE FROM:
theta_error = abs(theta - math.pi)  # ← Checking for bottom!

# CHANGE TO:
theta_error = abs(theta)  # ← Check for top (θ=0)
```

### Priority 3: Success Detection in Evaluation
```python
# evaluation.py line 538
# CHANGE FROM:
final_angle_error = np.abs(np.abs(final_theta) - np.pi)

# CHANGE TO:
final_angle_error = np.abs(final_theta)
```

### Priority 4: All Comments
Update all comments to reflect the **standard RL convention**:
- θ = 0 at TOP (upright, goal)
- θ = ±π at BOTTOM (hanging, wrapping point)

---

## Test Case to Verify

After fixes, test from hanging position:

**Initial state**: θ = π (or -π) = 180° = **hanging down**

**Expected behavior**:
- Classical controller should plan trajectory from θ=±π → θ=0
- Final state should be θ ≈ 0° (upright)
- Success detection should confirm θ ≈ 0°

**Current (broken) behavior**:
- Classical controller plans from θ=±π → θ=±π (no movement!)
- Or swings the wrong way
- "Succeeds" when reaching θ=-180° (still hanging!)

---

## Math Verification

With standard convention (θ=0 at top):

| Position | θ (rad) | θ (deg) | cos(θ)+1 | Physical |
|----------|---------|---------|----------|----------|
| Upright (goal) | 0 | 0° | **2.0** | Pendulum up |
| Right | π/2 | 90° | 1.0 | Horizontal right |
| Hanging | ±π | ±180° | **0.0** | Pendulum down |
| Left | -π/2 | -90° | 1.0 | Horizontal left |

The reward `cos(θ)+1` correctly gives maximum (2.0) at θ=0 and minimum (0.0) at θ=±π.

---

## Action Items

- [ ] Fix classical controller to target θ=0 (not θ=±π)
- [ ] Fix success detection in environment.py
- [ ] Fix success detection in evaluation.py
- [ ] Update all comments to standard convention
- [ ] Update IMPLEMENTATION_COMPARISON.md
- [ ] Test classical controller from θ=π → θ=0
- [ ] Verify RL vs Classical comparison makes sense
