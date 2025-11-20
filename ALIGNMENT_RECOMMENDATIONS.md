# Alignment Recommendations: Matching Reference FFFB Implementation

## Quick Reference: Key Differences

| Parameter | Reference (v6) | Current Default | Recommended Change |
|-----------|----------------|-----------------|-------------------|
| Q[0,0] (θ cost) | 1 | 10.0 | **Change to 1.0** |
| Q[1,1] (θ̇ cost) | 10 | 4.0 | **Change to 10.0** |
| Q[2,2] (x cost) | 1 | 10.0 | **Change to 1.0** |
| Q[3,3] (ẋ cost) | 1 | 2.0 | **Change to 1.0** |
| R (control cost) | 10 | 2.0 | **Change to 10.0** |
| Trajectory duration | 2π ≈ 2.0s | 3.5-5.0s | **Try shorter durations first** |

## Specific Code Changes Required

### 1. Update Cost Matrices in `classical_control.py`

**File**: `src/classical_control.py`
**Lines**: 86-89

**Current code:**
```python
if Q is None:
    Q = np.diag([10.0, 4.0, 10.0, 2.0])
if R is None:
    R = np.array([[2.0]])
```

**Recommended change to match reference:**
```python
if Q is None:
    Q = np.diag([1.0, 10.0, 1.0, 1.0])  # Match reference v6
if R is None:
    R = np.array([[10.0]])  # Match reference v6
```

**Explanation**:
- Reference emphasizes **angular velocity** (θ̇) more than position/angle
- This promotes **aggressive swingup** with less concern for cart position during swing
- Higher R (10 vs 2) penalizes control effort more → smoother trajectories

### 2. Update Trajectory Durations in `classical_control.py`

**File**: `src/classical_control.py`
**Line**: 186

**Current code:**
```python
for duration in [3.5, 4.0, 5.0]:
```

**Recommended change to match reference:**
```python
# Compute natural pendulum period (dimensionless time in reference: 2π)
T_period = 2 * np.pi / np.sqrt(self.g / self.l)  # ≈ 2.006 seconds

# Try: 1 period, 1.5 periods, 2 periods (matching reference approach)
for duration in [T_period, 1.5 * T_period, 2.0 * T_period]:
    # This gives approximately: [2.0s, 3.0s, 4.0s]
```

**Explanation**:
- Reference uses **1 pendulum period** (τ = 2π in dimensionless time)
- Physical equivalent: ~2.0 seconds for l=1m, g=9.81
- Shorter trajectories → more aggressive swingup, higher control cost
- Better matches reference's fast swingup strategy

### 3. Alternative: Make It Configurable

If you want flexibility to experiment with both tunings:

```python
def __init__(
    self,
    Q: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None,
    umax: float = 20.0,
    zeta: float = 0.01,
    prediction_time: float = 0.0,
    l: float = 1.0,
    g: float = 9.81,
    use_reference_tuning: bool = True,  # NEW PARAMETER
):
    """
    Initialize the trajectory planner.

    Args:
        ...
        use_reference_tuning: If True, use reference v6 cost matrices.
                             If False, use original tuning.
    """
    if Q is None:
        if use_reference_tuning:
            Q = np.diag([1.0, 10.0, 1.0, 1.0])  # Reference v6
        else:
            Q = np.diag([10.0, 4.0, 10.0, 2.0])  # Original

    if R is None:
        if use_reference_tuning:
            R = np.array([[10.0]])  # Reference v6
        else:
            R = np.array([[2.0]])  # Original

    # Compute trajectory durations based on pendulum period
    T_period = 2 * np.pi / np.sqrt(g / l)
    if use_reference_tuning:
        self.durations_to_try = [T_period, 1.5*T_period, 2.0*T_period]
    else:
        self.durations_to_try = [3.5, 4.0, 5.0]

    # ... rest of initialization
```

Then in `_plan_maneuver`:
```python
for θ_target in [θ_target_cw, θ_target_ccw]:
    for duration in self.durations_to_try:  # Use instance variable
        cost = self._plan_maneuver(s_pred, θ_target, duration)
        # ...
```

## Implementation Priority

### Phase 1: Minimal Changes (High Impact)
✅ **Just update the default Q and R matrices** (2-line change)

This alone will significantly change the feedback behavior to match reference.

### Phase 2: Duration Tuning (Medium Impact)
⚠️ Update trajectory duration search range

May improve swingup speed but requires more testing.

### Phase 3: Comprehensive Testing (Validation)
🔬 Create comparison test suite:

```python
# test_reference_match.py
def test_reference_vs_current():
    """Compare reference tuning vs original on standard test cases."""

    # Test case 1: Hanging down start
    s_init = np.array([0.0, 0.0, 0.0, 0.0])  # θ=0 (bottom)

    # Test case 2: Small perturbation
    s_init = np.array([0.2, 0.0, 0.0, 0.0])

    # Test case 3: Large angle
    s_init = np.array([np.pi/2, 0.0, 0.0, 0.0])

    # Compare:
    # - Trajectory cost
    # - Control effort (max |u|)
    # - Swingup time
    # - Final settling performance
```

## Expected Behavioral Changes

### With Reference Tuning (Q=[1,10,1,1], R=10):

**Positive Effects:**
- ✅ Faster, more aggressive swingup
- ✅ Higher angular velocity during swing → more energy injection
- ✅ Less cart position tracking during swing (focus on energy first)
- ✅ Better matches reference's aggressive strategy

**Potential Tradeoffs:**
- ⚠️ May approach cart position limits more
- ⚠️ Higher peak control effort
- ⚠️ More sensitive to initial conditions

### With Current Tuning (Q=[10,4,10,2], R=2):

**Characteristics:**
- Slower, more conservative swingup
- More emphasis on keeping cart centered
- Lower control effort (cheaper but slower)
- May be more robust to cart position constraints

## Mathematical Consistency Check

### Units and Scaling

Both implementations are **mathematically consistent** but use different unit systems:

**Reference (Dimensionless):**
```
Time: τ = t·√(g/l)
Length: same (meters)
Velocity: ω = v/√(g·l)

Dynamics: θ̈ = -sin(θ) - u·cos(θ)
Control: u = -λ_ẋ + λ_θ̇·cos(θ)
```

**Current (Physical):**
```
Time: t (seconds)
Length: same (meters)
Velocity: v (m/s)

Dynamics: θ̈ = -(g/l)·sin(θ) - (u/l)·cos(θ)
Control: u = -λ_ẋ + (λ_θ̇/l)·cos(θ)
```

These are **equivalent** with proper factor insertion. The current code correctly includes the (1/l) factor in the control law (line 251) and A matrix (line 302).

## Friction Handling ✓ VERIFIED CORRECT

**Both implementations:**
1. ✅ **Ignore friction in BVP** (optimal control assumption)
2. ✅ **Ignore friction in Riccati** (linearize around frictionless nominal)
3. ✅ **Include friction in simulation** (realistic dynamics)
4. ✅ **Include friction in prediction** (compensate planning delay)

This is **the correct approach** and matches the reference exactly.

## Summary: What to Change

### Minimum Required Changes (to match reference exactly):

```python
# In src/classical_control.py, lines 86-89:

# Change from:
Q = np.diag([10.0, 4.0, 10.0, 2.0])
R = np.array([[2.0]])

# Change to:
Q = np.diag([1.0, 10.0, 1.0, 1.0])   # Reference v6 values
R = np.array([[10.0]])                # Reference v6 value
```

### Optional Duration Change:

```python
# In src/classical_control.py, line 186:

# Add before the loop:
T_period = 2 * np.pi / np.sqrt(self.g / self.l)

# Change from:
for duration in [3.5, 4.0, 5.0]:

# Change to:
for duration in [T_period, 1.5*T_period, 2.0*T_period]:
```

## Testing Checklist

After making changes, verify:

- [ ] Swingup from θ=0 succeeds
- [ ] Control effort stays within bounds (|u| < 20 m/s²)
- [ ] Cart stays within track limits (|x| < 2.4m)
- [ ] Trajectory cost is reasonable (not >> 100)
- [ ] Final stabilization is stable
- [ ] Multiple initial conditions work

## Questions for User

Before implementing these changes, please confirm:

1. **Goal**: Do you want to exactly match reference v6 behavior, or maintain flexibility?
2. **Testing**: Do you have benchmark scenarios to validate the changes?
3. **Performance**: Are you seeing specific issues with current tuning that motivated this comparison?

Would you like me to implement these changes now?
