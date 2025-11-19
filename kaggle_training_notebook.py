#!/usr/bin/env python3
"""
Cart-Pendulum Training on Kaggle
=================================

This notebook trains a SAC agent for cart-pendulum swing-up control.

Setup:
1. GPU: Settings → Accelerator → GPU T4 x2
2. Internet: Settings → Internet → ON
3. Run all cells in order

Expected time: ~3-4 hours for full training (1.5M steps)
"""

# ============================================================================
# CELL 1: Clone Repository and Install Dependencies
# ============================================================================

# Clone repository
get_ipython().system('git clone https://github.com/ayyan-k98/cart-pendulum-stuff.git')
get_ipython().run_line_magic('cd', 'cart-pendulum-stuff')

# Install dependencies
get_ipython().system('pip install -q gymnasium stable-baselines3 scipy matplotlib tensorboard')

print("✓ Repository cloned and dependencies installed")


# ============================================================================
# CELL 2: Verify GPU and Environment
# ============================================================================

import torch
import os

print("="*80)
print("ENVIRONMENT CHECK")
print("="*80)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: No GPU detected! Enable GPU in Settings → Accelerator")

print(f"Working directory: {os.getcwd()}")
print(f"Output directory: /kaggle/working/")
print("="*80)


# ============================================================================
# CELL 3: Configure Training Parameters
# ============================================================================

# Training configuration (adjust as needed)
CONFIG = {
    'total_steps': 1_500_000,  # Full training (reduce to 100_000 for quick test)
    'n_envs': 8,               # Parallel environments
    'batch_size': 1024,        # Stable batch size
    'gradient_steps': 64,      # Efficient gradient updates
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'out_dir': '/kaggle/working/runs/sac_train',
    'seed': 42
}

print("TRAINING CONFIGURATION")
print("="*80)
for key, value in CONFIG.items():
    print(f"{key:20s}: {value}")
print("="*80)
print("\nEstimated training time:")
print(f"  - Quick test (100K steps):   ~8 minutes")
print(f"  - Standard (500K steps):     ~1.5 hours")
print(f"  - Full training (1.5M steps): ~3-4 hours")
print("="*80)


# ============================================================================
# CELL 4: Start Training
# ============================================================================

print("\nStarting training...")
print("Monitor progress below. Training will show:")
print("  - Phase 1: Stabilization (quick, ~5-10 min)")
print("  - Phase 2: Swing-up (main training)")
print("\n")

get_ipython().system(f"""python scripts/train.py \
    --total-steps {CONFIG['total_steps']} \
    --n-envs {CONFIG['n_envs']} \
    --batch-size {CONFIG['batch_size']} \
    --gradient-steps {CONFIG['gradient_steps']} \
    --device {CONFIG['device']} \
    --out-dir {CONFIG['out_dir']} \
    --seed {CONFIG['seed']}""")

print("\n✓ Training complete!")


# ============================================================================
# CELL 5: Monitor TensorBoard (Optional - Run in Parallel)
# ============================================================================

# Uncomment to monitor training in real-time
# get_ipython().run_line_magic('load_ext', 'tensorboard')
# get_ipython().run_line_magic('tensorboard', '--logdir /kaggle/working/runs/sac_train/phase2/logs')


# ============================================================================
# CELL 6: Load and Test Trained Model
# ============================================================================

from src import CartPendulumEnv, rollout_rl_timed
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from gymnasium.wrappers import TimeLimit
import numpy as np
import pandas as pd

print("\n" + "="*80)
print("LOADING TRAINED MODEL")
print("="*80)

# Load model
model_path = f"{CONFIG['out_dir']}/phase2/sac_model.zip"
vecnorm_path = f"{CONFIG['out_dir']}/phase2/vecnormalize.pkl"

def make_env():
    env = CartPendulumEnv()
    return TimeLimit(env, max_episode_steps=2000)

dummy_env = DummyVecEnv([make_env])
vec_env = VecNormalize.load(vecnorm_path, dummy_env)
vec_env.training = False
vec_env.norm_reward = False

model = SAC.load(model_path, device='cpu')

print(f"✓ Model loaded from: {model_path}")
print(f"✓ VecNormalize loaded from: {vecnorm_path}")


# ============================================================================
# CELL 7: Evaluate on Multiple Test Cases
# ============================================================================

print("\n" + "="*80)
print("EVALUATION ON MULTIPLE INITIAL STATES")
print("="*80)

# Test cases: challenging initial angles
test_cases = [
    ("Easy stabilization", np.array([np.deg2rad(5), 0.0, 0.0, 0.0])),
    ("Moderate swing-up", np.array([np.deg2rad(-90), 0.0, 0.0, 0.0])),
    ("Hard swing-up", np.array([np.deg2rad(-160), 0.0, 0.0, 0.0])),
    ("Inverted start", np.array([np.deg2rad(180), 0.0, 0.0, 0.0])),
]

results = []

for name, test_state in test_cases:
    trajectory, timing = rollout_rl_timed(
        model, vec_env, test_state, max_seconds=40.0
    )

    final_theta_deg = np.rad2deg(trajectory['theta'].iloc[-1])
    success = abs(trajectory['theta'].iloc[-1]) < np.deg2rad(10)
    max_cart_pos = trajectory['x'].abs().max()

    results.append({
        'Test Case': name,
        'Initial θ (°)': np.rad2deg(test_state[0]),
        'Final θ (°)': final_theta_deg,
        'Success': '✓' if success else '✗',
        'Max |x| (m)': max_cart_pos,
        'Duration (s)': trajectory['time'].iloc[-1],
        'Inference (ms)': timing['inference_time_mean_ms']
    })

# Display results
results_df = pd.DataFrame(results)
print("\n")
print(results_df.to_string(index=False))
print("\n")

# Summary statistics
success_rate = sum(1 for r in results if r['Success'] == '✓') / len(results)
print(f"Success rate: {success_rate*100:.1f}% ({sum(1 for r in results if r['Success'] == '✓')}/{len(results)})")
print("="*80)


# ============================================================================
# CELL 8: Save Output Dataset
# ============================================================================

import shutil
import datetime

print("\n" + "="*80)
print("SAVING OUTPUT DATASET")
print("="*80)

# Create output dataset directory
output_dataset_dir = "/kaggle/working/cart_pendulum_trained_model"
os.makedirs(output_dataset_dir, exist_ok=True)

# Copy model files
shutil.copytree(
    f"{CONFIG['out_dir']}/phase1",
    f"{output_dataset_dir}/phase1",
    dirs_exist_ok=True
)
shutil.copytree(
    f"{CONFIG['out_dir']}/phase2",
    f"{output_dataset_dir}/phase2",
    dirs_exist_ok=True
)

# Save evaluation results
results_df.to_csv(f"{output_dataset_dir}/evaluation_results.csv", index=False)

# Create README
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
readme_content = f"""# Cart-Pendulum Trained Model

## Training Details
- **Trained on**: {timestamp}
- **Total steps**: {CONFIG['total_steps']:,}
- **Device**: {CONFIG['device']}
- **Episode length**: 2000 steps (40 seconds at 50Hz)
- **Hard wall termination**: |x| > 2.4m
- **Curriculum learning**: Phase 1 (stabilization) → Phase 2 (swing-up)

## Model Files
- `phase1/sac_stabilize_model.zip` - Phase 1 stabilization model
- `phase1/vecnormalize_stabilize.pkl` - Phase 1 normalization stats
- `phase2/sac_model.zip` - **Final trained model**
- `phase2/vecnormalize.pkl` - **Final normalization stats**
- `phase2/logs/` - TensorBoard training logs
- `evaluation_results.csv` - Performance on test cases

## Evaluation Results

Success rate: {success_rate*100:.1f}%

{results_df.to_string(index=False)}

## Usage

```python
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from gymnasium.wrappers import TimeLimit
from src import CartPendulumEnv
import numpy as np

# Load model
model = SAC.load("phase2/sac_model.zip", device='cpu')

# Create environment
def make_env():
    env = CartPendulumEnv()
    return TimeLimit(env, max_episode_steps=2000)

dummy_env = DummyVecEnv([make_env])
vec_env = VecNormalize.load("phase2/vecnormalize.pkl", dummy_env)
vec_env.training = False
vec_env.norm_reward = False

# Run rollout
initial_state = np.array([np.deg2rad(-160), 0.0, 0.0, 0.0])
obs_raw = env.get_obs(initial_state)
obs = vec_env.normalize_obs(np.array([obs_raw]))[0]

done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step([action])
```

## Repository
https://github.com/ayyan-k98/cart-pendulum-stuff
"""

with open(f"{output_dataset_dir}/README.md", "w") as f:
    f.write(readme_content)

print(f"✓ Output saved to: {output_dataset_dir}")
print(f"\nFiles saved:")
print(f"  - Phase 1 model and stats")
print(f"  - Phase 2 model and stats")
print(f"  - TensorBoard logs")
print(f"  - Evaluation results CSV")
print(f"  - README with usage instructions")

print("\n" + "="*80)
print("TO PRESERVE THIS MODEL:")
print("="*80)
print("1. Click 'Save Version' at top right of this notebook")
print("2. Wait for save to complete (~1 min)")
print("3. Your model will be saved as a Kaggle Output Dataset")
print("4. Access later via: /kaggle/input/<your-notebook-name>/")
print("="*80)

print("\n✓ ALL DONE!")
