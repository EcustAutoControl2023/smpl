SMB Environment Guide
=====================

Overview
--------
- The SMB (Simulated Moving Bed) environment models an 8-column chromatography process for extract/raffinate separation. It follows the SMPL `smplEnvBase` API (`reset`, `step`) and mirrors the reference implementation in `ref/Quantitative-Comparison-of-RL-and-MPC/System/sys_smb.py`, with optional Matlab-based system identification (N4SID/NNARX) support.
- Core files: `smpl/envs/smbenv.py` (env + dynamics), `smpl/envs/smb_sysid.py` (Python/Matlab surrogates), `smpl/envs/smb_controller.py` (MPC controller), and helper scripts in `scripts/` for training, offline data generation, and plotting.

State, Action, Observation
--------------------------
- Internal state (`x_dim = 1602`): concentrations along 8 columns (4 species × 50 grid points each), mode (0–7) scaled by 0.1, and fractional switch timer.
- Action (`u_dim = 4`): flow rates for sections 2–5 (u1..u4). Bounds: `u_min = [0.0100, 0.0100, 0.0135, 0.0135]`, `u_max = [0.0135, 0.0135, 0.0165, 0.0165]`.
- Observation (`y_dim = 4`): outlet concentrations for extract and raffinate ports; `y_min = [0, 0, 0, 0]`, `y_max = [1, 1, 1, 1]`.
- Reference purity target: `ref = [0.99, 0.99]`.

Dynamics
--------
- One env `step`/`go_step` advances the PDE solver for one time slice: time index increments by 0.25; every 4 steps the mode increments (column roles rotate), consistent with 32 steps per hour in the reference.
- Column model: discretized Langmuir isotherm with mass transfer and diffusion, integrated via CasADi CVODES (`_column_step`).
- Disturbance: feed concentrations drift toward steady state with noise (`para_std = [0.01, 0.01]`).
- Tank accumulations track extract/raffinate outputs.

Reward/Cost
-----------
- Cost: `cost = 0.002 + 0.03*(ref[0] - pur_ex) + 0.03*(ref[1] - pur_ra) - (u2 - u1)`.
- Reward = `-cost`. Rewards are typically ≤ 0; they can become positive if `u2 - u1` dominates and purities are near target.

API
---
- `reset(seed=None, initial_state=None) -> (obs, info)`: if `initial_state` is provided, it seeds the observation (internal state stays at steady state).
- `step(action) -> (obs, reward, done, done, info)`: `info` includes `cost`, `mode`, `feed_concentration`, `extract_tank`, `raffinate_tank`, `timeout`.
- Normalization: `normalize=True` maps obs/actions to [-1, 1]; internal dynamics run in physical units; clipping enforces bounds before normalization.

System Identification Surrogates
--------------------------------
- `N4SIDSurrogate` (Python, `smpl/envs/smb_sysid.py`): y-as-state linear surrogate; fits A/B/bias on simulated data; exposes `dynamic_model`, `observe_model`, bounds, and steady-state control.
- `MatlabNNARXSurrogate` (uses reference NNARX sysid): generates rollout data from `SMBModel`, trains the reference NNARX (TensorFlow), and exposes `dynamic_model`/`observe_model` plus bounds/bias/scale.
- Compatibility aliases (`do_reset`, `get_observation`, etc.) allow reference utilities to operate on `SMBModel`.

Controllers
-----------
- `SMBMPCController` (CasADi): uses a provided sysid (e.g., `N4SIDSurrogate` or `MatlabNNARXSurrogate`) with the reference-style cost: purity penalties + flow term + base cost.
- `SMBBaselineSysId/SMBSystemId`: simple surrogate if no sysid is provided.
- Controllers expose `control(current_state)` returning physical actions.

Matlab Reference Integration
----------------------------
- `scripts/smb_offline_data_generation.py` supports Matlab-backed policies:
  - `mpc_n4sid_matlab`: fits Matlab N4SID via reference SYSID and runs reference MPC/estimator on `SMBModel`.
  - `mpc_nnarx_matlab`: fits Matlab NNARX via reference NNARX sysid and uses the stacking estimator.
- Requires Matlab Engine and the `LD_PRELOAD` shim per AGENTS instructions.

Data Generation & Plotting
--------------------------
- Offline datasets: `scripts/smb_offline_data_generation.py` generates d4rl-style pickles under `offline_datasets/smb/` with policies `mpc`, `random`, `mpc_n4sid_matlab`, `mpc_nnarx_matlab`. Key args: `--episodes`, `--horizon`, `--policy`, `--offline-switch` (N4SID data length), `--nnarx-data-length`.
- Plotting: `scripts/smb_plot_dataset.py` plots purities (with reference line) and actions per episode from a dataset pickle; args: `--dataset`, `--label`, `--ref`, `--out`.

Training Scripts
----------------
- Python-only MPC training: `scripts/smb_train.py` (SMBEnv + N4SID surrogate MPC) logs costs to `Data/smb_smpl/`.
- Matlab N4SID reference training: `scripts/smb_train_matlab.py` (requires Matlab Engine) logs offline costs/errors to `Data/smb_matlab/`.

Parameter Notes
---------------
- Reference `SmbConfig` highlights: `SYSID_signal_length=1e6`, `SYSID_signal_interval=160`, `STACK_u_order=4`, `STACK_y_order=4`, `NNARX_nn_model_node_num=[100, 60, 4]`, `NNARX_learning_rate=1e-4`, `NNARX_epochs=500`; KF covariances sized to N4SID dims; MPC prediction horizon often 10–32.
- Match horizons to reference: 32 steps ≈ 1 hour; set `max_steps = 32 * hours` to mirror reference runs.

Usage Tips
----------
- Rewards nearer to zero indicate better control; large negative spikes imply purity violations or aggressive flows.
- For pure SMPL use (no Matlab), prefer `N4SIDSurrogate` or `MatlabNNARXSurrogate` in `SMBMPCController`.
- Ensure datasets are saved with NumPy arrays (handled in provided scripts) for compatibility with offline RL libs (e.g., d3rlpy).
