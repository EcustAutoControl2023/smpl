CSTR Environment
================

The CSTR environment in ``smpl.envs.cstrenv`` mirrors the reference dynamics in
``ref/Quantitative-Comparison-of-RL-and-MPC/Systems/sys_cstr.py`` without any
Matlab dependency on the environment side.

Safety Detection (Conservative Defaults)
---------------------------------------

The environment terminates with the error reward when any unsafe condition is detected.

Thermal runaway
^^^^^^^^^^^^^^^

- Triggered when ``T_next >= hard_temp_threshold``.
- Triggered when the reactor temperature rate exceeds
  ``runaway_temp_rate_threshold`` and reactor temperature is above
  ``runaway_temp_threshold``.
- Enabled/disabled by ``enable_thermal_safety``.

Default values:

- ``hard_temp_threshold = 130.0`` (deg)
- ``runaway_temp_rate_threshold = 0.5`` (deg/step)
- ``runaway_temp_threshold = 115.0`` (deg)

Control input envelope
^^^^^^^^^^^^^^^^^^^^^^

- Triggered when action is out of bounds.
- Enabled/disabled by ``enable_action_bounds_safety``.

All thresholds are configurable via ``CSTREnv`` constructor arguments and are
returned in ``info["safety_thresholds"]`` on each step.
