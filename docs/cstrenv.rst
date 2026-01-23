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

- Triggered when the reactor temperature rate exceeds ``runaway_temp_rate_threshold``.
- Optionally gated by a minimum reactor temperature ``runaway_temp_threshold``.

Default values:

- ``runaway_temp_rate_threshold = 0.5`` (deg/step)
- ``runaway_temp_threshold = 115.0`` (deg)

Control input envelope
^^^^^^^^^^^^^^^^^^^^^^

- Triggered when action is out of bounds.
- Triggered when per-step ``|delta u|`` exceeds ``input_rate_threshold``.

Default values:

- ``input_rate_threshold = [3.0, 500.0]`` for ``[u1, u2]``

All thresholds are configurable via ``CSTREnv`` constructor arguments and are
returned in ``info["safety_thresholds"]`` on each step.
