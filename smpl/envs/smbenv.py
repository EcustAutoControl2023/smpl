"""
Simulated Moving Bed (SMB) environment implemented in SMPL style.

The dynamics mirror the reference implementation under ``ref/Quantitative-Comparison-of-RL-and-MPC``.
No Matlab dependencies are required; all computations rely on CasADi and NumPy.
"""

from pathlib import Path
from typing import Tuple

import casadi as ca
import numpy as np

from .utils import (
    denormalize_spaces,
    normalize_spaces,
    smplEnvBase,
)


def _zero_mean_scale(var: np.ndarray, var_min: np.ndarray, var_max: np.ndarray) -> np.ndarray:
    """Scale ``var`` to [-1, 1]."""
    return (2.0 * var - var_max - var_min) / (var_max - var_min)


def _zero_mean_descale(var: np.ndarray, var_min: np.ndarray, var_max: np.ndarray) -> np.ndarray:
    """Invert `_zero_mean_scale`."""
    return (var_max - var_min) / 2.0 * var + (var_max + var_min) / 2.0


class SMBModel:
    """Core SMB simulator that mirrors the reference SysSMB implementation."""

    def __init__(
        self,
        seed: int = 0,
        state_disturb: bool = False,
        measure_disturb: bool = False,
        para_disturb: bool = True,
        pressure_outlet: float = 1.0,
        pressure_fluid_viscosity: float = 1.0,
        pressure_fluid_density: float = 1.0,
        pressure_viscous_coeff: float = 0.05,
        pressure_inertial_coeff: float = 0.01,
        np_dtype=np.float64,
    ):
        self.process_type = "continuous"
        self.seed = seed
        self.state_disturb = state_disturb
        self.measure_disturb = measure_disturb
        self.para_disturb = para_disturb
        self.np_dtype = np_dtype

        self.grid_num = 50
        self.column_num = 8

        self.x_dim = 4 * self.column_num * self.grid_num + 2  # concentrations + mode + time
        self.u_dim = 4
        self.y_dim = 4
        self.p_dim = 2
        self.r_dim = 2

        self.state_std = np.zeros(self.x_dim, dtype=self.np_dtype)
        self.measure_std = np.zeros(self.y_dim, dtype=self.np_dtype)
        self.para_std = np.zeros(self.p_dim, dtype=self.np_dtype)
        if self.state_disturb:
            self.state_std = np.zeros(self.x_dim, dtype=self.np_dtype)
        if self.measure_disturb:
            self.measure_std = np.zeros(self.y_dim, dtype=self.np_dtype)
        if self.para_disturb:
            self.para_std = np.array([0.01, 0.01], dtype=self.np_dtype)

        self.x_min = np.zeros(self.x_dim, dtype=self.np_dtype)
        self.x_max = np.ones(self.x_dim, dtype=self.np_dtype)
        self.u_min = np.array([0.0100, 0.0100, 0.0135, 0.0135], dtype=self.np_dtype)
        self.u_max = np.array([0.0135, 0.0135, 0.0165, 0.0165], dtype=self.np_dtype)
        self.y_min = np.zeros(self.y_dim, dtype=self.np_dtype)
        self.y_max = np.ones(self.y_dim, dtype=self.np_dtype)
        self.p_min = np.array([0.9, 0.9], dtype=self.np_dtype)
        self.p_max = np.array([1.1, 1.1], dtype=self.np_dtype)

        self.scale_grad_for_a_column = 2.0 / (
            self.x_max[0 : 4 * self.grid_num] - self.x_min[0 : 4 * self.grid_num]
        )

        ini_state_path = (
            Path(__file__).resolve().parent.parent / "configdata" / "smb_initial_state.txt"
        )
        self.ini_x = self._load_state_array(ini_state_path)
        self.ini_u = np.array([0.013, 0.013, 0.014, 0.014], dtype=self.np_dtype)
        self.ini_y = np.array([0.0, 0.0, 0.0, 0.0], dtype=self.np_dtype)
        self.ini_p = np.array([1.0, 1.0], dtype=self.np_dtype)

        self.ss_x = self._load_state_array(ini_state_path)
        self.ss_y = np.array(
            [0.00006034, 0.00000000, 0.00000000, 0.00708570], dtype=self.np_dtype
        )
        self.ss_u = np.array([0.013, 0.013, 0.014, 0.014], dtype=self.np_dtype)
        self.ss_p = np.array([1.0, 1.0], dtype=self.np_dtype)
        self.ref = np.array([0.99, 0.99], dtype=self.np_dtype)

        self.p_now = self.ini_p
        self.extract_tank = np.array([0.0001, 0.0], dtype=self.np_dtype)
        self.raffinate_tank = np.array([0.0, 0.01], dtype=self.np_dtype)

        self.length = 1.0
        self.diameter = 0.1
        self.porosity = 0.66
        self.diffusion_coeff = 1e-5
        self.equili_const = [0.5, 0.2]
        self.langmuir_coeff = [5, 5]
        self.mass_transfer_coeff = [2, 2]
        self.switch_time = 120

        self.area = np.pi * self.diameter * self.diameter / 4
        self.volume = self.area * self.length
        self.grid_length = self.np_dtype(self.length / self.grid_num)
        self.time_grid_num = 30
        self.time_interval = self.np_dtype(self.switch_time / 4 / self.time_grid_num)
        self.section1_flow_rate = 0.022
        self.section4_flow_rate = 0.010
        self.pressure_outlet = self.np_dtype(pressure_outlet)
        self.pressure_fluid_viscosity = self.np_dtype(pressure_fluid_viscosity)
        self.pressure_fluid_density = self.np_dtype(pressure_fluid_density)
        self.pressure_viscous_coeff = self.np_dtype(pressure_viscous_coeff)
        self.pressure_inertial_coeff = self.np_dtype(pressure_inertial_coeff)

        self.step_fcn = self._make_step_function()

    @staticmethod
    def _load_state_array(path: Path) -> np.ndarray:
        if path.exists():
            return np.loadtxt(path).astype(np.float64)
        return np.zeros(1602, dtype=np.float64)

    def reset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        np.random.seed(self.seed)
        self.p_now = np.array([1.0, 1.0], dtype=self.np_dtype)
        self.extract_tank = np.array([0.0001, 0.0], dtype=self.np_dtype)
        self.raffinate_tank = np.array([0.0, 0.01], dtype=self.np_dtype)
        return (
            self.ss_x.copy(),
            self.ss_u.copy(),
            self.ss_y.copy(),
            self.ss_p.copy(),
            self.ref.copy(),
        )

    # Compatibility helpers for reference utilities/controllers
    def do_reset(self):
        return self.reset()

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        self.p_now, p_bdd = self._disturbance_generation(self.p_now)
        feed_con = p_bdd
        u = self._total_inputs(u)
        n_grid = self.grid_num
        mode = int(x[4 * self.column_num * n_grid] * 10)
        time_step_index = x[4 * self.column_num * n_grid + 1]

        role_index = np.arange(8, dtype=int)
        role_index = np.concatenate((role_index[8 - mode : 8], role_index[0 : 8 - mode]))

        s1, s2, s3, s4 = (
            x[0 * n_grid : 4 * n_grid],
            x[4 * n_grid : 8 * n_grid],
            x[8 * n_grid : 12 * n_grid],
            x[12 * n_grid : 16 * n_grid],
        )
        s5, s6, s7, s8 = (
            x[16 * n_grid : 20 * n_grid],
            x[20 * n_grid : 24 * n_grid],
            x[24 * n_grid : 28 * n_grid],
            x[28 * n_grid : 32 * n_grid],
        )

        for _ in range(self.time_grid_num):
            s1_in = np.array([s8[n_grid - 1], s8[3 * n_grid - 1]], dtype=self.np_dtype)
            s2_in = np.array([s1[n_grid - 1], s1[3 * n_grid - 1]], dtype=self.np_dtype)
            s3_in = np.array([s2[n_grid - 1], s2[3 * n_grid - 1]], dtype=self.np_dtype)
            s4_in = np.array([s3[n_grid - 1], s3[3 * n_grid - 1]], dtype=self.np_dtype)
            s5_in = np.array([s4[n_grid - 1], s4[3 * n_grid - 1]], dtype=self.np_dtype)
            s6_in = np.array([s5[n_grid - 1], s5[3 * n_grid - 1]], dtype=self.np_dtype)
            s7_in = np.array([s6[n_grid - 1], s6[3 * n_grid - 1]], dtype=self.np_dtype)
            s8_in = np.array([s7[n_grid - 1], s7[3 * n_grid - 1]], dtype=self.np_dtype)

            con_in = np.array(
                [s1_in, s2_in, s3_in, s4_in, s5_in, s6_in, s7_in, s8_in], dtype=self.np_dtype
            )
            con_in[np.where(role_index == 0)] = con_in[np.where(role_index == 0)] * u[7] / u[0]
            con_in[np.where(role_index == 1)] = con_in[np.where(role_index == 1)] * u[0] / u[1]
            con_in[np.where(role_index == 2)] = con_in[np.where(role_index == 2)]
            con_in[np.where(role_index == 3)] = con_in[np.where(role_index == 3)] * u[2] / u[3]
            con_in[np.where(role_index == 4)] = (
                con_in[np.where(role_index == 4)] * u[3] + feed_con * (u[4] - u[3])
            ) / u[4]
            con_in[np.where(role_index == 5)] = con_in[np.where(role_index == 5)] * u[4] / u[5]
            con_in[np.where(role_index == 6)] = con_in[np.where(role_index == 6)]
            con_in[np.where(role_index == 7)] = con_in[np.where(role_index == 7)] * u[6] / u[7]

            next_con1, s1_out = self._column_step(s1, con_in[0], u[role_index[0]], [])
            next_con2, s2_out = self._column_step(s2, con_in[1], u[role_index[1]], [])
            next_con3, s3_out = self._column_step(s3, con_in[2], u[role_index[2]], [])
            next_con4, s4_out = self._column_step(s4, con_in[3], u[role_index[3]], [])
            next_con5, s5_out = self._column_step(s5, con_in[4], u[role_index[4]], [])
            next_con6, s6_out = self._column_step(s6, con_in[5], u[role_index[5]], [])
            next_con7, s7_out = self._column_step(s7, con_in[6], u[role_index[6]], [])
            next_con8, s8_out = self._column_step(s8, con_in[7], u[role_index[7]], [])
            s_out = np.array(
                [s1_out, s2_out, s3_out, s4_out, s5_out, s6_out, s7_out, s8_out],
                dtype=self.np_dtype,
            )
            self.extract_tank += s_out[np.where(role_index == 1)][0] * u[1]
            self.raffinate_tank += s_out[np.where(role_index == 5)][0] * u[5]

            s1, s2, s3, s4 = next_con1, next_con2, next_con3, next_con4
            s5, s6, s7, s8 = next_con5, next_con6, next_con7, next_con8

        time_step_index += 0.25
        if time_step_index > 0.999:
            time_step_index = 0.0
            mode += 1
        if mode == 8:
            mode = 0

        next_x = np.concatenate(
            (
                next_con1,
                next_con2,
                next_con3,
                next_con4,
                next_con5,
                next_con6,
                next_con7,
                next_con8,
                [0.1 * mode],
                [time_step_index],
            )
        )
        return next_x

    def go_step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.step(x, u)

    def observe(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        u = self._total_inputs(u)
        y = np.zeros(self.y_dim, dtype=self.np_dtype)
        n_grid = self.grid_num
        mode = int(x[4 * self.column_num * n_grid] * 10) - 1
        if mode < 0:
            mode = 7
        role_index = np.arange(8, dtype=int)
        role_index = np.concatenate((role_index[8 - mode : 8], role_index[0 : 8 - mode]))
        s1 = np.array([x[0 * n_grid + n_grid - 1], x[0 * n_grid + 3 * n_grid - 1]]).squeeze()
        s2 = np.array([x[4 * n_grid + n_grid - 1], x[4 * n_grid + 3 * n_grid - 1]]).squeeze()
        s3 = np.array([x[8 * n_grid + n_grid - 1], x[8 * n_grid + 3 * n_grid - 1]]).squeeze()
        s4 = np.array([x[12 * n_grid + n_grid - 1], x[12 * n_grid + 3 * n_grid - 1]]).squeeze()
        s5 = np.array([x[16 * n_grid + n_grid - 1], x[16 * n_grid + 3 * n_grid - 1]]).squeeze()
        s6 = np.array([x[20 * n_grid + n_grid - 1], x[20 * n_grid + 3 * n_grid - 1]]).squeeze()
        s7 = np.array([x[24 * n_grid + n_grid - 1], x[24 * n_grid + 3 * n_grid - 1]]).squeeze()
        s8 = np.array([x[28 * n_grid + n_grid - 1], x[28 * n_grid + 3 * n_grid - 1]]).squeeze()
        s_out = np.array([s1, s2, s3, s4, s5, s6, s7, s8])
        y[0:2] = s_out[np.where(role_index == 1)][0] * u[1] + self.measure_std[0:2] * np.random.normal(
            0, 1, 2
        )
        y[2:4] = s_out[np.where(role_index == 5)][0] * u[5] + self.measure_std[2:4] * np.random.normal(
            0, 1, 2
        )
        return y

    def get_observation(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.observe(x, u)

    def cost(self, y: np.ndarray, u: np.ndarray) -> float:
        product = -1.0
        extract_purity_penalty = 0.03
        raffinate_purity_penalty = 0.03

        puri_e = y[0] / (y[0] + y[1] + 1e-10)
        puri_r = y[3] / (y[2] + y[3] + 1e-10)

        penalty_extract = extract_purity_penalty * (self.ref[0] - puri_e)
        penalty_raffinate = raffinate_purity_penalty * (self.ref[1] - puri_r)
        return 0.0041 + product * (u[2] - u[1]) + penalty_extract + penalty_raffinate

    def get_cost(self, y: np.ndarray, u: np.ndarray, ref: np.ndarray) -> float:
        # ref kept for signature parity; uses self.ref internally.
        return self.cost(y, u)

    def mode(self, x: np.ndarray) -> int:
        return int(x[4 * self.column_num * self.grid_num] * 10)

    def get_mode(self, x: np.ndarray) -> int:
        return self.mode(x)

    def feed_concentration(self) -> np.ndarray:
        return self.p_now

    def get_feed_concentration(self):
        return self.feed_concentration()

    def tank_information(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.extract_tank, self.raffinate_tank

    def get_tank_information(self):
        return self.tank_information()

    def role_index(self, x: np.ndarray) -> np.ndarray:
        mode = int(x[4 * self.column_num * self.grid_num] * 10)
        role_index = np.arange(self.column_num, dtype=int)
        return np.concatenate((role_index[self.column_num - mode :], role_index[: self.column_num - mode]))

    def compute_pressure_metrics(self, x: np.ndarray, u: np.ndarray) -> dict:
        total_u = self._total_inputs(np.asarray(u, dtype=self.np_dtype))
        role_index = self.role_index(np.asarray(x, dtype=self.np_dtype))
        column_flow_rates = np.asarray(total_u[role_index], dtype=self.np_dtype)
        controlled_mask = np.isin(role_index, np.array([2, 3, 4, 5], dtype=int))
        superficial_velocity = np.asarray(column_flow_rates / self.area, dtype=self.np_dtype)
        delta_p_columns = (
            self.pressure_viscous_coeff * self.pressure_fluid_viscosity * superficial_velocity
            + self.pressure_inertial_coeff
            * self.pressure_fluid_density
            * superficial_velocity
            * np.abs(superficial_velocity)
        ).astype(self.np_dtype)
        pin_columns = (
            self.pressure_outlet + np.cumsum(delta_p_columns[::-1], dtype=self.np_dtype)[::-1]
        ).astype(self.np_dtype)
        controlled_delta_p_columns = np.asarray(delta_p_columns[controlled_mask], dtype=self.np_dtype)
        controlled_pin_columns = np.asarray(pin_columns[controlled_mask], dtype=self.np_dtype)
        return {
            "section_flow_rates": np.asarray(total_u, dtype=self.np_dtype),
            "column_roles": np.asarray(role_index, dtype=int),
            "controlled_mask": np.asarray(controlled_mask, dtype=bool),
            "column_flow_rates": column_flow_rates,
            "superficial_velocity": superficial_velocity,
            "delta_p_columns": delta_p_columns,
            "delta_p_max": self.np_dtype(np.max(controlled_delta_p_columns)),
            "delta_p_sum": self.np_dtype(np.sum(controlled_delta_p_columns)),
            "delta_p_max_global": self.np_dtype(np.max(delta_p_columns)),
            "delta_p_sum_global": self.np_dtype(np.sum(delta_p_columns)),
            "pin_columns": pin_columns,
            "pin_max": self.np_dtype(np.max(controlled_pin_columns)),
            "pin_max_global": self.np_dtype(np.max(pin_columns)),
        }

    def _make_step_function(self):
        x_ca = ca.SX.sym("x", 4 * self.grid_num)
        u_ca = ca.SX.sym("u", 1)
        p_ca = ca.SX.sym("p", 0)
        up_ca = ca.vcat([u_ca, p_ca])

        x_d = _zero_mean_descale(x_ca, self.x_min[0 : 4 * self.grid_num], self.x_max[0 : 4 * self.grid_num])
        u_d = _zero_mean_descale(u_ca, self.u_min[0], self.u_max[0])
        p_d = _zero_mean_descale(p_ca, np.array([]), np.array([]))

        xdot = self._column_dynamics(x_d, u_d, p_d)
        xdot = np.multiply(xdot, self.scale_grad_for_a_column)
        ode = {"x": x_ca, "p": up_ca, "ode": xdot}
        return ca.integrator("Integrator", "cvodes", ode, 0, [self.time_interval], {})

    def _column_dynamics(self, x, u, p):
        y, v = x, u
        n_grid, diff, delz, porosity = (
            self.grid_num,
            self.diffusion_coeff,
            self.grid_length,
            self.porosity,
        )
        equili_const, mass_transfer_coeff, langmuir_coeff = (
            self.equili_const,
            self.mass_transfer_coeff,
            self.langmuir_coeff,
        )
        ee = (1 - porosity) / porosity
        xdot = ca.SX.zeros(4 * n_grid, 1)

        xdot[0 * n_grid] = 0
        xdot[1 * n_grid - 1] = diff * (y[n_grid - 1] - 2 * y[n_grid - 1] + y[n_grid - 2]) / (
            delz**2
        ) - v * (y[n_grid - 1] - y[n_grid - 2]) / delz - ee * mass_transfer_coeff[0] * (
            langmuir_coeff[0]
            * equili_const[0]
            * y[n_grid - 1]
            / (1 + equili_const[0] * y[n_grid - 1] + equili_const[1] * y[3 * n_grid - 1])
            - y[2 * n_grid - 1]
        )
        xdot[1 * n_grid] = mass_transfer_coeff[0] * (
            langmuir_coeff[0]
            * equili_const[0]
            * y[0]
            / (1 + equili_const[0] * y[0] + equili_const[1] * y[2 * n_grid])
            - y[n_grid]
        )
        xdot[2 * n_grid - 1] = mass_transfer_coeff[0] * (
            langmuir_coeff[0]
            * equili_const[0]
            * y[n_grid - 1]
            / (1 + equili_const[0] * y[n_grid - 1] + equili_const[1] * y[3 * n_grid - 1])
            - y[2 * n_grid - 1]
        )
        xdot[2 * n_grid] = 0
        xdot[3 * n_grid - 1] = diff * (y[3 * n_grid - 1] - 2 * y[3 * n_grid - 1] + y[3 * n_grid - 2]) / (
            delz**2
        ) - v * (y[3 * n_grid - 1] - y[3 * n_grid - 2]) / delz - ee * mass_transfer_coeff[1] * (
            langmuir_coeff[1]
            * equili_const[1]
            * y[3 * n_grid - 1]
            / (1 + equili_const[0] * y[n_grid - 1] + equili_const[1] * y[3 * n_grid - 1])
            - y[4 * n_grid - 1]
        )
        xdot[3 * n_grid] = mass_transfer_coeff[1] * (
            langmuir_coeff[1]
            * equili_const[1]
            * y[2 * n_grid]
            / (1 + equili_const[0] * y[0] + equili_const[1] * y[2 * n_grid])
            - y[3 * n_grid]
        )
        xdot[4 * n_grid - 1] = mass_transfer_coeff[1] * (
            langmuir_coeff[1]
            * equili_const[1]
            * y[3 * n_grid - 1]
            / (1 + equili_const[0] * y[n_grid - 1] + equili_const[1] * y[3 * n_grid - 1])
            - y[4 * n_grid - 1]
        )

        for i in range(n_grid - 2):
            xdot[0 * n_grid + i + 1] = diff * (y[i + 2] - 2 * y[i + 1] + y[i]) / (
                delz**2
            ) - v * (y[i + 1] - y[i]) / delz - ee * mass_transfer_coeff[0] * (
                langmuir_coeff[0]
                * equili_const[0]
                * y[i + 1]
                / (1 + equili_const[0] * y[i + 1] + equili_const[1] * y[2 * n_grid + i + 1])
                - y[n_grid + i + 1]
            )
            xdot[1 * n_grid + i + 1] = mass_transfer_coeff[0] * (
                langmuir_coeff[0]
                * equili_const[0]
                * y[i + 1]
                / (1 + equili_const[0] * y[i + 1] + equili_const[1] * y[2 * n_grid + i + 1])
                - y[n_grid + i + 1]
            )
            xdot[2 * n_grid + i + 1] = diff * (
                y[2 * n_grid + i + 2] - 2 * y[2 * n_grid + i + 1] + y[2 * n_grid + i]
            ) / (delz**2) - v * (y[2 * n_grid + i + 1] - y[2 * n_grid + i]) / delz - ee * mass_transfer_coeff[1] * (
                langmuir_coeff[1]
                * equili_const[1]
                * y[2 * n_grid + i + 1]
                / (1 + equili_const[0] * y[i + 1] + equili_const[1] * y[2 * n_grid + i + 1])
                - y[3 * n_grid + i + 1]
            )
            xdot[3 * n_grid + i + 1] = mass_transfer_coeff[1] * (
                langmuir_coeff[1]
                * equili_const[1]
                * y[2 * n_grid + i + 1]
                / (1 + equili_const[0] * y[i + 1] + equili_const[1] * y[2 * n_grid + i + 1])
                - y[3 * n_grid + i + 1]
            )
        return xdot

    def _column_step(self, x: np.ndarray, c_in: np.ndarray, u: float, p):
        p = np.asarray(p, dtype=self.np_dtype)
        x[0] = c_in[0]
        x[2 * self.grid_num] = c_in[1]

        scaled_x = _zero_mean_scale(x, self.x_min[0 : 4 * self.grid_num], self.x_max[0 : 4 * self.grid_num])
        scaled_u = _zero_mean_scale(u, self.u_min[0], self.u_max[0])
        scaled_p = _zero_mean_scale(p, np.array([]), np.array([]))
        scaled_up = np.hstack((scaled_u, scaled_p))

        result = self.step_fcn(x0=scaled_x, p=scaled_up)
        scaled_next_x = np.squeeze(np.array(result["xf"]))

        next_x = _zero_mean_descale(
            scaled_next_x, self.x_min[0 : 4 * self.grid_num], self.x_max[0 : 4 * self.grid_num]
        )
        terminal_con = np.array([next_x[self.grid_num - 1], next_x[3 * self.grid_num - 1]])
        return next_x, terminal_con

    def _total_inputs(self, u: np.ndarray) -> np.ndarray:
        total_u = np.array(
            [
                self.section1_flow_rate,
                self.section1_flow_rate,
                0.013,
                0.013,
                0.014,
                0.014,
                self.section4_flow_rate,
                self.section4_flow_rate,
            ],
            dtype=self.np_dtype,
        )
        total_u[2:6] = u
        return total_u

    def _disturbance_generation(self, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p_next = p - 0.1 * (p - self.ss_p) + self.para_std * np.random.normal(0, 1, self.p_dim)
        p_bdd = np.clip(p_next, self.p_min, self.p_max)
        return p_next, p_bdd


class SMBEnv(smplEnvBase):
    """Gym-style SMB environment following SMPL conventions."""

    def __init__(
        self,
        dense_reward: bool = True,
        normalize: bool = True,
        debug_mode: bool = False,
        max_steps: int = 32 * 6,
        error_reward: float = -100.0,
        model: SMBModel = None,
        seed: int = 0,
    ):
        self.model = model if model is not None else SMBModel(seed=seed)
        self.seed = seed
        super().__init__(
            dense_reward=dense_reward,
            normalize=normalize,
            debug_mode=debug_mode,
            action_dim=self.model.u_dim,
            observation_dim=self.model.y_dim,
            max_observations=self.model.y_max,
            min_observations=self.model.y_min,
            max_actions=self.model.u_max,
            min_actions=self.model.u_min,
            np_dtype=self.model.np_dtype,
            max_steps=max_steps,
            error_reward=error_reward,
        )
        self.reward_function = self._reward_from_cost
        self.done_calculator = self.done_calculator_standard
        self.reset()

    def reset(self, *, seed=None, options=None, initial_state=None):
        if seed is not None:
            self.seed = seed
            self.model.seed = seed
        x, u, y, _, _ = self.model.reset()
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        self.previous_state = x
        self.previous_action = u
        observation = y.copy()
        if initial_state is not None:
            # Accept provided observation as starting point; keep internal state at steady state.
            observation = np.array(initial_state, dtype=self.np_dtype)
        if self.normalize:
            observation, _, _ = normalize_spaces(observation, self.max_observations, self.min_observations)
        observation = observation.astype(self.observation_space.dtype)
        self.previous_observation = observation
        return observation, {}

    def step(self, action, normalize=None):
        reward = None
        done = None
        done_info = {"terminal": False, "timeout": False}
        normalize = self.normalize if normalize is None else normalize
        action = np.array(action, dtype=self.np_dtype)
        if normalize:
            action, _, _ = denormalize_spaces(action, self.max_actions, self.min_actions)

        pressure_metrics = self.model.compute_pressure_metrics(self.previous_state, action)
        next_state = self.model.step(self.previous_state, action)
        observation = self.model.observe(next_state, action)
        cost = self.model.cost(observation, action)

        reward = -cost
        obs_for_check = observation.astype(self.observation_space.dtype)
        reward = self.reward_function(self.previous_observation, action, obs_for_check, reward=reward)
        done, done_info = self.done_calculator(
            obs_for_check, self.step_count, reward, done=done, done_info=done_info
        )

        self.previous_state = next_state
        self.previous_action = action
        self.previous_observation = obs_for_check
        self.step_count += 1
        self.total_reward += reward

        if not self.dense_reward and not done:
            reward = 0.0
        elif not self.dense_reward:
            reward = self.total_reward

        observation = observation.clip(self.min_observations, self.max_observations)
        if normalize:
            observation, _, _ = normalize_spaces(observation, self.max_observations, self.min_observations)
        info = {
            "cost": cost,
            "mode": self.model.mode(next_state),
            "feed_concentration": self.model.feed_concentration().copy(),
            "extract_tank": self.model.tank_information()[0].copy(),
            "raffinate_tank": self.model.tank_information()[1].copy(),
            "section_flow_rates": pressure_metrics["section_flow_rates"].copy(),
            "column_roles": pressure_metrics["column_roles"].copy(),
            "controlled_mask": pressure_metrics["controlled_mask"].copy(),
            "column_flow_rates": pressure_metrics["column_flow_rates"].copy(),
            "superficial_velocity": pressure_metrics["superficial_velocity"].copy(),
            "delta_p_columns": pressure_metrics["delta_p_columns"].copy(),
            "delta_p_max": float(pressure_metrics["delta_p_max"]),
            "delta_p_sum": float(pressure_metrics["delta_p_sum"]),
            "delta_p_max_global": float(pressure_metrics["delta_p_max_global"]),
            "delta_p_sum_global": float(pressure_metrics["delta_p_sum_global"]),
            "pin_columns": pressure_metrics["pin_columns"].copy(),
            "pin_max": float(pressure_metrics["pin_max"]),
            "pin_max_global": float(pressure_metrics["pin_max_global"]),
            "timeout": done_info.get("timeout", False),
        }
        info.update(done_info)
        observation = observation.astype(self.observation_space.dtype)
        reward = float(reward)
        return observation, reward, done, done, info

    def _reward_from_cost(self, previous_observation, action, current_observation, reward=None):
        if reward is None:
            return self.error_reward
        if np.any(np.isnan(current_observation)) or np.any(np.isinf(current_observation)):
            return self.error_reward
        return reward

    def observation_beyond_box(self, observation):
        observation = np.asarray(observation, dtype=self.observation_space.dtype)
        return (
            (not self.observation_space.contains(observation))
            or np.any(np.isnan(observation))
            or np.any(np.isinf(observation))
        )
