"""
Pressure Vessel ODE Simulator
==============================
Simulates a simplified industrial pressure vessel with coupled
pressure–temperature dynamics using scipy.integrate.solve_ivp.

Physics model (normal operation):
    dP/dt = (Q_in - k_v * P) / V  -  alpha * (T - T_env)
    dT/dt = (q_heater - h * A * (T - T_env)) / (m * Cp)

Where:
    P = vessel pressure (bar)
    T = vessel temperature (K)
    Q_in = inlet mass flow (kg/s)
    k_v  = valve discharge coefficient
    V    = vessel volume (m^3)
    alpha = thermal-pressure coupling coefficient
    q_heater = heater power (W)
    h    = heat transfer coefficient (W/m^2·K)
    A    = vessel surface area (m^2)
    m    = fluid mass (kg)
    Cp   = specific heat capacity (J/kg·K)
    T_env = ambient temperature (K)
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class VesselParams:
    """Physical parameters for the pressure vessel model."""
    V: float = 2.0          # vessel volume (m^3)
    Q_in: float = 0.5       # inlet mass flow (kg/s)
    k_v: float = 0.1        # valve discharge coefficient
    alpha: float = 0.002    # thermal-pressure coupling (bar/K/s)
    q_heater: float = 5000.0  # heater power (W)
    h: float = 10.0         # heat transfer coeff (W/m^2·K)
    A: float = 6.0          # vessel surface area (m^2)
    m: float = 100.0        # fluid mass (kg)
    Cp: float = 4186.0      # specific heat capacity (J/kg·K) — water
    T_env: float = 298.15   # ambient temperature (K) — ~25°C
    P0: float = 5.0         # initial pressure (bar)
    T0: float = 310.0       # initial temperature (K) — ~37°C


@dataclass
class FaultConfig:
    """Configuration for fault injection."""
    fault_type: str = "none"        # "none", "seal_degradation", "heater_drift", "blockage"
    onset_time: float = 50.0        # time (s) when fault begins
    severity: float = 1.0           # 0..1 scaling of fault magnitude
    # For progressive faults, ramp duration:
    ramp_duration: float = 20.0


@dataclass
class SimConfig:
    """Full simulation configuration."""
    params: VesselParams = field(default_factory=VesselParams)
    fault: FaultConfig = field(default_factory=FaultConfig)
    t_end: float = 100.0            # simulation duration (s)
    dt: float = 0.1                 # output time step (s)
    noise_std_P: float = 0.0        # Gaussian noise std for pressure sensor
    noise_std_T: float = 0.0        # Gaussian noise std for temperature sensor
    seed: int = 42


def _fault_factor(t: float, fault: FaultConfig) -> float:
    """Compute a smooth ramp factor [0, severity] starting at onset_time."""
    if t < fault.onset_time:
        return 0.0
    elapsed = t - fault.onset_time
    ramp = min(elapsed / max(fault.ramp_duration, 1e-6), 1.0)
    return ramp * fault.severity


def vessel_odes(
    t: float,
    y: np.ndarray,
    params: VesselParams,
    fault: FaultConfig,
) -> list[float]:
    """
    Right-hand side of the vessel ODE system.

    State vector y = [P, T].
    Returns [dP/dt, dT/dt].
    """
    P, T = y
    p = params

    # --- Fault modifications ---
    f = _fault_factor(t, fault)

    # Effective parameters under fault
    k_v_eff = p.k_v
    q_heater_eff = p.q_heater
    Q_in_eff = p.Q_in

    if fault.fault_type == "seal_degradation":
        # Seal leak → increased discharge coefficient (pressure drops faster)
        k_v_eff = p.k_v * (1.0 + 2.0 * f)

    elif fault.fault_type == "heater_drift":
        # Heater drifts upward → overheating
        q_heater_eff = p.q_heater * (1.0 + 1.5 * f)

    elif fault.fault_type == "blockage":
        # Sudden inlet blockage → reduced flow
        Q_in_eff = p.Q_in * (1.0 - 0.9 * f)

    # --- ODEs ---
    dP_dt = (Q_in_eff - k_v_eff * P) / p.V - p.alpha * (T - p.T_env)
    dT_dt = (q_heater_eff - p.h * p.A * (T - p.T_env)) / (p.m * p.Cp)

    return [dP_dt, dT_dt]


def simulate(config: SimConfig | None = None) -> dict[str, np.ndarray]:
    """
    Run the vessel simulation.

    Returns
    -------
    dict with keys:
        - "time": shape (N,) time array
        - "pressure": shape (N,) pressure readings (with noise if configured)
        - "temperature": shape (N,) temperature readings (with noise if configured)
        - "pressure_clean": shape (N,) noise-free pressure
        - "temperature_clean": shape (N,) noise-free temperature
        - "label": shape (N,) int array — 0 = normal, 1 = anomaly
        - "fault_type": str
    """
    if config is None:
        config = SimConfig()

    rng = np.random.default_rng(config.seed)
    p = config.params
    f = config.fault

    t_span = (0.0, config.t_end)
    t_eval = np.arange(0.0, config.t_end, config.dt)

    sol = solve_ivp(
        fun=lambda t, y: vessel_odes(t, y, p, f),
        t_span=t_span,
        y0=[p.P0, p.T0],
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    time = sol.t
    pressure_clean = sol.y[0]
    temperature_clean = sol.y[1]

    # Add sensor noise
    pressure = pressure_clean + rng.normal(0, config.noise_std_P, size=len(time))
    temperature = temperature_clean + rng.normal(0, config.noise_std_T, size=len(time))

    # Labels: 0 = normal, 1 = anomaly (after fault onset)
    if f.fault_type == "none":
        label = np.zeros(len(time), dtype=int)
    else:
        label = (time >= f.onset_time).astype(int)

    return {
        "time": time,
        "pressure": pressure,
        "temperature": temperature,
        "pressure_clean": pressure_clean,
        "temperature_clean": temperature_clean,
        "label": label,
        "fault_type": f.fault_type,
    }


if __name__ == "__main__":
    # Quick smoke test: simulate normal operation and print summary
    result = simulate()
    print("=== Vessel Simulator Smoke Test (Normal) ===")
    print(f"Time points: {len(result['time'])}")
    print(f"Pressure range: [{result['pressure'].min():.3f}, {result['pressure'].max():.3f}] bar")
    print(f"Temperature range: [{result['temperature'].min():.3f}, {result['temperature'].max():.3f}] K")
    print(f"Labels (unique): {np.unique(result['label'])}")
    print("[OK] Normal simulation OK")
