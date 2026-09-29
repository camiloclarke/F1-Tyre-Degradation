"""Estimate a car's degradation live, lap by lap, with a Kalman filter.

On the pit wall you don't get to fit a model after the race - you need to
know *during* the stint how fast the tyres are going off. This treats the
tyre as a hidden state and updates the estimate every time a lap time comes in.

State:        x = [pace, deg]      (current tyre pace in s, degradation in s/lap)
Each lap:     pace <- pace + deg   (tyres get slower by `deg` every lap)
              deg  <- deg          (deg rate drifts slowly)
Measurement:  lap_time = pace + noise

It's the same constant-velocity filter as a tracking problem - here the
"position" is lap time and the "velocity" is the degradation rate.
"""
from __future__ import annotations

import numpy as np


def kalman_deg(lap_times, meas_std: float = 0.35, pace_drift: float = 0.02,
               deg_drift: float = 0.005, deg_prior: float = 0.05, deg_prior_std: float = 0.1):
    """Run the filter over one stint of fuel-corrected lap times.

    Returns arrays (pace, deg, deg_std) - one value per lap, each using
    only the laps seen so far (no hindsight).
    """
    z = np.asarray(lap_times, dtype=float)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])   # state transition
    H = np.array([[1.0, 0.0]])               # we only measure pace
    Q = np.diag([pace_drift**2, deg_drift**2])
    R = np.array([[meas_std**2]])

    x = np.array([z[0], deg_prior])
    P = np.diag([meas_std**2, deg_prior_std**2])
    pace, deg, deg_std = [x[0]], [x[1]], [np.sqrt(P[1, 1])]

    for zk in z[1:]:
        x = F @ x                                 # predict
        P = F @ P @ F.T + Q
        y = zk - H @ x                            # innovation
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)            # Kalman gain
        x = x + (K @ y).ravel()                   # update
        P = (np.eye(2) - K @ H) @ P
        pace.append(x[0]); deg.append(x[1]); deg_std.append(np.sqrt(P[1, 1]))

    return np.array(pace), np.array(deg), np.array(deg_std)
