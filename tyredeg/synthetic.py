"""Generate a fake race with known degradation - used to check the model
recovers the numbers we put in (and to run the code without internet)."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRUE = {"SOFT": (0.0, 0.080), "MEDIUM": (0.45, 0.045), "HARD": (0.85, 0.025)}


def fake_race(total_laps=57, n_drivers=20, fuel_effect=0.055, noise=0.3, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    plans = [("SOFT", "HARD"), ("MEDIUM", "HARD"), ("SOFT", "MEDIUM", "HARD"), ("HARD", "MEDIUM")]
    for i in range(n_drivers):
        drv = f"D{i:02d}"
        pace = 90 + 0.08 * i + rng.normal(0, 0.1)
        seq = plans[i % len(plans)]
        cuts = np.sort(rng.choice(np.arange(12, total_laps - 10), len(seq) - 1, replace=False))
        bounds = [1, *cuts, total_laps + 1]
        for s, comp in enumerate(seq):
            off, deg = TRUE[comp]
            for age, lap in enumerate(range(bounds[s], bounds[s + 1]), start=1):
                fuel = fuel_effect * (total_laps - lap)
                t = pace + off + deg * age + fuel + rng.normal(0, noise)
                rows.append((drv, "Team", lap, s + 1, comp, age, t))
    df = pd.DataFrame(rows, columns=["Driver", "Team", "LapNumber", "Stint", "Compound",
                                     "TyreLife", "LapTime_s"])
    df = df[df["LapNumber"] > 1]
    df["FuelCorrected_s"] = df["LapTime_s"] - fuel_effect * (total_laps - df["LapNumber"])
    df.attrs["total_laps"] = total_laps
    df.attrs["fuel_effect"] = fuel_effect
    return df.reset_index(drop=True)
