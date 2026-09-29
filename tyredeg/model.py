"""Fit a tyre degradation model to fuel-corrected lap times.

Model (one equation for every clean lap in the race):

    lap_time = driver_pace[d] + compound_offset[c] + deg[c] * age (+ quad[c] * age^2)

  * driver_pace   - each driver/car has its own baseline speed
  * compound_offset - how much faster/slower a fresh tyre of that compound is
  * deg           - seconds lost per lap of tyre age (the degradation rate)
  * quad          - optional curvature (tyres often fall off a "cliff")

Fitting all drivers together with their own baselines means a slow car on
softs doesn't make the soft tyre look slow. It's solved as ordinary least
squares with numpy - no black box.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

COMPOUND_ORDER = ("SOFT", "MEDIUM", "HARD")


@dataclass
class DegModel:
    compounds: list[str]
    offset: dict[str, float]          # fresh-tyre pace relative to reference compound (s)
    deg: dict[str, float]             # s/lap
    deg_se: dict[str, float]          # standard error of deg (s/lap)
    quad: dict[str, float]            # s/lap^2 (0 if linear model)
    driver_pace: dict[str, float]     # baseline pace per driver (s)
    residual_std: float               # typical lap-to-lap noise left over (s)
    r2: float
    n_laps: int
    quadratic: bool = False
    stint_counts: dict[str, int] = field(default_factory=dict)
    fuel_effect: float = 0.0          # s per lap remaining (fuel + track evolution)
    fuel_effect_se: float = 0.0
    fuel_fitted: bool = False

    @property
    def reference_pace(self) -> float:
        """A representative car: the median driver baseline."""
        return float(np.median(list(self.driver_pace.values())))

    def lap_time(self, compound: str, age, driver: str | None = None):
        """Predicted fuel-corrected lap time on `compound` at tyre age `age`."""
        age = np.asarray(age, dtype=float)
        base = self.driver_pace[driver] if driver else self.reference_pace
        return base + self.offset[compound] + self.deg[compound] * age + self.quad[compound] * age**2

    def summary(self) -> str:
        lines = [f"Fitted on {self.n_laps} clean laps  |  R^2 = {self.r2:.3f}  |  "
                 f"residual noise = {self.residual_std:.3f} s"]
        how = f"fitted from data, ± {1.96*self.fuel_effect_se:.3f}" if self.fuel_fitted else "assumed"
        lines.append(f"  Fuel + track evolution: {self.fuel_effect:.3f} s per lap remaining ({how})")
        for c in self.compounds:
            q = f", curvature {self.quad[c]*1000:+.2f} ms/lap^2" if self.quadratic else ""
            lines.append(
                f"  {c:<6}  fresh-tyre offset {self.offset[c]:+.3f} s   "
                f"deg {self.deg[c]:+.3f} ± {1.96*self.deg_se[c]:.3f} s/lap (95% CI){q}"
                f"   [{self.stint_counts.get(c, 0)} stints]"
            )
        return "\n".join(lines)


def fit_degradation(df: pd.DataFrame, quadratic: bool = False, fit_fuel: bool = True,
                    min_laps_per_compound: int = 30) -> DegModel:
    """Least-squares fit of the model above to the cleaned lap data.

    With fit_fuel=True the race-progress effect is estimated from the data too:

        lap_time = ... + k * laps_remaining

    k soaks up fuel burn AND the track getting faster as rubber goes down.
    It's separable from tyre age because different cars pit on different
    laps, so "age 5" happens at many different points in the race.
    """
    counts = df["Compound"].value_counts()
    compounds = [c for c in COMPOUND_ORDER if counts.get(c, 0) >= min_laps_per_compound]
    if not compounds:
        raise ValueError("Not enough clean dry laps to fit a model.")
    df = df[df["Compound"].isin(compounds)]

    drivers = sorted(df["Driver"].unique())
    ref = compounds[0]                       # offsets are relative to this compound
    age = df["TyreLife"].to_numpy(dtype=float)
    laps_left = (df.attrs["total_laps"] - df["LapNumber"]).to_numpy(dtype=float)

    cols, names = [], []
    if fit_fuel:
        y = df["LapTime_s"].to_numpy()
        cols.append(laps_left); names.append(("fuel", "k"))
    else:
        y = df["FuelCorrected_s"].to_numpy()
    for d in drivers:                                       # driver baselines
        cols.append((df["Driver"] == d).to_numpy(float)); names.append(("drv", d))
    for c in compounds[1:]:                                 # compound offsets
        cols.append((df["Compound"] == c).to_numpy(float)); names.append(("off", c))
    for c in compounds:                                     # deg slopes
        is_c = (df["Compound"] == c).to_numpy(float)
        cols.append(is_c * age); names.append(("deg", c))
        if quadratic:
            cols.append(is_c * age**2); names.append(("quad", c))
    X = np.column_stack(cols)

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))

    vals = {n: (b, s) for n, b, s in zip(names, beta, se)}
    offset = {c: (vals[("off", c)][0] if c != ref else 0.0) for c in compounds}
    deg = {c: vals[("deg", c)][0] for c in compounds}
    deg_se = {c: vals[("deg", c)][1] for c in compounds}
    quad = {c: (vals[("quad", c)][0] if quadratic else 0.0) for c in compounds}
    driver_pace = {d: vals[("drv", d)][0] for d in drivers}
    r2 = 1 - (resid @ resid) / np.sum((y - y.mean()) ** 2)
    stints = df.drop_duplicates(["Driver", "Stint"])["Compound"].value_counts().to_dict()

    if fit_fuel:
        k, k_se = vals[("fuel", "k")]
        # R^2 on the fuel-corrected series, comparable with the fixed-fuel fit
        yc = y - k * laps_left
        r2 = 1 - (resid @ resid) / np.sum((yc - yc.mean()) ** 2)
    else:
        k, k_se = df.attrs.get("fuel_effect", 0.0), 0.0

    return DegModel(compounds, offset, deg, deg_se, quad, driver_pace,
                    float(np.sqrt(sigma2)), float(r2), len(y), quadratic, stints,
                    float(k), float(k_se), fit_fuel)


def apply_fuel_correction(df: pd.DataFrame, model: DegModel) -> pd.DataFrame:
    """Recompute fuel-corrected times using the model's (fitted) fuel effect."""
    df = df.copy()
    df["FuelCorrected_s"] = df["LapTime_s"] - model.fuel_effect * (df.attrs["total_laps"] - df["LapNumber"])
    return df


def detrend_for_plot(df: pd.DataFrame, model: DegModel) -> pd.Series:
    """Remove each driver's baseline so all cars can be shown on one plot."""
    shift = df["Driver"].map(model.driver_pace) - model.reference_pace
    return df["FuelCorrected_s"] - shift
