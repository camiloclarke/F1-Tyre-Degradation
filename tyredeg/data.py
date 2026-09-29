"""Load a race from FastF1 and turn it into clean, fuel-corrected lap data.

Race lap times mix several effects together:
  * tyre degradation (what we want)
  * fuel burn - the car gets lighter, so laps get faster as the race goes on
  * traffic, safety cars, pit stops, driver mistakes (noise we want to remove)

This module removes the obvious noise and corrects for fuel, so the
remaining trend within a stint is mostly the tyre.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

# Typical F1 fuel effect: ~0.03 s per kg of fuel, burning ~1.8 kg per lap
# -> roughly 0.055 s per lap. It varies by circuit, so it's a parameter.
DEFAULT_FUEL_EFFECT = 0.055  # seconds per lap of fuel remaining

DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")


def load_race(year: int, race: str | int, cache_dir: str = "cache"):
    """Download (or read from cache) a race session using FastF1."""
    import fastf1  # imported here so the rest of the package works without it

    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    session = fastf1.get_session(year, race, "R")
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    return session


def pit_loss_from_session(session) -> float:
    """Estimate time lost to a pit stop from the actual race.

    For each stop: (in-lap + out-lap) - 2 x that driver's median clean lap.
    Returns the median across all stops, which is robust to slow stops.
    """
    laps = session.laps
    losses = []
    for drv, dl in laps.groupby("Driver"):
        lt = dl["LapTime"].dt.total_seconds()
        clean = dl[dl["PitInTime"].isna() & dl["PitOutTime"].isna()]
        ref = clean["LapTime"].dt.total_seconds().median()
        if np.isnan(ref):
            continue
        for idx in dl.index[dl["PitInTime"].notna()]:
            nxt = dl.index[dl.index.get_loc(idx) + 1] if dl.index.get_loc(idx) + 1 < len(dl) else None
            if nxt is None or pd.isna(dl.at[nxt, "PitOutTime"]):
                continue
            inlap, outlap = lt.loc[idx], lt.loc[nxt]
            if np.isnan(inlap) or np.isnan(outlap):
                continue
            losses.append(inlap + outlap - 2 * ref)
    losses = [x for x in losses if 10 < x < 45]  # drop red flags / penalties
    return float(np.median(losses)) if losses else 22.0


def clean_laps(session, fuel_effect: float = DEFAULT_FUEL_EFFECT) -> pd.DataFrame:
    """Return green-flag, representative race laps with a fuel-corrected time.

    Columns: Driver, Team, LapNumber, Stint, Compound, TyreLife,
             LapTime_s, FuelCorrected_s
    """
    laps = session.laps
    total_laps = int(laps["LapNumber"].max())

    laps = laps.pick_wo_box()                          # drop in-laps and out-laps
    laps = laps.pick_track_status("1", how="equals")   # green flag only (no SC/VSC/yellow)
    laps = laps[laps["LapNumber"] > 1]                 # lap 1 is a standing start
    laps = laps[laps["IsAccurate"] == True]            # FastF1's own timing sanity check
    laps = laps[laps["Compound"].isin(DRY_COMPOUNDS)]
    laps = laps.dropna(subset=["LapTime", "TyreLife"])

    df = pd.DataFrame({
        "Driver": laps["Driver"].values,
        "Team": laps["Team"].values,
        "LapNumber": laps["LapNumber"].astype(int).values,
        "Stint": laps["Stint"].astype(int).values,
        "Compound": laps["Compound"].values,
        "TyreLife": laps["TyreLife"].astype(int).values,
        "LapTime_s": laps["LapTime"].dt.total_seconds().values,
    })
    df = remove_outliers(df)

    # Fuel correction: express every lap as if run on an empty tank.
    # A lap with more fuel left is slower, so we subtract that penalty.
    fuel_laps_left = total_laps - df["LapNumber"]
    df["FuelCorrected_s"] = df["LapTime_s"] - fuel_effect * fuel_laps_left
    df.attrs["total_laps"] = total_laps
    df.attrs["fuel_effect"] = fuel_effect
    return df.reset_index(drop=True)


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop laps that clearly aren't representative (traffic, mistakes, etc.).

    1. Anything slower than 107% of the fastest clean lap in the race.
    2. Within each driver's stint, anything more than 3 robust standard
       deviations (via the median absolute deviation) from the stint median.
    """
    df = df[df["LapTime_s"] < 1.07 * df["LapTime_s"].min()]

    grp = df.groupby(["Driver", "Stint"])["LapTime_s"]
    med = grp.transform("median")
    mad = (df["LapTime_s"] - med).abs().groupby([df["Driver"], df["Stint"]]).transform("median") * 1.4826
    keep = (mad == 0) | mad.isna() | ((df["LapTime_s"] - med).abs() < 3 * mad)
    return df[keep]
