"""Use the degradation model to compare race strategies.

Total race time for a plan = sum of predicted lap times over every stint
+ one pit-stop loss per stop. Fuel is ignored here because every strategy
burns the same fuel on the same laps, so it cancels out when comparing.

Rules applied:
  * a dry race must use at least two different compounds
  * a stint can't be longer than the oldest tyre actually seen in the data
    for that compound (the model shouldn't extrapolate beyond evidence)
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from .model import DegModel


def stint_time(model: DegModel, compound: str, length: int) -> float:
    ages = np.arange(1, length + 1)
    return float(np.sum(model.lap_time(compound, ages)))


def evaluate(model: DegModel, plan: list[tuple[str, int]], pit_loss: float) -> float:
    return sum(stint_time(model, c, n) for c, n in plan) + pit_loss * (len(plan) - 1)


def enumerate_strategies(model: DegModel, total_laps: int, pit_loss: float,
                         max_age: dict[str, int], max_stops: int = 2,
                         min_stint: int = 5) -> pd.DataFrame:
    """Try every 1- and 2-stop plan; return them ranked fastest first."""
    rows = []
    comps = model.compounds
    for stops in range(1, max_stops + 1):
        for seq in product(comps, repeat=stops + 1):
            if len(set(seq)) < 2:
                continue
            for cuts in _pit_laps(total_laps, stops, min_stint):
                lengths = np.diff([0, *cuts, total_laps])
                if any(n > max_age.get(c, 0) for c, n in zip(seq, lengths)):
                    continue
                plan = list(zip(seq, lengths.tolist()))
                rows.append({
                    "stops": stops,
                    "plan": " → ".join(f"{c[0]}({n})" for c, n in plan),
                    "compounds": "-".join(seq),
                    "pit_laps": tuple(cuts),
                    "race_time_s": evaluate(model, plan, pit_loss),
                })
    out = pd.DataFrame(rows).sort_values("race_time_s").reset_index(drop=True)
    out["gap_to_best_s"] = out["race_time_s"] - out["race_time_s"].iloc[0]
    return out


def best_per_sequence(strats: pd.DataFrame) -> pd.DataFrame:
    """The optimal pit lap(s) for each set of compounds.

    In this model the order of stints doesn't change total time (fuel is
    corrected out and there's no warm-up or traffic), so Soft→Hard and
    Hard→Soft are the same strategy - keep one of each.
    """
    s = strats.copy()
    s["set"] = s["compounds"].map(lambda c: "-".join(sorted(c.split("-"))))
    return (s.sort_values("race_time_s").groupby("set", as_index=False).first()
             .drop(columns="set").sort_values("race_time_s").reset_index(drop=True))


def _pit_laps(total_laps: int, stops: int, min_stint: int):
    if stops == 1:
        for a in range(min_stint, total_laps - min_stint + 1):
            yield (a,)
    elif stops == 2:
        for a in range(min_stint, total_laps - 2 * min_stint + 1):
            for b in range(a + min_stint, total_laps - min_stint + 1):
                yield (a, b)
    else:
        raise ValueError("Only 1- and 2-stop strategies are supported.")
