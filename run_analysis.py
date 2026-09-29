"""Tyre degradation + strategy analysis for one F1 race.

Examples:
    python run_analysis.py --year 2024 --race Bahrain
    python run_analysis.py --year 2024 --race Silverstone --driver NOR --quadratic
    python run_analysis.py --synthetic          # offline test with known answers
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from tyredeg.data import DEFAULT_FUEL_EFFECT, clean_laps, load_race, pit_loss_from_session
from tyredeg.kalman import kalman_deg
from tyredeg.model import apply_fuel_correction, fit_degradation
from tyredeg.plots import plot_degradation, plot_kalman, plot_strategy, plot_strategy_ranking
from tyredeg.strategy import best_per_sequence, enumerate_strategies


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--race", default="Bahrain", help="Grand Prix name, circuit or round number")
    ap.add_argument("--driver", help="Driver code for the live Kalman example (default: winner)")
    ap.add_argument("--fuel-effect", type=float, default=DEFAULT_FUEL_EFFECT,
                    help="Assumed s/lap of fuel remaining, used only with --fixed-fuel (default %(default)s)")
    ap.add_argument("--fixed-fuel", action="store_true",
                    help="Use the assumed fuel effect instead of fitting it from the data")
    ap.add_argument("--pit-loss", type=float, help="Override the pit loss estimated from the race")
    ap.add_argument("--quadratic", action="store_true", help="Allow a tyre 'cliff' (curvature)")
    ap.add_argument("--synthetic", action="store_true", help="Use a fake race with known deg")
    args = ap.parse_args()

    if args.synthetic:
        from tyredeg.synthetic import TRUE, fake_race
        df = fake_race(fuel_effect=args.fuel_effect)
        pit_loss, name, winner = args.pit_loss or 22.0, "Synthetic GP", "D00"
    else:
        race_arg = int(args.race) if str(args.race).isdigit() else args.race
        session = load_race(args.year, race_arg)
        df = clean_laps(session, fuel_effect=args.fuel_effect)
        pit_loss = args.pit_loss or pit_loss_from_session(session)
        name = f"{args.year} {session.event['EventName']}"
        winner = session.results.sort_values("Position").iloc[0]["Abbreviation"]

    total_laps = df.attrs["total_laps"]
    slug = name.lower().replace(" ", "_")
    out = os.path.join("figures", slug)
    os.makedirs(out, exist_ok=True)

    # 1. Degradation model -------------------------------------------------
    model = fit_degradation(df, quadratic=args.quadratic, fit_fuel=not args.fixed_fuel)
    df = apply_fuel_correction(df, model)
    print(f"\n=== {name} ({total_laps} laps) ===")
    print(model.summary())
    if args.synthetic:
        print("  true values:", {c: TRUE[c][1] for c in model.compounds},
              f"| true fuel effect {args.fuel_effect}")
    plot_degradation(df, model, f"{name}: tyre degradation by compound",
                     os.path.join(out, "degradation.png"))

    # 2. Live Kalman estimate for one driver's longest stint ----------------
    drv = (args.driver or winner).upper()
    d = df[df["Driver"] == drv]
    if d.empty:
        raise SystemExit(f"No clean laps for driver {drv}")
    stint = d.groupby("Stint").size().idxmax()
    s = d[d["Stint"] == stint].sort_values("TyreLife")
    comp = s["Compound"].iloc[0]
    pace, deg, deg_std = kalman_deg(s["FuelCorrected_s"], meas_std=model.residual_std)
    plot_kalman(s, pace, deg, deg_std, model.deg[comp],
                f"{drv}, stint {stint} ({comp.title()})", comp,
                os.path.join(out, "kalman_live_deg.png"))
    print(f"\nKalman ({drv} stint {stint}, {comp}): final deg estimate "
          f"{deg[-1]:+.3f} ± {1.96*deg_std[-1]:.3f} s/lap  vs whole-race fit {model.deg[comp]:+.3f}")

    # 3. Strategy comparison ------------------------------------------------
    max_age = df.groupby("Compound")["TyreLife"].max().to_dict()
    strats = enumerate_strategies(model, total_laps, pit_loss, max_age)
    best = best_per_sequence(strats)
    plot_strategy_ranking(best, max_age, f"{name}: fastest plan for each tyre combination",
                          os.path.join(out, "strategy_ranking.png"))
    if (strats["stops"] == 1).any():
        plot_strategy(strats, best, total_laps, f"{name}: 1-stop options (pit loss {pit_loss:.1f} s)",
                      os.path.join(out, "strategy.png"))
    else:
        print("\nNo 1-stop fits within the tyre life seen in this race "
              f"(max: {', '.join(f'{c.title()} {n}' for c, n in max_age.items())} laps).")
    print(f"\nPit loss used: {pit_loss:.1f} s")
    print("Best plan for each compound sequence:")
    with pd.option_context("display.width", 120):
        print(best[["stops", "plan", "gap_to_best_s"]].head(8).to_string(index=False,
              float_format=lambda v: f"{v:6.1f}"))
    print(f"\nFigures saved to {out}/")


if __name__ == "__main__":
    main()
