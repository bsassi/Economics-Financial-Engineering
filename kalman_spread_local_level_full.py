#!/usr/bin/env python3
"""
Kalman Filter on Credit Spread (Local Level) — End-to-end script

What it does
------------
1) Load spread series from 'Spread_data.xlsx' (sheet 'Series').
2) Fit a Local Level state-space model:   y_t = s_t + eps_t,   s_t = s_{t-1} + eta_t
3) Export observed vs latent "fair" spread + innovations and z-scores to CSV.
4) Plot:
   - Observed vs Latent "fair" spread
   - Standardized innovations (z-score)
5) Extremes table (Top-|z|) and mean-reversion diagnostic (Δy vs z for H=1/3/5/10).
   Also plots a scatter for H=5 with fitted line.
6) Simple z-rule backtest (|z|): entry at |z|>=2, exit at |z|<=0.5, costs=0.5 bp per switch, DV01=7.
   Exports backtest path to CSV and plots cumulative P&L vs Buy&Hold long credit.

Notes
-----
- Uses only matplotlib for charts (as requested).
- File paths assume this script runs in the same environment where the Excel file lives.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Avoid seaborn to comply with constraints
import statsmodels.api as sm

# -------------------------
# 0) Paths & settings
# -------------------------
DATA_PATH = Path("Spread_data.xlsx") 
SERIES_SHEET = "Series"

OUT_DIR = Path("/mnt/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUTPUT_MAIN   = OUT_DIR / "kalman_local_level_output.csv"
CSV_OUTPUT_EXTREM = OUT_DIR / "kalman_extremes_top12.csv"
CSV_OUTPUT_DIAG   = OUT_DIR / "kalman_mean_reversion_diag.csv"
CSV_OUTPUT_BT     = OUT_DIR / "kalman_backtest_path.csv"

FIG_OBS_LATENT    = OUT_DIR / "kalman_obs_vs_latent.png"
FIG_ZSCORE        = OUT_DIR / "kalman_zscore.png"
FIG_SCATTER_H5    = OUT_DIR / "kalman_scatter_h5.png"
FIG_CUMPNL        = OUT_DIR / "kalman_cumpnl.png"

TOP_N_EXTREMES = 12

# Backtest parameters
Z_ENTRY = 2.0
Z_EXIT  = 0.5
COST_BP = 0.5   # per switch (entry/exit), in spread bp
DV01    = 7.0   # price-bp per 1bp spread change


def load_series(path: Path, sheet: str = SERIES_SHEET) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    df = pd.read_excel(xls, sheet_name=sheet)
    df.columns = [str(c).strip().replace("\\n", " ") for c in df.columns]
    date_col = [c for c in df.columns if "date" in c.lower()][0]
    value_cols = [c for c in df.columns if c != date_col]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df = df.rename(columns={value_cols[0]: "spread_bp"})
    return df.rename(columns={date_col: "Date"})[["Date", "spread_bp"]]


def fit_local_level(y: pd.Series):
    """
    Fit local level model with statsmodels and return result + derived series.
    """
    mod = sm.tsa.UnobservedComponents(y, level="local level")
    res = mod.fit(disp=False)

    # Smoothed latent level
    latent = pd.Series(res.smoothed_state[0], index=y.index, name="Latent_fair_bp")

    # Innovations & standardized innovations (z)
    fr = res.filter_results
    innov = pd.Series(fr.forecasts_error[0], index=y.index, name="Innovation_bp")
    z = pd.Series(fr.standardized_forecasts_error[0], index=y.index, name="z")

    return res, latent, innov, z


def plot_obs_vs_latent(dates: pd.Series, y: pd.Series, latent: pd.Series, out_path: Path):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, y, label="Observed spread (bp)")
    plt.plot(dates, latent, label="Latent fair spread (bp)")
    plt.legend()
    plt.title("Observed vs Latent 'Fair' Spread (Local Level Kalman)")
    plt.xlabel("Date")
    plt.ylabel("Basis points")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()


def plot_zscore(dates: pd.Series, z: pd.Series, out_path: Path):
    plt.figure(figsize=(10, 4))
    plt.plot(dates, z)
    plt.axhline(0, linestyle="--")
    plt.title("Standardized Innovation (z-score)")
    plt.xlabel("Date")
    plt.ylabel("z")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()


def extremes_table(dates: pd.Series, y: pd.Series, latent: pd.Series, z: pd.Series, top_n: int = TOP_N_EXTREMES) -> pd.DataFrame:
    dev = y - latent
    ext = pd.DataFrame({"Date": dates, "Observed_bp": y, "Latent_fair_bp": latent, "Deviation_bp": dev, "z": z})
    ext["abs_z"] = ext["z"].abs()
    ext_top = ext.sort_values("abs_z", ascending=False).head(top_n)
    return ext_top[["Date", "Observed_bp", "Latent_fair_bp", "Deviation_bp", "z"]]


def mean_reversion_diag(y: pd.Series, z: pd.Series, horizons=(1, 3, 5, 10)) -> pd.DataFrame:
    rows = []
    for h in horizons:
        fwd = y.shift(-h) - y  # Δy_{t→t+h}
        valid = (~z.isna()) & (~fwd.isna())
        zz = z[valid].values
        ff = fwd[valid].values

        # OLS: Δy = a + b z + e
        X = sm.add_constant(zz)
        model = sm.OLS(ff, X).fit()
        a, b = model.params
        t_b = model.tvalues[1]
        r2 = model.rsquared
        hit = (np.sign(ff) == -np.sign(zz)).mean()  # opposite sign = mean reversion
        rows.append({"H": h, "Slope_b_bp_per_z": float(b), "t_stat_b": float(t_b), "R2": float(r2), "HitRate_opposite_sign": float(hit)})
    return pd.DataFrame(rows)


def scatter_h(z: pd.Series, y: pd.Series, h: int, out_path: Path):
    fwd = y.shift(-h) - y
    mask = (~z.isna()) & (~fwd.isna())
    zz = z[mask]
    ff = fwd[mask]

    # Linear fit
    b1, b0 = np.polyfit(zz.values, ff.values, 1)
    grid = np.linspace(zz.min(), zz.max(), 200)

    plt.figure(figsize=(10, 6))
    plt.scatter(zz.values, ff.values, alpha=0.5)
    plt.plot(grid, b1 * grid + b0, linewidth=2, label="OLS fit")
    plt.axhline(0, linestyle="--")
    plt.axvline(0, linestyle="--")
    plt.title(f"Δ spread (bp) in {h} business days vs current z-score")
    plt.xlabel("z-score (today)")
    plt.ylabel(f"Δ spread in {h} days (bp)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()


def zrule_backtest(dates: pd.Series, y: pd.Series, z: pd.Series,
                   z_entry: float = Z_ENTRY, z_exit: float = Z_EXIT,
                   cost_bp: float = COST_BP, dv01: float = DV01) -> pd.DataFrame:
    position = pd.Series(0.0, index=y.index)
    state = 0.0
    for t in range(len(z)):
        zz = z.iloc[t]
        if state == 0.0:
            if zz >= z_entry:
                state = +1.0
            elif zz <= -z_entry:
                state = -1.0
        else:
            if abs(zz) <= z_exit:
                state = 0.0
        position.iloc[t] = state

    dy = y.shift(-1) - y
    pnl_spread = -position * dy  # long credit wins when spreads tighten
    switch = position.diff().abs().fillna(0.0)
    pnl_spread = pnl_spread - (switch * cost_bp)
    pnl_price_bp = pnl_spread * dv01

    bh_pnl_price_bp = (-1.0 * (y.shift(-1) - y)) * dv01  # buy & hold long credit

    out = pd.DataFrame({
        "Date": dates,
        "Spread_bp": y,
        "z": z,
        "Position": position,
        "dSpread_next_bp": dy,
        "PnL_spread_bp": pnl_spread,
        "PnL_price_bp": pnl_price_bp,
        "CumPnL_spread_bp": pnl_spread.cumsum(),
        "CumPnL_price_bp": pnl_price_bp.cumsum(),
        "BH_PnL_price_bp": bh_pnl_price_bp,
        "BH_CumPnL_price_bp": bh_pnl_price_bp.cumsum()
    })
    return out


def plot_cumpnl(bt: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(10, 5))
    plt.plot(bt["Date"], bt["CumPnL_price_bp"], label="Strategy (price bp)")
    plt.plot(bt["Date"], bt["BH_CumPnL_price_bp"], label="Buy & Hold long credit (price bp)")
    plt.legend()
    plt.title("Cumulative P&L (assumed DV01 = 7.0) — aligned")
    plt.xlabel("Date")
    plt.ylabel("Cumulative price bp")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()


def main():
    # 1) Load
    df = load_series(DATA_PATH, SERIES_SHEET)
    dates = df["Date"]
    y = df["spread_bp"].astype(float)

    # 2) Fit local level
    res, latent, innov, z = fit_local_level(y)

    # 3) Export main output
    out = pd.DataFrame({
        "Date": dates,
        "Observed_bp": y.values,
        "Latent_fair_bp": latent.values,
        "Innovation_bp": innov.values,
        "z": z.values
    })
    out.to_csv(CSV_OUTPUT_MAIN, index=False)

    # 4) Plots
    plot_obs_vs_latent(dates, y, latent, FIG_OBS_LATENT)
    plot_zscore(dates, z, FIG_ZSCORE)

    # 5) Extremes + diagnostic + scatter H=5
    ext_top = extremes_table(dates, y, latent, z, top_n=TOP_N_EXTREMES)
    ext_top.to_csv(CSV_OUTPUT_EXTREM, index=False)

    diag = mean_reversion_diag(y, z, horizons=(1, 3, 5, 10))
    diag.to_csv(CSV_OUTPUT_DIAG, index=False)

    scatter_h(z, y, h=5, out_path=FIG_SCATTER_H5)

    # 6) Backtest
    bt = zrule_backtest(dates, y, z, Z_ENTRY, Z_EXIT, COST_BP, DV01)
    bt.to_csv(CSV_OUTPUT_BT, index=False)
    plot_cumpnl(bt, FIG_CUMPNL)

    # Print a tiny summary
    print("Fitted params (local level):", res.params.to_dict())
    print("Files written:")
    print(" -", CSV_OUTPUT_MAIN)
    print(" -", CSV_OUTPUT_EXTREM)
    print(" -", CSV_OUTPUT_DIAG)
    print(" -", CSV_OUTPUT_BT)
    print(" -", FIG_OBS_LATENT)
    print(" -", FIG_ZSCORE)
    print(" -", FIG_SCATTER_H5)
    print(" -", FIG_CUMPNL)


if __name__ == "__main__":
    main()
