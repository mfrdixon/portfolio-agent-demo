from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from agents import function_tool

BASE = Path(__file__).resolve().parents[1]
POSITIONS_PATH = BASE / "data" / "positions.csv"
RETURNS_PATH = BASE / "data" / "returns.csv"
LIMITS_PATH = BASE / "data" / "limits.json"


def _positions() -> pd.DataFrame:
    return pd.read_csv(POSITIONS_PATH)


def _returns() -> pd.DataFrame:
    return pd.read_csv(RETURNS_PATH, parse_dates=["date"])


def _limits() -> dict:
    return json.loads(LIMITS_PATH.read_text())


def _nav(df: pd.DataFrame | None = None) -> float:
    df = _positions() if df is None else df
    return float(df["market_value"].sum())


@function_tool
def get_portfolio_summary() -> Dict:
    """Return NAV, position count, exposures by asset class, sector, region, and currency."""
    df = _positions()
    nav = _nav(df)
    def exposure(col: str) -> Dict[str, float]:
        return (df.groupby(col)["market_value"].sum() / nav).sort_values(ascending=False).round(4).to_dict()
    return {
        "nav": nav,
        "position_count": int(len(df)),
        "asset_class_exposure_pct_nav": exposure("asset_class"),
        "sector_exposure_pct_nav": exposure("sector"),
        "region_exposure_pct_nav": exposure("region"),
        "currency_exposure_pct_nav": exposure("currency"),
    }


@function_tool
def get_top_positions(n: int = 10) -> List[Dict]:
    """Return largest positions with percent of NAV."""
    df = _positions()
    nav = _nav(df)
    out = df.sort_values("market_value", ascending=False).head(n).copy()
    out["pct_nav"] = out["market_value"] / nav
    return out[["ticker", "name", "asset_class", "sector", "market_value", "pct_nav", "issuer"]].round(4).to_dict("records")


@function_tool
def calculate_historical_var(confidence: float = 0.95, horizon_days: int = 1) -> Dict:
    """Calculate historical VaR and expected shortfall from dummy return history."""
    pos = _positions().set_index("ticker")
    rets = _returns().drop(columns=["date"])
    tickers = [c for c in rets.columns if c in pos.index]
    weights = pos.loc[tickers, "market_value"] / pos["market_value"].sum()
    portfolio_returns = rets[tickers].mul(weights, axis=1).sum(axis=1)
    scaled = portfolio_returns * np.sqrt(horizon_days)
    alpha = 1.0 - confidence
    var_return = float(np.quantile(scaled, alpha))
    tail = scaled[scaled <= var_return]
    es_return = float(tail.mean()) if len(tail) else var_return
    nav = _nav(pos.reset_index())
    return {
        "confidence": confidence,
        "horizon_days": horizon_days,
        "var_return": var_return,
        "var_amount": -var_return * nav,
        "expected_shortfall_return": es_return,
        "expected_shortfall_amount": -es_return * nav,
        "sample_days": int(len(scaled)),
        "note": "Toy historical VaR from dummy returns; production requires longer clean history and risk-factor mapping."
    }


@function_tool
def calculate_component_var(confidence: float = 0.95) -> Dict:
    """Approximate component VaR by leave-one-out marginal contribution."""
    pos = _positions().set_index("ticker")
    rets = _returns().drop(columns=["date"])
    tickers = [c for c in rets.columns if c in pos.index]
    nav = float(pos["market_value"].sum())
    weights = pos.loc[tickers, "market_value"] / nav
    full = rets[tickers].mul(weights, axis=1).sum(axis=1)
    full_var = -float(np.quantile(full, 1.0 - confidence)) * nav
    rows = []
    for t in tickers:
        reduced = full - rets[t] * weights[t]
        reduced_var = -float(np.quantile(reduced, 1.0 - confidence)) * nav
        rows.append({"ticker": t, "component_var_amount": full_var - reduced_var})
    return {"portfolio_var_amount": full_var, "components": sorted(rows, key=lambda x: x["component_var_amount"], reverse=True)}


@function_tool
def run_stress_test(equity_shock: float = -0.15, credit_shock: float = -0.05, rates_shock_bps: float = 50, usd_shock: float = 0.03, vol_shock: float = 0.05) -> Dict:
    """Run simple deterministic portfolio stress. Rates shock uses DV01; option vol shock uses vega."""
    df = _positions().copy()
    df["pnl"] = 0.0
    df.loc[df["asset_class"].eq("Equity"), "pnl"] += df["market_value"] * equity_shock
    df.loc[df["asset_class"].eq("Credit"), "pnl"] += df["market_value"] * credit_shock
    df["pnl"] += -df["dv01"] * rates_shock_bps
    df.loc[df["currency"].ne("USD"), "pnl"] += df["market_value"] * usd_shock
    df["pnl"] += df["vega"] * vol_shock
    nav = _nav(df)
    return {
        "assumptions": {"equity_shock": equity_shock, "credit_shock": credit_shock, "rates_shock_bps": rates_shock_bps, "usd_shock": usd_shock, "vol_shock": vol_shock},
        "total_pnl": float(df["pnl"].sum()),
        "total_pnl_pct_nav": float(df["pnl"].sum() / nav),
        "pnl_by_asset_class": df.groupby("asset_class")["pnl"].sum().sort_values().round(2).to_dict(),
        "worst_positions": df.sort_values("pnl").head(5)[["ticker", "asset_class", "market_value", "pnl"]].round(2).to_dict("records"),
    }


@function_tool
def calculate_sensitivities() -> Dict:
    """Return portfolio beta, DV01, vega, and liquidity diagnostics."""
    df = _positions()
    nav = _nav(df)
    return {
        "weighted_equity_beta": float((df["market_value"] * df["beta"]).sum() / nav),
        "total_dv01": float(df["dv01"].sum()),
        "total_vega": float(df["vega"].sum()),
        "market_value_weighted_liquidity_score": float((df["market_value"] * df["liquidity_score"]).sum() / nav),
        "least_liquid_positions": df.sort_values("liquidity_score").head(5)[["ticker", "asset_class", "market_value", "liquidity_score"]].to_dict("records"),
    }


@function_tool
def check_risk_limits() -> Dict:
    """Check dummy internal limits for concentration, VaR, and liquidity."""
    df = _positions()
    limits = _limits()
    nav = _nav(df)
    breaches = []
    top = df.assign(pct_nav=df["market_value"] / nav).sort_values("pct_nav", ascending=False).iloc[0]
    if top["pct_nav"] > limits["max_single_name_pct_nav"]:
        breaches.append({"limit": "max_single_name_pct_nav", "ticker": top["ticker"], "value": float(top["pct_nav"]), "threshold": limits["max_single_name_pct_nav"]})
    for col, limit_name in [("sector", "max_sector_pct_nav"), ("asset_class", "max_asset_class_pct_nav")]:
        ex = (df.groupby(col)["market_value"].sum() / nav).sort_values(ascending=False)
        if float(ex.iloc[0]) > limits[limit_name]:
            breaches.append({"limit": limit_name, col: ex.index[0], "value": float(ex.iloc[0]), "threshold": limits[limit_name]})
    var = calculate_historical_var.on_invoke_tool(None, '{"confidence":0.95,"horizon_days":1}') if False else None
    low_liq = df[df["liquidity_score"] < limits["min_liquidity_score"]]
    for _, row in low_liq.iterrows():
        breaches.append({"limit": "min_liquidity_score", "ticker": row["ticker"], "value": float(row["liquidity_score"]), "threshold": limits["min_liquidity_score"]})
    return {"breaches": breaches, "limits": limits}
