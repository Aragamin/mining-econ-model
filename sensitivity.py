from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import pandas as pd
import numpy as np

from model_inputs import ModelInputs
from project_model import MineProjectModel


DEFAULT_PRICE_COLUMNS = ("gold_price_rub_g", "silver_price_rub_g")


def vary_price(
    model_inputs: ModelInputs,
    deltas: Iterable[float],
    rate: float | None = None,
    price_columns: Sequence[str] = DEFAULT_PRICE_COLUMNS,
) -> pd.DataFrame:
    """Evaluate NPV sensitivity to metal price changes."""

    results = []
    discount_rate = rate if rate is not None else model_inputs.discount_rate

    for delta in deltas:
        adjusted_prices = model_inputs.prices.copy()
        for column in price_columns:
            if column in adjusted_prices:
                adjusted_prices[column] = adjusted_prices[column] * (1.0 + delta)
        updated_inputs = model_inputs.copy_with(prices=adjusted_prices)
        model = MineProjectModel(updated_inputs)
        npv = model.calc_npv(discount_rate)
        results.append({"parameter": "metal_price", "delta": delta, "NPV": npv})

    return pd.DataFrame(results)


def run_scenarios(
    base_inputs: ModelInputs,
    price_multipliers: Sequence[float],
    opex_multipliers: Optional[Sequence[float]] = None,
    power_modes: Optional[Sequence[str]] = None,
    use_levered: bool = False,
    discount_rate_override: Optional[float] = None,
) -> pd.DataFrame:
    """Evaluate KPI grid across price/OPEX/power-mode combinations.

    Args:
        base_inputs: Canonical `ModelInputs` object to clone for scenarios.
        price_multipliers: Factors applied to all metal-price columns (decimal).
        opex_multipliers: Optional factors applied to OPEX breakdown and
            operating-cost totals. If ``None``, only the base multiplier of 1.0
            is used.
        power_modes: Optional list of power-mode strings ("purchase", "selfgen").
            When omitted the base scenario's mode is preserved.
        use_levered: When True, include levered NPV/IRR based on
            `calc_levered_cashflow()`.
        discount_rate_override: Optional discount rate applied to both levered
            and unlevered KPIs; falls back to each scenario's `ModelInputs` rate.

    Returns:
        pandas.DataFrame containing scenario descriptors and KPI columns. NPV
        columns are in thousand RUB; IRR columns are decimal values
        (e.g. 0.15 == 15%). Paybacks are reported in project periods.
    """

    if not price_multipliers:
        raise ValueError("price_multipliers must contain at least one value.")

    effective_opex = list(opex_multipliers) if opex_multipliers else [1.0]
    base_power_mode = base_inputs.power_mode or (
        base_inputs.power_inputs.base_mode if base_inputs.power_inputs else "purchase"
    )
    effective_power_modes = list(power_modes) if power_modes else [base_power_mode]

    results: List[dict[str, float | str | None]] = []
    base_power_mode = base_inputs.power_mode

    price_columns = [
        column
        for column in base_inputs.prices.columns
        if "gold_price" in column or "silver_price" in column
    ]

    for price_mult in price_multipliers:
        for opex_mult in effective_opex:
            for power_mode in effective_power_modes:
                scenario_prices = base_inputs.prices.copy()
                scenario_prices.loc[:, price_columns] = (
                    scenario_prices.loc[:, price_columns] * price_mult
                )

                scenario_opex = base_inputs.opex.copy()
                scenario_opex_breakdown = base_inputs.opex_breakdown.copy()
                if opex_mult != 1.0:
                    scenario_opex_breakdown = scenario_opex_breakdown * opex_mult
                    scenario_opex["operating_costs"] = (
                        scenario_opex["operating_costs"] * opex_mult
                    )

                powered_mode = power_mode or base_power_mode
                updated_inputs = base_inputs.copy_with(
                    prices=scenario_prices,
                    opex=scenario_opex,
                    opex_breakdown=scenario_opex_breakdown,
                    power_mode=powered_mode,
                )

                model = MineProjectModel(updated_inputs)
                discount_rate = (
                    discount_rate_override
                    if discount_rate_override is not None
                    else updated_inputs.discount_rate
                )

                unlev_npv = model.calc_npv(discount_rate)
                unlev_irr = model.calc_irr()
                unlev_pb = model.calc_payback()
                unlev_dpb = model.calc_discounted_payback(discount_rate)
                if use_levered:
                    lev_npv = model.calc_levered_npv(discount_rate)
                    lev_irr = model.calc_levered_irr()
                else:
                    lev_npv = float("nan")
                    lev_irr = float("nan")

                unlev_pb_value = (
                    float(unlev_pb) if unlev_pb is not None and not pd.isna(unlev_pb) else np.nan
                )
                unlev_dpb_value = (
                    float(unlev_dpb) if unlev_dpb is not None and not pd.isna(unlev_dpb) else np.nan
                )

                results.append(
                    {
                        "price_mult": price_mult,
                        "opex_mult": opex_mult,
                        "power_mode": powered_mode,
                        "unlev_npv": unlev_npv,
                        "unlev_irr": unlev_irr,
                        "unlev_pb": unlev_pb_value,
                        "unlev_dpb": unlev_dpb_value,
                        "lev_npv": lev_npv,
                        "lev_irr": lev_irr,
                    }
                )

    return pd.DataFrame(results)
