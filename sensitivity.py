from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

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

