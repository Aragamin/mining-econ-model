from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import pandas as pd

from project_model import MineProjectModel


ValidationTarget = Tuple[str, Optional[str], str]

DEFAULT_VALIDATION_TARGETS: Sequence[ValidationTarget] = (
    ("revenue", None, "Revenue"),
    ("core_opex", None, "Core OPEX"),
    ("tax", "mineral_extraction_tax", "NDPI"),
    ("tax", "property_tax", "Property tax"),
    ("tax", "profit_tax", "Profit tax"),
    ("tax", "total_tax", "Total tax"),
    ("unlevered_cashflow", None, "Unlevered CF"),
    ("levered_cashflow", None, "Levered CF"),
    ("financing", "debt_balance", "Debt balance"),
)


def build_validation_table(model: MineProjectModel) -> pd.DataFrame:
    """Return the full reconciliation table for downstream validation."""

    return model.check_against_excel()


def summarize_validation(
    model: MineProjectModel,
    targets: Sequence[ValidationTarget] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarize FEM vs Python deviations for a curated list of metrics.

    Returns:
        (summary_df, raw_table)
    """

    recon = build_validation_table(model)
    selected = targets or DEFAULT_VALIDATION_TARGETS
    rows: List[dict[str, float | str | None]] = []

    for metric, subcategory, label in selected:
        subset = recon[recon["metric"] == metric]
        if subcategory is not None:
            subset = subset[subset["subcategory"] == subcategory]
        if subset.empty:
            continue

        abs_idx = subset["diff_abs"].abs().idxmax()
        abs_row = subset.loc[abs_idx]
        pct_series = subset.loc[subset["excel_value"].abs() > 1e-6, "diff_pct"].abs().dropna()
        if not pct_series.empty:
            pct_idx = pct_series.idxmax()
            pct_row = subset.loc[pct_idx]
            max_pct = float(pct_row["diff_pct"])
            pct_period = int(pct_row["period"])
            pct_excel = float(pct_row["excel_value"])
            pct_python = float(pct_row["python_value"])
        else:
            max_pct = None
            pct_period = None
            pct_excel = None
            pct_python = None

        rows.append(
            {
                "label": label,
                "metric": metric,
                "subcategory": subcategory,
                "max_abs_diff": float(abs_row["diff_abs"]),
                "max_abs_period": int(abs_row["period"]),
                "excel_at_max_abs": float(abs_row["excel_value"]),
                "python_at_max_abs": float(abs_row["python_value"]),
                "max_pct_diff": max_pct,
                "max_pct_period": pct_period,
                "excel_at_max_pct": pct_excel,
                "python_at_max_pct": pct_python,
            }
        )

    summary_df = pd.DataFrame(rows)
    return summary_df, recon
