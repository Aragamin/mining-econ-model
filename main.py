from __future__ import annotations

import pandas as pd

from model_inputs import load_model_inputs
from project_model import MineProjectModel


def main() -> None:
    inputs = load_model_inputs()
    model = MineProjectModel(inputs)

    unlevered_cf = model.calc_unlevered_cashflow()
    npv = model.calc_npv()
    irr = model.calc_irr()
    payback = model.calc_payback()
    discounted_payback = model.calc_discounted_payback()
    levered_npv = model.calc_levered_npv(inputs.discount_rate)
    levered_irr = model.calc_levered_irr()

    print("Mine project metrics (base case):")
    print(f"  Scenario: {inputs.scenario}")
    print(f"  Discount rate: {inputs.discount_rate:.2%}")
    print(f"  Profit tax rate: {inputs.profit_tax_rate:.2%}")
    print()
    print("  NPV: {:,.0f} thousand RUB".format(npv))
    print("  IRR: {:.2%}".format(irr))
    print("  Payback (periods): {:.2f}".format(payback))
    print("  Discounted Payback (periods): {:.2f}".format(discounted_payback))
    print("  Levered NPV: {:,.0f} thousand RUB".format(levered_npv))
    print("  Levered IRR: {:.2%}".format(levered_irr))
    print()
    print("Unlevered cash flow by period (thousand RUB):")
    print(unlevered_cf.round(2))
    print()

    reconciliation = model.check_against_excel()
    worst_abs_row = reconciliation.loc[reconciliation["diff_abs"].idxmax()]
    worst_abs = {
        "metric": worst_abs_row["metric"],
        "subcategory": worst_abs_row["subcategory"],
        "period": int(worst_abs_row["period"]),
        "excel_value": round(float(worst_abs_row["excel_value"]), 2),
        "python_value": round(float(worst_abs_row["python_value"]), 2),
        "diff_abs": round(float(worst_abs_row["diff_abs"]), 2),
    }

    pct_series = reconciliation["diff_pct"].abs().dropna()
    worst_pct = None
    if not pct_series.empty:
        worst_pct_row = reconciliation.loc[pct_series.idxmax()]
        worst_pct = {
            "metric": worst_pct_row["metric"],
            "subcategory": worst_pct_row["subcategory"],
            "period": int(worst_pct_row["period"]),
            "excel_value": round(float(worst_pct_row["excel_value"]), 2),
            "python_value": round(float(worst_pct_row["python_value"]), 2),
            "diff_pct": round(float(worst_pct_row["diff_pct"]) * 100.0, 2),
        }

    tax_rows = reconciliation[reconciliation["metric"] == "tax"]
    worst_tax = None
    if not tax_rows.empty:
        worst_tax_row = tax_rows.loc[tax_rows["diff_abs"].idxmax()]
        worst_tax = {
            "subcategory": worst_tax_row["subcategory"],
            "period": int(worst_tax_row["period"]),
            "excel_value": round(float(worst_tax_row["excel_value"]), 2),
            "python_value": round(float(worst_tax_row["python_value"]), 2),
            "diff_abs": round(float(worst_tax_row["diff_abs"]), 2),
            "diff_pct": (
                round(float(worst_tax_row["diff_pct"]) * 100.0, 2)
                if not pd.isna(worst_tax_row["diff_pct"])
                else None
            ),
        }
    tax_debug_sample = None
    if worst_tax is not None:
        tax_period = worst_tax["period"]
        tax_debug_rows = reconciliation[
            (reconciliation["metric"] == "tax_debug") & (reconciliation["period"] == tax_period)
        ]
        if not tax_debug_rows.empty:
            preferred = tax_debug_rows[tax_debug_rows["subcategory"] == "taxable_profit"]
            sample_row = preferred.iloc[0] if not preferred.empty else tax_debug_rows.iloc[0]
            tax_debug_sample = {
                "subcategory": sample_row["subcategory"],
                "period": int(sample_row["period"]),
                "excel_value": (
                    round(float(sample_row["excel_value"]), 2)
                    if not pd.isna(sample_row["excel_value"])
                    else None
                ),
                "python_value": round(float(sample_row["python_value"]), 2),
                "diff_abs": round(float(sample_row["diff_abs"]), 2),
                "diff_pct": (
                    round(float(sample_row["diff_pct"]) * 100.0, 2)
                    if not pd.isna(sample_row["diff_pct"])
                    else None
                ),
            }

    print("Reconciliation spot checks:")
    print(f"  Largest absolute deviation: {worst_abs}")
    if worst_pct is not None:
        print(f"  Largest percentage deviation: {worst_pct}")
    else:
        print("  Largest percentage deviation: n/a (no non-zero Excel values)")
    if worst_tax is not None:
        print(f"  Largest tax deviation: {worst_tax}")
        if tax_debug_sample is not None:
            print(f"  Tax debug sample: {tax_debug_sample}")
        else:
            print("  Tax debug sample: n/a (no debug rows for that period)")
    else:
        print("  Largest tax deviation: n/a (no tax rows)")


if __name__ == "__main__":
    main()

