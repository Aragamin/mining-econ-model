from __future__ import annotations

import argparse

import pandas as pd

from interface_utils import parse_optional_float_list, parse_optional_str_list
from model_inputs import ModelInputs, load_model_inputs
from project_model import MineProjectModel
from sensitivity import run_scenarios


def _run_base_case(inputs: ModelInputs) -> None:
    """Execute the base-case report with unlevered/levered KPIs and checks."""
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
    core_recon = reconciliation[reconciliation["metric"] != "financing"]
    fin_recon = reconciliation[reconciliation["metric"] == "financing"]

    def _format_row(row: pd.Series, include_pct: bool = True) -> dict[str, float | str | None]:
        return {
            "metric": row["metric"],
            "subcategory": row["subcategory"],
            "period": int(row["period"]),
            "excel_value": round(float(row["excel_value"]), 2),
            "python_value": round(float(row["python_value"]), 2),
            "diff_abs": round(float(row["diff_abs"]), 2),
            "diff_pct": (
                round(float(row["diff_pct"]) * 100.0, 2)
                if include_pct and not pd.isna(row["diff_pct"])
                else None
            ),
        }

    core_abs = core_pct = None
    if not core_recon.empty:
        core_abs = _format_row(core_recon.loc[core_recon["diff_abs"].abs().idxmax()], include_pct=False)
        pct_series = core_recon["diff_pct"].abs().dropna()
        if not pct_series.empty:
            core_pct = _format_row(core_recon.loc[pct_series.idxmax()])

    tax_rows = core_recon[core_recon["metric"] == "tax"]
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
    fin_abs = None
    if not fin_recon.empty:
        fin_abs = _format_row(fin_recon.loc[fin_recon["diff_abs"].abs().idxmax()], include_pct=False)

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

    print("Reconciliation spot checks (core model excluding financing):")
    if core_abs is not None:
        print(f"  Largest absolute deviation: {core_abs}")
    else:
        print("  Largest absolute deviation: n/a (no core deviations)")
    if core_pct is not None:
        print(f"  Largest percentage deviation: {core_pct}")
    else:
        print("  Largest percentage deviation: n/a (no non-zero core deviations)")
    if worst_tax is not None:
        print(f"  Largest tax deviation: {worst_tax}")
        if tax_debug_sample is not None:
            print(f"  Tax debug sample: {tax_debug_sample}")
        else:
            print("  Tax debug sample: n/a (no debug rows for that period)")
    else:
        print("  Largest tax deviation: n/a (no tax rows)")
    # Financing rows are separated because FEM.xlsx has no debt schedule; large differences are expected.
    print("Financing deviations (Python schedule vs Excel financing inputs):")
    if fin_abs is not None:
        print(f"  Largest financing deviation: {fin_abs}")
    else:
        print("  Largest financing deviation: n/a (no financing rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine project model runner.")
    parser.add_argument(
        "--scenario-grid",
        action="store_true",
        help="Run scenario grid instead of the single base case.",
    )
    parser.add_argument(
        "--price-mults",
        help="Comma-separated price multipliers (e.g. 0.9,1.0,1.1).",
    )
    parser.add_argument(
        "--opex-mults",
        help="Optional comma-separated OPEX multipliers (defaults to 1.0).",
    )
    parser.add_argument(
        "--power-modes",
        help="Optional comma-separated power modes (purchase,selfgen).",
    )
    parser.add_argument(
        "--levered",
        action="store_true",
        help="Include levered KPIs when running the scenario grid.",
    )
    parser.add_argument(
        "--discount-rate",
        type=float,
        default=None,
        help="Override discount rate for scenario KPIs (decimal, e.g. 0.15).",
    )

    args = parser.parse_args()

    inputs = load_model_inputs()

    if not args.scenario_grid:
        _run_base_case(inputs)
        return

    try:
        price_mults = parse_optional_float_list(args.price_mults, "price-mults") or [1.0]
        opex_mults = parse_optional_float_list(args.opex_mults, "opex-mults")
    except ValueError as exc:
        parser.error(str(exc))
    power_modes = parse_optional_str_list(args.power_modes)

    scenario_df = run_scenarios(
        base_inputs=inputs,
        price_multipliers=price_mults,
        opex_multipliers=opex_mults,
        power_modes=power_modes,
        use_levered=args.levered,
        discount_rate_override=args.discount_rate,
    )
    if scenario_df.empty:
        print("No scenarios were generated.")
        return
    print("Scenario grid results (NPV in thousand RUB, IRR decimal):")
    print(scenario_df.to_string(index=False))


if __name__ == "__main__":
    main()

