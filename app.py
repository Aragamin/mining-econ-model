from __future__ import annotations

import pandas as pd
import streamlit as st

from interface_utils import parse_required_float_list, parse_required_str_list
from model_inputs import ModelInputs, load_model_inputs
from project_model import MineProjectModel
from sensitivity import run_scenarios


@st.cache_resource(show_spinner=False)
def _load_inputs_cached() -> ModelInputs:
    """Load FEM.xlsx inputs once per Streamlit session."""

    return load_model_inputs()


def _apply_multipliers(
    base_inputs: ModelInputs,
    price_multiplier: float,
    opex_multiplier: float,
    power_mode: str,
) -> ModelInputs:
    """Return a copy of ModelInputs with simple price/OPEX/power overrides."""

    prices = base_inputs.prices.copy()
    price_columns = [
        column
        for column in prices.columns
        if "gold_price" in column or "silver_price" in column
    ]
    for column in price_columns:
        prices[column] = prices[column] * price_multiplier

    opex_breakdown = base_inputs.opex_breakdown.copy() * opex_multiplier
    opex_total = base_inputs.opex.copy()
    opex_total["operating_costs"] = opex_total["operating_costs"] * opex_multiplier

    updated_inputs = base_inputs.copy_with(
        prices=prices,
        opex_breakdown=opex_breakdown,
        opex=opex_total,
        power_mode=power_mode,
    )
    return updated_inputs


def _format_kpis(
    model: MineProjectModel,
    discount_rate: float,
    include_levered: bool,
) -> pd.DataFrame:
    """Compute KPI table for the current model settings."""

    unlev_npv = model.calc_npv(discount_rate)
    unlev_irr = model.calc_irr()
    unlev_pb = model.calc_payback()
    unlev_dpb = model.calc_discounted_payback(discount_rate)

    data = {
        "NPV (thousand RUB)": [round(unlev_npv, 2), ""],
        "IRR": [f"{unlev_irr:.2%}" if pd.notna(unlev_irr) else "n/a", ""],
        "Payback (periods)": [round(unlev_pb, 2) if pd.notna(unlev_pb) else "n/a", ""],
        "Discounted Payback": [
            round(unlev_dpb, 2) if pd.notna(unlev_dpb) else "n/a",
            "",
        ],
    }
    index = ["Unlevered", "Levered"]

    if include_levered:
        lev_npv = model.calc_levered_npv(discount_rate)
        lev_irr = model.calc_levered_irr()
        data["NPV (thousand RUB)"][1] = round(lev_npv, 2)
        data["IRR"][1] = f"{lev_irr:.2%}" if pd.notna(lev_irr) else "n/a"
        data["Payback (periods)"][1] = "n/a"
        data["Discounted Payback"][1] = "n/a"
    else:
        data["NPV (thousand RUB)"][1] = "—"
        data["IRR"][1] = "—"
        data["Payback (periods)"][1] = "—"
        data["Discounted Payback"][1] = "—"

    return pd.DataFrame(data, index=index)


def main() -> None:
    """Streamlit entry point for interactive mining-econ-model exploration."""

    st.set_page_config(page_title="Mining Econ Model UI", layout="wide")
    st.title("Mining Economics Model")
    st.caption("Interactively explore tax, power, financing, and pricing scenarios.")

    base_inputs = _load_inputs_cached()

    with st.sidebar:
        st.header("Controls")
        tax_mode = st.selectbox(
            "Tax mode",
            ["basic", "standard_loss_pool", "excel_cumulative"],
            index=1,
            help="Choose profit-tax calculation logic.",
        )
        power_mode = st.selectbox(
            "Power mode",
            ["purchase", "selfgen"],
            index=0 if base_inputs.power_mode == "purchase" else 1,
            help="Override power-supply scenario used by the model.",
        )
        use_financing = st.checkbox(
            "Use financing (levered KPIs)",
            value=True,
            help="Show levered KPIs derived from the financing schedule.",
        )
        price_multiplier = st.slider(
            "Price multiplier",
            min_value=0.5,
            max_value=1.5,
            value=1.0,
            step=0.01,
            help="Scales all metal-price inputs.",
        )
        opex_multiplier = st.slider(
            "OPEX multiplier",
            min_value=0.5,
            max_value=1.5,
            value=1.0,
            step=0.01,
            help="Scales total operating costs and their breakdown.",
        )
        discount_rate = st.number_input(
            "Discount rate (decimal)",
            min_value=0.0,
            value=float(base_inputs.discount_rate),
            step=0.01,
            format="%.4f",
        )

    adjusted_inputs = _apply_multipliers(
        base_inputs=base_inputs,
        price_multiplier=price_multiplier,
        opex_multiplier=opex_multiplier,
        power_mode=power_mode,
    )

    model = MineProjectModel(adjusted_inputs, taxes_mode=tax_mode)

    st.subheader("Base-case KPIs")
    info_cols = st.columns(3)
    info_cols[0].metric("Tax mode", tax_mode)
    info_cols[1].metric("Power mode", power_mode)
    info_cols[2].metric("Financing enabled", "Yes" if use_financing else "No")

    kpi_table = _format_kpis(model, discount_rate, include_levered=use_financing)
    st.table(kpi_table)

    st.markdown("---")
    st.subheader("Scenario grid")

    scenario_cols = st.columns(3)
    price_mult_input = scenario_cols[0].text_input(
        "Price multipliers",
        value="0.9, 1.0, 1.1",
        help="Comma-separated list (e.g., 0.9,1.0,1.1).",
    )
    opex_mult_input = scenario_cols[1].text_input(
        "OPEX multipliers (optional)",
        value="",
        help="Optional comma-separated list; leave blank to reuse sidebar multiplier.",
    )
    power_mode_input = scenario_cols[2].text_input(
        "Power modes (optional)",
        value="",
        help="Optional comma-separated list of power modes (purchase,selfgen).",
    )

    scenario_error = None
    price_multipliers: list[float] | None = None
    opex_multipliers: list[float] | None = None
    power_modes_list: list[str] | None = None

    try:
        price_multipliers = parse_required_float_list(price_mult_input, "price multipliers")
    except ValueError as exc:
        scenario_error = str(exc)

    if not scenario_error and opex_mult_input.strip():
        try:
            opex_multipliers = parse_required_float_list(opex_mult_input, "OPEX multipliers")
        except ValueError as exc:
            scenario_error = str(exc)

    if not scenario_error and power_mode_input.strip():
        try:
            power_modes_list = parse_required_str_list(power_mode_input, "power modes")
        except ValueError as exc:
            scenario_error = str(exc)

    if scenario_error:
        st.error(scenario_error)
    else:
        scenario_df = run_scenarios(
            base_inputs=adjusted_inputs,
            price_multipliers=price_multipliers or [1.0],
            opex_multipliers=opex_multipliers,
            power_modes=power_modes_list,
            use_levered=use_financing,
            discount_rate_override=discount_rate,
        )
        payback_cols = ["unlev_pb", "unlev_dpb", "lev_pb", "lev_dpb"]
        for column in payback_cols:
            if column in scenario_df.columns:
                scenario_df[column] = pd.to_numeric(scenario_df[column], errors="coerce")
        st.dataframe(scenario_df, width="stretch")

    st.markdown("---")
    with st.expander("Reconciliation vs FEM.xlsx", expanded=False):
        recon = model.check_against_excel()
        core_recon = recon[recon["metric"] != "financing"]
        fin_recon = recon[recon["metric"] == "financing"]

        def fmt_row(row: pd.Series, include_pct: bool = True) -> dict[str, float | str | None]:
            return {
                "metric": row["metric"],
                "subcategory": row["subcategory"],
                "period": int(row["period"]),
                "excel_value": row["excel_value"],
                "python_value": row["python_value"],
                "diff_abs": row["diff_abs"],
                "diff_pct": row["diff_pct"] if include_pct else None,
            }

        summary_core: list[dict[str, float | str | None]] = []
        if not core_recon.empty:
            summary_core.append(
                {"type": "Largest abs deviation", **fmt_row(core_recon.loc[core_recon["diff_abs"].abs().idxmax()], include_pct=False)}
            )
            pct_series = core_recon["diff_pct"].abs().dropna()
            if not pct_series.empty:
                summary_core.append(
                    {"type": "Largest \%\ deviation", **fmt_row(core_recon.loc[pct_series.idxmax()])}
                )
            tax_rows = core_recon[core_recon["metric"] == "tax"]
            if not tax_rows.empty:
                summary_core.append(
                    {"type": "Largest tax deviation", **fmt_row(tax_rows.loc[tax_rows["diff_abs"].idxmax()])}
                )

        summary_fin: list[dict[str, float | str | None]] = []
        if not fin_recon.empty:
            summary_fin.append(
                {
                    "type": "Largest financing deviation",
                    **fmt_row(fin_recon.loc[fin_recon["diff_abs"].abs().idxmax()], include_pct=False),
                }
            )

        st.caption("Core deviations (excluding financing)")
        if summary_core:
            st.table(pd.DataFrame(summary_core))
        else:
            st.write("No core deviations available.")

        st.caption("Financing deviations (expected to differ from FEM.xlsx)")
        if summary_fin:
            st.table(pd.DataFrame(summary_fin))
        else:
            st.write("No financing deviations available.")

        if st.checkbox("Show full reconciliation table"):
            st.dataframe(recon, width="stretch")


if __name__ == "__main__":
    main()
