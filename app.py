from __future__ import annotations

import pandas as pd
import streamlit as st
import altair as alt

from interface_utils import (
    configure_model_inputs,
    parse_required_float_list,
    parse_required_str_list,
)
from model_inputs import ModelInputs, load_model_inputs
from project_model import MineProjectModel
from sensitivity import run_scenarios
from validation import summarize_validation


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
        financing_mode = st.selectbox(
            "Financing mode",
            ["excel", "synthetic"],
            index=0 if (base_inputs.financing and base_inputs.financing.financing_mode == "excel") else 1,
            help="Use FEM debt schedule or synthetic schedule for new scenarios.",
        )
        interest_shield = st.checkbox(
            "Interest tax shield",
            value=bool(base_inputs.tax_parameters.interest_tax_deductible),
            help="Deduct interest expense from the profit-tax base (Excel financing recommended).",
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

    try:
        configured_inputs = configure_model_inputs(
            adjusted_inputs,
            financing_mode=financing_mode,
            interest_tax_deductible=interest_shield,
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    model = MineProjectModel(configured_inputs, taxes_mode=tax_mode)
    tax_detail = model.calc_taxes_detailed()
    unlevered_cf = model.calc_unlevered_cashflow()
    levered_cf = model.calc_levered_cashflow() if use_financing else None
    revenue_series = model.calc_revenue()
    operating_costs = model.calc_costs()

    scenario_df: pd.DataFrame | None = None
    scenario_tabs = st.tabs(["KPIs", "Scenarios", "Validation"])

    with scenario_tabs[0]:
        st.subheader("Base-case KPIs")
        info_cols = st.columns(4)
        info_cols[0].metric("Tax mode", tax_mode)
        info_cols[1].metric("Power mode", power_mode)
        info_cols[2].metric("Financing mode", financing_mode)
        info_cols[3].metric("Interest shield", "On" if interest_shield else "Off")

        kpi_table = _format_kpis(model, discount_rate, include_levered=use_financing)
        st.table(kpi_table)

        st.markdown("### Cash flow profile")
        cashflow_df = pd.DataFrame({"period": unlevered_cf.index, "Unlevered CF": unlevered_cf.values})
        if levered_cf is not None:
            cashflow_df["Levered CF"] = levered_cf.values
        cashflow_long = cashflow_df.melt(id_vars="period", var_name="series", value_name="thousand_rub")
        cf_chart = (
            alt.Chart(cashflow_long)
            .mark_line(point=True)
            .encode(x="period:Q", y="thousand_rub:Q", color="series:N")
            .properties(height=300)
        )
        st.altair_chart(cf_chart, use_container_width=True)

        st.markdown("### Revenue vs costs and taxes")
        revenue_cost_df = pd.DataFrame(
            {
                "period": revenue_series.index,
                "Revenue": revenue_series.values,
                "Operating costs": operating_costs.values,
                "Total tax": tax_detail["total_tax"].values,
            }
        )
        revenue_cost_long = revenue_cost_df.melt(id_vars="period", var_name="category", value_name="thousand_rub")
        rev_chart = (
            alt.Chart(revenue_cost_long)
            .mark_line()
            .encode(x="period:Q", y="thousand_rub:Q", color="category:N")
            .properties(height=300)
        )
        st.altair_chart(rev_chart, use_container_width=True)

    with scenario_tabs[1]:
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
            try:
                scenario_df = run_scenarios(
                    base_inputs=configured_inputs,
                    price_multipliers=price_multipliers or [1.0],
                    opex_multipliers=opex_multipliers,
                    power_modes=power_modes_list,
                    use_levered=use_financing,
                    discount_rate_override=discount_rate,
                    tax_mode=tax_mode,
                    financing_mode=financing_mode,
                    interest_tax_deductible=interest_shield,
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                payback_cols = ["unlev_pb", "unlev_dpb", "lev_pb", "lev_dpb"]
                for column in payback_cols:
                    if column in scenario_df.columns:
                        scenario_df[column] = pd.to_numeric(scenario_df[column], errors="coerce")
                st.dataframe(scenario_df, width="stretch")

                st.markdown("### NPV vs price multiplier")
                npv_line_df = (
                    scenario_df.groupby("price_mult", as_index=False)["unlev_npv"]
                    .mean()
                    .rename(columns={"unlev_npv": "NPV"})
                )
                line_chart = (
                    alt.Chart(npv_line_df)
                    .mark_line(point=True)
                    .encode(x="price_mult:Q", y="NPV:Q")
                )
                st.altair_chart(line_chart, use_container_width=True)

                st.markdown("### Price vs OPEX multiplier (NPV heatmap)")
                scatter = (
                    alt.Chart(scenario_df)
                    .mark_circle(size=120)
                    .encode(
                        x="price_mult:Q",
                        y="opex_mult:Q",
                        color="unlev_npv:Q",
                        tooltip=["price_mult", "opex_mult", "unlev_npv"],
                    )
                    .properties(height=300)
                )
                st.altair_chart(scatter, use_container_width=True)

    with scenario_tabs[2]:
        st.subheader("Validation & Excel reconciliation")
        st.write(
            "Run the validation report to compare Python outputs against FEM.xlsx for the current configuration."
        )
        if "validation_cache" not in st.session_state:
            st.session_state.validation_cache = None

        if st.button("Run validation report", type="primary"):
            summary_df, recon_df = summarize_validation(model)
            st.session_state.validation_cache = (summary_df, recon_df)

        cache = st.session_state.validation_cache
        if cache is None:
            st.info("Press the button above to generate the validation report.")
        else:
            summary_df, recon_df = cache
            st.markdown("#### Summary deviations")
            summary_sorted = summary_df.sort_values("max_abs_diff", ascending=False)
            styled_summary = summary_sorted.style.format(
                {
                    "max_abs_diff": "{:,.2f}",
                    "excel_at_max_abs": "{:,.2f}",
                    "python_at_max_abs": "{:,.2f}",
                    "max_pct_diff": lambda v: f"{v*100:,.2f}%" if pd.notna(v) else "n/a",
                }
            )
            st.dataframe(styled_summary, use_container_width=True)

            st.markdown("#### Detailed reconciliation")
            metric_options = ["All"] + sorted(recon_df["metric"].unique())
            metric_choice = st.selectbox("Metric filter", metric_options)
            filtered_recon = recon_df.copy()
            if metric_choice != "All":
                filtered_recon = filtered_recon[filtered_recon["metric"] == metric_choice]
            filtered_recon = filtered_recon.sort_values("diff_abs", ascending=False)
            st.dataframe(filtered_recon, use_container_width=True)

            residual_cols = [col for col in tax_detail.columns if col.endswith("_residual_vs_excel")]
            if residual_cols:
                st.markdown("#### Tax residual diagnostics (Python − Excel)")
                residual_df = tax_detail[residual_cols]
                st.dataframe(residual_df, use_container_width=True)


if __name__ == "__main__":
    main()
