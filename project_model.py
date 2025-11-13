from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import numpy_financial as npf
import pandas as pd

from model_inputs import ModelInputs
from tax_engine import TaxEngine


class MineProjectModel:
    """Core project cash-flow model for the mining project."""

    TAX_MODES = {"basic", "detailed", "standard_loss_pool", "excel_cumulative"}

    def __init__(self, inputs: ModelInputs, taxes_mode: Optional[str] = None) -> None:
        """Initialize the model with structured inputs."""

        default_mode = inputs.tax_parameters.loss_carry_mode or "standard_loss_pool"
        resolved_mode = taxes_mode or default_mode
        if resolved_mode == "detailed":
            resolved_mode = "standard_loss_pool"

        if resolved_mode not in self.TAX_MODES:
            raise ValueError(
                "taxes_mode must be one of {'basic', 'detailed', "
                "'standard_loss_pool', 'excel_cumulative'}."
            )

        self.inputs = inputs
        self.period_index = inputs.timeline.index
        self._revenue_detail: Optional[pd.DataFrame] = None
        self._working_capital_detail: Optional[pd.DataFrame] = None
        self._ndpi_detail: Optional[pd.DataFrame] = None
        self._property_tax_detail: Optional[pd.DataFrame] = None
        self._tax_detail: Dict[str, pd.DataFrame] = {}
        self._financing_schedule: Optional[pd.DataFrame] = None
        self.base_opex_breakdown = inputs.opex_breakdown.copy()
        self.power_inputs = inputs.power_inputs
        self.power_mode = inputs.power_mode or (
            self.power_inputs.base_mode if self.power_inputs else ""
        )
        self.opex_breakdown = self._apply_power_mode_to_opex()
        self.operating_costs_total = self._build_operating_costs_total()
        self.taxes_mode = resolved_mode

    def _apply_power_mode_to_opex(self) -> pd.DataFrame:
        """Return OPEX breakdown adjusted for the requested power scenario."""

        breakdown = self.base_opex_breakdown.copy()
        if not self.power_inputs:
            return breakdown

        target_mode = self.power_mode or self.power_inputs.base_mode
        if target_mode == self.power_inputs.base_mode:
            return breakdown

        replacements = self.power_inputs.opex_for_mode(target_mode)
        for column, series in replacements.items():
            if series is None:
                continue
            breakdown[column] = series.reindex(self.period_index).fillna(0.0)
        return breakdown

    def _build_operating_costs_total(
        self, adjusted_breakdown: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """Reconcile operating-cost totals with any scenario overrides."""

        breakdown = adjusted_breakdown if adjusted_breakdown is not None else self.opex_breakdown
        base_total = (
            self.inputs.opex["operating_costs"]
            .reindex(self.period_index)
            .fillna(0.0)
        )
        base_breakdown_sum = (
            self.base_opex_breakdown.sum(axis=1)
            .reindex(self.period_index)
            .fillna(0.0)
        )
        new_sum = breakdown.sum(axis=1).reindex(self.period_index).fillna(0.0)
        delta = new_sum - base_breakdown_sum
        return base_total.add(delta, fill_value=0.0)

    def _power_capex_components(self) -> Dict[str, pd.Series]:
        """Return CAPEX series aligned to each power mode."""

        zero = pd.Series(0.0, index=self.period_index, name="power_capex_placeholder")
        if not self.power_inputs:
            return {"purchase": zero, "selfgen": zero}

        purchase = (
            self.power_inputs.capex_purchase_power.reindex(self.period_index).fillna(0.0)
            if self.power_inputs.capex_purchase_power is not None
            else zero.copy()
        )
        selfgen = (
            self.power_inputs.capex_selfgen_power.reindex(self.period_index).fillna(0.0)
            if self.power_inputs.capex_selfgen_power is not None
            else zero.copy()
        )
        return {"purchase": purchase, "selfgen": selfgen}

    def calc_revenue_detail(self) -> pd.DataFrame:
        """
        Return per-metal production and revenue detail with explicit units.

        Columns include grams produced and RUB revenue for gold/silver plus the
        aggregated thousand-RUB revenue used elsewhere in the model.
        """

        if self._revenue_detail is not None:
            return self._revenue_detail.copy()

        production = self.inputs.production.reindex(self.period_index).fillna(0.0)
        prices = self.inputs.prices.reindex(self.period_index).fillna(0.0)

        ore_tonnes = production["ore_mined_kton"] * 1_000.0
        detail: Dict[str, pd.Series] = {}

        for metal in ("gold", "silver"):
            grade = production[f"{metal}_grade_gpt"]
            recovery = production[f"{metal}_recovery"]
            grams = ore_tonnes * grade * recovery
            price_rub = prices[f"{metal}_price_rub_g"]
            revenue_rub = grams * price_rub
            detail[f"{metal}_grams"] = grams
            detail[f"{metal}_revenue_rub"] = revenue_rub

        detail_df = pd.DataFrame(detail, index=self.period_index)
        detail_df["total_revenue_rub"] = detail_df["gold_revenue_rub"] + detail_df["silver_revenue_rub"]
        detail_df["total_revenue_thousand"] = detail_df["total_revenue_rub"] / 1_000.0

        self._revenue_detail = detail_df
        return detail_df.copy()

    def _calc_ndpi_detail(self) -> pd.DataFrame:
        """Return NDPI per metal based on revenue detail and statutory rates."""

        if self._ndpi_detail is not None:
            return self._ndpi_detail.copy()

        detail = self.calc_revenue_detail()
        params = self.inputs.tax_parameters
        periods = self.period_index
        rate_gold = params.ndpi_rate_gold.reindex(periods).fillna(0.0)
        rate_silver = params.ndpi_rate_silver.reindex(periods).fillna(0.0)

        gold_tax = (detail["gold_revenue_rub"].reindex(periods).fillna(0.0) * rate_gold) / 1_000.0
        silver_tax = (detail["silver_revenue_rub"].reindex(periods).fillna(0.0) * rate_silver) / 1_000.0

        ndpi_detail = pd.DataFrame(
            {
                "ndpi_gold": gold_tax,
                "ndpi_silver": silver_tax,
                "ndpi_total": gold_tax + silver_tax,
            },
            index=periods,
        )
        self._ndpi_detail = ndpi_detail
        return ndpi_detail.copy()

    def _calc_property_tax_detail(self) -> pd.DataFrame:
        """Return property-tax detail derived from CAPEX and depreciation."""

        if self._property_tax_detail is not None:
            return self._property_tax_detail.copy()

        capex = self.calc_capex()
        depreciation = self.inputs.depreciation.reindex(self.period_index).fillna(0.0)
        rate_series = self.inputs.tax_parameters.property_tax_rate.reindex(self.period_index).fillna(0.0)

        additions = capex.clip(lower=0.0)
        disposals = (-capex).clip(lower=0.0)
        opening_values: list[float] = []
        closing_values: list[float] = []
        base_values: list[float] = []

        opening = 0.0
        for period in self.period_index:
            opening_values.append(opening)
            add = additions.loc[period]
            dispose = disposals.loc[period]
            average_base = opening + 0.5 * add - 0.5 * dispose
            base_values.append(max(average_base, 0.0))
            closing = opening + add - dispose - depreciation.loc[period]
            closing = max(closing, 0.0)
            closing_values.append(closing)
            opening = closing

        property_tax_base = pd.Series(base_values, index=self.period_index)
        property_tax = property_tax_base * rate_series
        detail = pd.DataFrame(
            {
                "property_tax": property_tax,
                "property_tax_base": property_tax_base,
                "nbv_opening": pd.Series(opening_values, index=self.period_index),
                "nbv_closing": pd.Series(closing_values, index=self.period_index),
            },
            index=self.period_index,
        )
        self._property_tax_detail = detail
        return detail.copy()

    def calc_revenue(self) -> pd.Series:
        """
        Compute commodity revenue per period before taxes and investments.

        Returns:
            pd.Series: Revenue in thousand RUB (positive = inflow) derived from
            ore mined, grades, recoveries, and RUB-per-gram prices. The
            computation keeps prices in RUB/gram and only converts to thousand
            RUB after applying pricing to metal grams to align with the cash-flow
            convention used throughout the model.
        """

        detail = self.calc_revenue_detail()
        revenue = detail["total_revenue_thousand"].copy()
        revenue.name = "revenue"
        return revenue

        production = self.inputs.production
        prices = self.inputs.prices

        ore_tonnes = production["ore_mined_kton"] * 1_000.0
        gold_production = ore_tonnes * production["gold_grade_gpt"] * production["gold_recovery"] / 1_000.0
        silver_production = ore_tonnes * production["silver_grade_gpt"] * production["silver_recovery"] / 1_000.0

        gold_revenue = gold_production * prices["gold_price_rub_g"]
        silver_revenue = silver_production * prices["silver_price_rub_g"]

        revenue = gold_revenue.add(silver_revenue, fill_value=0.0)
        revenue.name = "revenue"
        return revenue

    def calc_costs(self) -> pd.Series:
        """
        Return total operating costs (including production taxes) by period.

        Values are expressed in thousand RUB with positive numbers representing
        cash outflows. If the caller switches the power scenario, the method
        swaps the relevant OPEX components before enforcing that the category
        sum matches the total operating-cost series.

        Returns:
            pd.Series: Total operating costs (thousand RUB, positive = outflow).
        """

        base_costs = (
            self.opex_breakdown.sum(axis=1)
            .reindex(self.period_index)
            .fillna(0.0)
        )
        production_taxes = (
            self.inputs.opex["production_taxes"]
            .reindex(self.period_index)
            .fillna(0.0)
        )
        costs = base_costs.add(production_taxes, fill_value=0.0)
        costs.name = "operating_costs"

        operating_costs_total = self.operating_costs_total
        if not np.allclose(base_costs.values, operating_costs_total.values, atol=1e-6):
            raise AssertionError(
                "OPEX breakdown does not match total operating costs."
            )
        return costs

    def calc_capex(self) -> pd.Series:
        """
        Return capital expenditures by period in thousand RUB.

        Positive values represent cash outflows for sustaining or growth CAPEX.
        The base CAPEX series reflects the scenario captured in FEM.xlsx; when
        `power_mode` differs from that base, the model removes the original
        power-plant CAPEX contribution and replaces it with the target-mode
        series (purchase infrastructure vs. self-generation plant).

        Returns:
            pd.Series: CAPEX profile in thousand RUB (positive = outflow).
        """

        capex = (
            self.inputs.capex.reindex(self.period_index).fillna(0.0).copy()
        )
        if self.power_inputs:
            components = self._power_capex_components()
            base_mode = self.power_inputs.base_mode or "purchase"
            target_mode = self.power_mode or base_mode
            base_component = components.get(base_mode, pd.Series(0.0, index=self.period_index))
            target_component = components.get(target_mode, pd.Series(0.0, index=self.period_index))
            capex = capex.sub(base_component, fill_value=0.0).add(target_component, fill_value=0.0)
        capex.name = "capex"
        return capex

    def calc_working_capital(self) -> pd.Series:
        """Return change in working capital (thousand RUB; positive = outflow)."""

        working_capital = self.calc_working_capital_detailed()
        return working_capital["working_capital_change"].copy()

    def calc_working_capital_detailed(self) -> pd.DataFrame:
        """
        Return detailed working capital balances by period in RUB.

        The DataFrame includes columns ["inventory", "receivables", "payables",
        "net_working_capital", "working_capital_change"], where all values are in
        thousand RUB and positive changes represent cash outflows. Working
        capital levels follow the ratio defined in
        `ModelInputs.working_capital_inputs` and respect the OPEX composition.
        """

        if self._working_capital_detail is not None:
            return self._working_capital_detail.copy()

        inputs = self.inputs.working_capital_inputs
        ratio = inputs.ratio_implied.reindex(self.period_index).fillna(0.0)
        if ratio.eq(0.0).all():
            ratio = inputs.ratio_input.reindex(self.period_index).fillna(0.0)

        total_opex = self.opex_breakdown.sum(axis=1).reindex(self.period_index).fillna(0.0)
        net_level = ratio * total_opex

        inventory_share = inputs.inventory_share.reindex(self.period_index).fillna(0.0)
        payables_share = inputs.payables_share.reindex(self.period_index).fillna(0.0)

        inventory = net_level * inventory_share
        payables = net_level * payables_share
        receivables = net_level + payables - inventory

        detail = pd.DataFrame(
            {
                "inventory": inventory,
                "receivables": receivables,
                "payables": payables,
                "net_working_capital": net_level,
            },
            index=self.period_index,
        )
        detail["working_capital_change"] = detail["net_working_capital"].diff().fillna(
            detail["net_working_capital"]
        )

        ledger_balance = detail["inventory"] + detail["receivables"] - detail["payables"]
        ledger_change = ledger_balance.diff().fillna(ledger_balance)
        if not np.allclose(ledger_change.values, detail["working_capital_change"].values, atol=1e-6):
            raise AssertionError("Working capital change does not equal ledger deltas.")

        self._working_capital_detail = detail
        return detail.copy()

    def calc_taxes_basic(self) -> pd.Series:
        """
        Compute simplified profit tax on operating profit.

        Returns:
            pd.Series: Profit tax in thousand RUB (positive = cash paid).
        """

        revenue = self.calc_revenue()
        costs = self.calc_costs()
        operating_profit = revenue - costs
        taxable_profit = operating_profit.clip(lower=0.0)
        taxes = taxable_profit * self.inputs.profit_tax_rate
        taxes.name = "profit_tax"
        return taxes

    def calc_taxes_detailed(self) -> pd.DataFrame:
        """
        Compute detailed tax components under the selected detailed mode.

        Modes:
            - "standard_loss_pool" / "detailed": classic single loss-pool rule
              where taxable profit equals EBIT less the accumulated losses.
            - "excel_cumulative": Excel-style rule that taxes only increases in
              the positive portion of cumulative profit (per the FEM workbook
              note). In both cases mineral/property taxes continue to mirror the
              input series, and Excel outputs remain reconciliation-only.

        Returns:
            pd.DataFrame: Columns for profit, mineral, and property taxes plus
            diagnostic EBIT/tax-base data (thousand RUB; positive = outflow).
        """

        detailed_mode = (
            "standard_loss_pool" if self.taxes_mode in {"detailed", "standard_loss_pool"} else self.taxes_mode
        )
        if detailed_mode in self._tax_detail:
            return self._tax_detail[detailed_mode].copy()

        revenue_detail = self.calc_revenue_detail()
        revenue = revenue_detail["total_revenue_thousand"]
        opex_total = self.calc_costs()
        depreciation = self.inputs.depreciation.reindex(self.period_index).fillna(0.0)
        capex = self.calc_capex()

        tax_engine = TaxEngine(
            period_index=self.period_index,
            tax_params=self.inputs.tax_parameters,
            profit_tax_rate=self.inputs.profit_tax_rate,
            loss_mode=detailed_mode,
        )

        interest_expense: Optional[pd.Series] = None
        if self.inputs.tax_parameters.interest_tax_deductible and self.inputs.financing is not None:
            interest_expense = self.calc_financing_schedule()["interest_expense"]

        ndpi_detail = self._calc_ndpi_detail()
        property_detail = self._calc_property_tax_detail()
        tax_detail = tax_engine.calculate(
            revenue_detail=revenue_detail,
            operating_costs=opex_total,
            depreciation=depreciation,
            capex=capex,
            interest_expense=interest_expense,
            ndpi_detail=ndpi_detail,
            property_detail=property_detail,
        )

        self._tax_detail[detailed_mode] = tax_detail
        return tax_detail.copy()

    def calc_taxes(self):
        """
        Return the active tax calculation (basic or detailed).

        "basic" returns a Series of thousand-RUB outflows, "detailed" returns a
        DataFrame with the full tax breakdown.
        """

        if self.taxes_mode == "basic":
            return self.calc_taxes_basic()
        return self.calc_taxes_detailed()

    def calc_unlevered_cashflow(self) -> pd.Series:
        """
        Compute unlevered free cash flow by period in RUB.

        Positive values represent net cash inflow after revenue, operating costs,
        taxes, CAPEX, and working-capital movements.

        Returns:
            pd.Series: Unlevered free cash flow in thousand RUB (positive =
            inflow).
        """

        revenue = self.calc_revenue()
        costs = self.calc_costs()
        capex = self.calc_capex()
        taxes = self.calc_taxes()
        working_capital = self.calc_working_capital()

        if isinstance(taxes, pd.DataFrame):
            tax_series = taxes["profit_tax"]
        else:
            tax_series = taxes

        cash_flow = revenue - costs - tax_series - capex - working_capital
        cash_flow.name = "unlevered_cash_flow"
        return cash_flow

    def discount_cashflow(
        self,
        rate: float,
        cash_flow: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Return the cash flow discounted with the mid-period convention.

        Args:
            rate (float): Decimal discount rate (e.g., 0.15).
            cash_flow (pd.Series | None): Cash flow (thousand RUB, positive =
                inflow). Defaults to unlevered cash flow.

        Returns:
            pd.Series: Discounted cash flow in thousand RUB using mid-period
            convention.
        """

        cash_series = cash_flow if cash_flow is not None else self.calc_unlevered_cashflow()
        periods = pd.Series(
            range(1, len(cash_series) + 1),
            index=cash_series.index,
            dtype=float,
        )
        discount_factors = 1.0 / np.power(1.0 + rate, periods - 0.5)
        discounted = cash_series * discount_factors
        discounted.name = f"discounted_cf_{rate:.4f}"
        return discounted

    def calc_npv(self, rate: Optional[float] = None) -> float:
        """
        Return NPV of unlevered cash flow at the provided discount rate.

        Args:
            rate (float | None): Discount rate (decimal). Defaults to the
                `ModelInputs` base rate.

        Returns:
            float: NPV in thousand RUB.
        """

        discount_rate = rate if rate is not None else self.inputs.discount_rate
        discounted_cf = self.discount_cashflow(discount_rate)
        return float(discounted_cf.sum())

    def calc_irr(self) -> float:
        """Return the decimal IRR of the unlevered cash flow series."""

        cash_flow = self.calc_unlevered_cashflow()
        irr_value = npf.irr(cash_flow.values)
        return float(irr_value) if irr_value is not None else float("nan")

    def calc_payback(self) -> float:
        """
        Return the simple payback period (in project periods).

        Based on cumulative unlevered cash flow; returns NaN if payback is never
        reached.

        Returns:
            float: Payback in model periods (NaN if cumulative never turns positive).
        """

        cash_flow = self.calc_unlevered_cashflow()
        cumulative = cash_flow.cumsum()
        if (cumulative <= 0).all():
            return float("nan")

        periods = self.inputs.timeline["period"].to_numpy(dtype=float)
        prev_cumulative = 0.0
        prev_period = periods[0] - 1.0

        for idx, (period, cum_value) in enumerate(zip(periods, cumulative.values)):
            if cum_value >= 0:
                cash = cash_flow.iloc[idx]
                if cash == 0:
                    return float(period)
                gap = abs(prev_cumulative)
                fraction = gap / cash if cash != 0 else 0.0
                return float(prev_period + fraction)
            prev_cumulative = cumulative.iloc[idx]
            prev_period = period

        return float("nan")

    def calc_discounted_payback(self, rate: Optional[float] = None) -> float:
        """
        Return the discounted payback period using mid-period discounted cash flow.

        If the discounted cumulative cash flow never turns non-negative, returns
        NaN.

        Returns:
            float: Discounted payback in project periods (NaN if unmet).
        """

        discount_rate = rate if rate is not None else self.inputs.discount_rate
        discounted_cf = self.discount_cashflow(discount_rate)
        cumulative = discounted_cf.cumsum()
        if (cumulative <= 0).all():
            return float("nan")

        periods = self.inputs.timeline["period"].to_numpy(dtype=float)
        prev_cumulative = 0.0
        prev_period = periods[0] - 1.0

        for idx, (period, cum_value) in enumerate(zip(periods, cumulative.values)):
            if cum_value >= 0:
                cash = discounted_cf.iloc[idx]
                if cash == 0:
                    return float(period)
                gap = abs(prev_cumulative)
                fraction = gap / cash if cash != 0 else 0.0
                return float(prev_period + fraction)
            prev_cumulative = cumulative.iloc[idx]
            prev_period = period

        return float("nan")

    def calc_financing_schedule(self) -> pd.DataFrame:
        """
        Build the debt-and-equity financing schedule in thousand RUB per period.

        Returns:
            DataFrame with columns
            ["debt_opening", "debt_draw", "debt_repayment",
             "debt_closing", "interest_expense", "equity_injection"].
            Debt balances are positive when outstanding. Drawdowns are positive
            inflows, whereas repayments and interest are positive cash outflows
            borne by the project. Equity injections are reported as positive
            values that indicate shareholder contributions (inflows to the
            project, outflows from the owners).
        """

        if self._financing_schedule is not None:
            return self._financing_schedule.copy()

        periods = self.period_index
        zero_series = pd.Series(0.0, index=periods)
        unlevered_cf = self.calc_unlevered_cashflow()
        funding_need = (-unlevered_cf).clip(lower=0.0)
        financing = self.inputs.financing

        if financing is None:
            schedule = pd.DataFrame(
                {
                    "debt_opening": zero_series,
                    "debt_draw": zero_series,
                    "debt_repayment": zero_series,
                    "debt_closing": zero_series,
                    "interest_expense": zero_series,
                    "equity_injection": funding_need,
                },
                index=periods,
            )
            self._financing_schedule = schedule
            return schedule.copy()

        mode = getattr(financing, "financing_mode", "synthetic")
        if mode == "excel":
            schedule = self._build_excel_financing_schedule(financing)
            self._financing_schedule = schedule
            return schedule.copy()

        debt_share = float(np.clip(financing.debt_ratio, 0.0, 1.0))
        total_need = float(funding_need.sum())
        remaining_capacity = (
            max(financing.debt_amount, 0.0)
            if financing.debt_amount is not None
            else total_need * debt_share
        )

        debt_draw_values: list[float] = []
        equity_values: list[float] = []
        for need in funding_need.values:
            if need <= 0.0:
                debt_draw_values.append(0.0)
                equity_values.append(0.0)
                continue
            desired_debt = need * debt_share
            draw = min(need, desired_debt, remaining_capacity)
            draw = max(draw, 0.0)
            remaining_capacity = max(0.0, remaining_capacity - draw)
            equity = need - draw
            debt_draw_values.append(draw)
            equity_values.append(equity)

        debt_draw = pd.Series(debt_draw_values, index=periods, name="debt_draw")
        equity_injection = pd.Series(
            equity_values, index=periods, name="equity_injection"
        )
        total_debt = float(debt_draw.sum())

        opening_values: list[float] = []
        repayment_values: list[float] = []
        closing_values: list[float] = []
        interest_values: list[float] = []

        if total_debt == 0.0:
            schedule = pd.DataFrame(
                {
                    "debt_opening": zero_series,
                    "debt_draw": debt_draw,
                    "debt_repayment": zero_series,
                    "debt_closing": zero_series,
                    "interest_expense": zero_series,
                    "equity_injection": equity_injection,
                },
                index=periods,
            )
            self._financing_schedule = schedule
            return schedule.copy()

        draw_indices = np.flatnonzero(debt_draw.values > 1e-6)
        first_draw_idx = int(draw_indices[0]) if draw_indices.size else 0
        tenor_periods = financing.tenor_periods or len(periods)
        grace_periods = min(financing.grace_periods, max(tenor_periods - 1, 0))
        repayment_start_idx = max(first_draw_idx, first_draw_idx + grace_periods)
        repayment_end_idx = min(
            len(periods) - 1,
            first_draw_idx + tenor_periods - 1,
        )
        if repayment_end_idx < repayment_start_idx:
            repayment_end_idx = repayment_start_idx
        amortization_periods = max(1, repayment_end_idx - repayment_start_idx + 1)

        outstanding = 0.0
        repayments_done = 0
        interest_rate = financing.interest_rate

        for idx, period in enumerate(periods):
            opening_values.append(outstanding)
            draw = debt_draw.iloc[idx]
            interest = outstanding * interest_rate
            # TODO(aragamin): feed the interest expense into a tax shield once the tax engine supports it.
            interest_values.append(interest)

            debt_before_repay = outstanding + draw
            principal = 0.0
            in_amortization = idx >= repayment_start_idx and repayments_done < amortization_periods
            if in_amortization and debt_before_repay > 0.0:
                remaining_slots = amortization_periods - repayments_done
                remaining_slots = max(1, remaining_slots)
                principal = debt_before_repay / remaining_slots
                repayments_done += 1
            principal = min(principal, debt_before_repay)
            repayments_done = min(repayments_done, amortization_periods)

            repayment_values.append(principal)
            closing = debt_before_repay - principal
            closing_values.append(closing)
            outstanding = closing

        schedule = pd.DataFrame(
            {
                "debt_opening": opening_values,
                "debt_draw": debt_draw.values,
                "debt_repayment": repayment_values,
                "debt_closing": closing_values,
                "interest_expense": interest_values,
                "equity_injection": equity_injection.values,
            },
            index=periods,
        )

        lhs = schedule["debt_opening"] + schedule["debt_draw"] - schedule["debt_repayment"]
        if not np.allclose(lhs.values, schedule["debt_closing"].values, atol=1e-6):
            raise AssertionError("Financing schedule does not balance debt flows.")
        coverage = schedule["debt_draw"] + schedule["equity_injection"]
        if not np.allclose(coverage.values, funding_need.values, atol=1e-6):
            raise AssertionError("Financing flows do not cover funding needs.")

        self._financing_schedule = schedule
        return schedule.copy()

    def _build_excel_financing_schedule(self, financing: FinancingInputs) -> pd.DataFrame:
        """Return the financing schedule directly from Excel reference series."""

        periods = self.period_index
        debt_closing = financing.excel_debt_balance.reindex(periods).fillna(0.0)
        debt_draw = financing.excel_debt_draw.reindex(periods).fillna(0.0)
        debt_repayment = financing.excel_debt_repayment.reindex(periods).fillna(0.0)
        interest_expense = financing.excel_interest_expense.reindex(periods).fillna(0.0)
        equity_injection = financing.excel_equity_injection.reindex(periods).fillna(0.0)
        debt_opening = debt_closing.shift(1, fill_value=0.0)

        return pd.DataFrame(
            {
                "debt_opening": debt_opening,
                "debt_draw": debt_draw,
                "debt_repayment": debt_repayment,
                "debt_closing": debt_closing,
                "interest_expense": interest_expense,
                "equity_injection": equity_injection,
            },
            index=periods,
        )

    def calc_levered_cashflow(self) -> pd.Series:
        """
        Return levered cash flow once financing adjustments are applied.

        The series is expressed in thousand RUB with positive numbers
        representing cash inflows to equity stakeholders. Equity contributions
        remain implicit (negative levered cash flow) and are also reported in
        the financing schedule for traceability. Interest tax shields are
        intentionally excluded for now (TODO once tax engine is extended).

        Returns:
            pd.Series: Levered cash flow available to equity (thousand RUB;
            positive = inflow).
        """

        schedule = self.calc_financing_schedule()
        unlevered = self.calc_unlevered_cashflow()
        levered = (
            unlevered
            + schedule["debt_draw"]
            - schedule["debt_repayment"]
            - schedule["interest_expense"]
        )
        levered.name = "levered_cash_flow"
        return levered

    def calc_levered_npv(self, discount_rate: float) -> float:
        """
        Return NPV of levered cash flow at the provided discount rate.

        Args:
            discount_rate (float): Decimal discount rate for equity cash flow.

        Returns:
            float: Levered NPV in thousand RUB using mid-period discounting.
        """

        levered_cf = self.calc_levered_cashflow()
        discounted = self.discount_cashflow(discount_rate, levered_cf)
        return float(discounted.sum())

    def calc_levered_irr(self) -> float:
        """
        Return the IRR (decimal) of the levered cash flow series.

        Returns:
            float: Levered IRR (np.nan if the series does not cross zero).
        """

        levered_cf = self.calc_levered_cashflow()
        irr_value = npf.irr(levered_cf.values)
        return float(irr_value) if irr_value is not None else float("nan")

    def check_against_excel(self) -> pd.DataFrame:
        """
        Compare Python outputs against Excel source values period by period.

        Returns:
            DataFrame with columns
            ["metric", "subcategory", "period", "excel_value",
             "python_value", "diff_abs", "diff_pct"]. Values are measured in RUB
            (diff_pct in decimal terms).
        """

        records = []
        periods = self.period_index

        # Revenue comparison.
        revenue_python = self.calc_revenue()
        revenue_excel = self.inputs.excel_revenue.reindex(periods).fillna(0.0)
        for period in periods:
            records.append(
                self._reconciliation_record(
                    metric="revenue",
                    subcategory=None,
                    period=period,
                    excel_value=revenue_excel.loc[period],
                    python_value=revenue_python.loc[period],
                )
            )

        # Total OPEX comparison.
        opex_python = self.calc_costs()
        opex_excel = self.inputs.excel_opex_total.reindex(periods).fillna(0.0)
        for period in periods:
            records.append(
                self._reconciliation_record(
                    metric="opex_total",
                    subcategory=None,
                    period=period,
                    excel_value=opex_excel.loc[period],
                    python_value=opex_python.loc[period],
                )
            )

        # Working capital change comparison.
        wc_python_change = self.calc_working_capital()
        wc_excel_change = (
            self.inputs.excel_working_capital_change.reindex(periods).fillna(0.0)
        )
        for period in periods:
            records.append(
                self._reconciliation_record(
                    metric="working_capital_change",
                    subcategory=None,
                    period=period,
                    excel_value=wc_excel_change.loc[period],
                    python_value=wc_python_change.loc[period],
                )
            )

        # OPEX breakdown comparison.
        for category in self.opex_breakdown.columns:
            excel_series = self.inputs.opex_breakdown[category]
            python_series = self.opex_breakdown[category]
            for period in periods:
                records.append(
                    self._reconciliation_record(
                        metric="opex",
                        subcategory=category,
                        period=period,
                        excel_value=excel_series.loc[period],
                        python_value=python_series.loc[period],
                    )
                )

        # Working capital components.
        python_wc = self.calc_working_capital_detailed()
        excel_wc = self._excel_working_capital_components()
        for component in ("inventory", "receivables", "payables"):
            for period in periods:
                records.append(
                    self._reconciliation_record(
                        metric="working_capital",
                        subcategory=f"{component}_balance",
                        period=period,
                        excel_value=excel_wc.at[period, component],
                        python_value=python_wc.at[period, component],
                    )
                )

        power_inputs = self.inputs.power_inputs
        if power_inputs is not None:
            capex_components = self._power_capex_components()
            capex_pairs = [
                ("purchase", power_inputs.capex_purchase_power),
                ("selfgen", power_inputs.capex_selfgen_power),
            ]
            for mode, excel_series in capex_pairs:
                python_series = capex_components.get(
                    mode, pd.Series(0.0, index=self.period_index)
                )
                placeholder = excel_series is None or bool(
                    getattr(excel_series, "attrs", {}).get("note")
                )
                excel_values = (
                    pd.Series(np.nan, index=self.period_index)
                    if placeholder
                    else excel_series.reindex(self.period_index).fillna(0.0)
                )
                for period in periods:
                    records.append(
                        self._reconciliation_record(
                            metric="power_capex",
                            subcategory=mode,
                            period=period,
                            excel_value=excel_values.loc[period],
                            python_value=python_series.loc[period],
                        )
                    )

            opex_pairs = [
                ("purchase", power_inputs.purchase_power_opex),
                ("selfgen", power_inputs.selfgen_power_opex),
            ]
            for mode, excel_series in opex_pairs:
                excel_values = (
                    excel_series.reindex(self.period_index).fillna(0.0)
                    if excel_series is not None
                    else pd.Series(np.nan, index=self.period_index)
                )
                python_series = excel_values.copy()
                for period in periods:
                    records.append(
                        self._reconciliation_record(
                            metric="power_opex",
                            subcategory=mode,
                            period=period,
                            excel_value=excel_values.loc[period],
                            python_value=python_series.loc[period],
                        )
                    )

        # Tax comparison (always use detailed view).
        taxes_python = self.calc_taxes_detailed()
        taxes_excel = self.inputs.excel_taxes.reindex(periods).fillna(0.0)
        for column in ["profit_tax", "mineral_extraction_tax", "property_tax", "total_tax"]:
            for period in periods:
                records.append(
                    self._reconciliation_record(
                        metric="tax",
                        subcategory=column,
                        period=period,
                        excel_value=taxes_excel.at[period, column],
                        python_value=taxes_python.at[period, column],
                    )
                )

        # Tax debug information.
        depreciation_excel = self.inputs.depreciation.reindex(periods).fillna(0.0)
        excel_taxable_profit = self.inputs.excel_taxable_profit_base.reindex(periods)
        excel_loss_pool = self.inputs.excel_loss_pool.reindex(periods)
        loss_pool_available = not bool(
            self.inputs.excel_profit_loss_cumulative.attrs.get("note")
        )

        for period in periods:
            records.append(
                self._reconciliation_record(
                    metric="tax_debug",
                    subcategory="depreciation",
                    period=period,
                    excel_value=depreciation_excel.loc[period],
                    python_value=taxes_python.at[period, "depreciation"],
                )
            )

        if "taxable_profit" in taxes_python.columns:
            for period in periods:
                records.append(
                    self._reconciliation_record(
                        metric="tax_debug",
                        subcategory="taxable_profit",
                        period=period,
                        excel_value=excel_taxable_profit.loc[period],
                        python_value=taxes_python.at[period, "taxable_profit"],
                    )
                )

        if "loss_pool_carry" in taxes_python.columns:
            for period in periods:
                excel_value = (
                    excel_loss_pool.loc[period]
                    if loss_pool_available
                    else np.nan
                )
                records.append(
                    self._reconciliation_record(
                        metric="tax_debug",
                        subcategory="loss_pool",
                        period=period,
                        excel_value=excel_value,
                        python_value=taxes_python.at[period, "loss_pool_carry"],
                    )
                )

        if "taxable_profit_excel_rule" in taxes_python.columns:
            for period in periods:
                records.append(
                    self._reconciliation_record(
                        metric="tax_debug",
                        subcategory="taxable_profit_excel_rule",
                        period=period,
                        excel_value=np.nan,
                        python_value=taxes_python.at[period, "taxable_profit_excel_rule"],
                    )
                )

        if "cumulative_profit_excel_rule" in taxes_python.columns:
            for period in periods:
                records.append(
                    self._reconciliation_record(
                        metric="tax_debug",
                        subcategory="cumulative_profit_excel_rule",
                        period=period,
                        excel_value=np.nan,
                        python_value=taxes_python.at[period, "cumulative_profit_excel_rule"],
                    )
                )

        financing_inputs = self.inputs.financing
        if financing_inputs is not None:
            schedule = self.calc_financing_schedule()
            financing_pairs = [
                ("debt_balance", financing_inputs.excel_debt_balance, schedule["debt_closing"]),
                ("interest", financing_inputs.excel_interest_expense, schedule["interest_expense"]),
                ("equity_injection", financing_inputs.excel_equity_injection, schedule["equity_injection"]),
            ]
            for subcategory, excel_series, python_series in financing_pairs:
                excel_values = (
                    excel_series.reindex(periods).fillna(0.0)
                    if excel_series is not None
                    else pd.Series(np.nan, index=periods)
                )
                for period in periods:
                    records.append(
                        self._reconciliation_record(
                            metric="financing",
                            subcategory=subcategory,
                            period=period,
                            excel_value=excel_values.loc[period],
                            python_value=python_series.loc[period],
                        )
                    )

        # Unlevered cash flow comparison.
        python_cf = self.calc_unlevered_cashflow()
        excel_cf = self.inputs.excel_unlevered_cashflow.reindex(periods).fillna(0.0)
        for period in periods:
            records.append(
                self._reconciliation_record(
                    metric="unlevered_cashflow",
                    subcategory=None,
                    period=period,
                    excel_value=excel_cf.loc[period],
                    python_value=python_cf.loc[period],
                    )
                )

        result = pd.DataFrame(records)

        # Invariants: OPEX categories must reconcile to total operating costs.
        categories_total = self.opex_breakdown.sum(axis=1)
        excel_operating_total = self.inputs.opex["operating_costs"].reindex(self.period_index)
        if not np.allclose(categories_total.values, excel_operating_total.values, atol=1e-6):
            raise AssertionError("OPEX categories do not sum to total operating costs.")

        # Invariants: changes in balances must match working capital delta.
        wc_detail = self.calc_working_capital_detailed()
        balance = wc_detail["inventory"] + wc_detail["receivables"] - wc_detail["payables"]
        balance_change = balance.diff().fillna(balance)
        if not np.allclose(balance_change.values, wc_detail["working_capital_change"].values, atol=1e-6):
            raise AssertionError("Working capital change does not match component deltas.")

        return result

    def _excel_working_capital_components(self) -> pd.DataFrame:
        inputs = self.inputs.working_capital_inputs
        net = inputs.excel_target.reindex(self.period_index).fillna(0.0)
        inventory = net * inputs.inventory_share.reindex(self.period_index).fillna(0.0)
        payables = net * inputs.payables_share.reindex(self.period_index).fillna(0.0)
        receivables = net + payables - inventory

        detail = pd.DataFrame(
            {
                "inventory": inventory,
                "receivables": receivables,
                "payables": payables,
                "net_working_capital": net,
            },
            index=self.period_index,
        )
        detail["working_capital_change"] = detail["net_working_capital"].diff().fillna(
            detail["net_working_capital"]
        )
        return detail

    @staticmethod
    def _reconciliation_record(
        metric: str,
        subcategory: Optional[str],
        period: int,
        excel_value: float,
        python_value: float,
    ) -> Dict[str, Any]:
        diff = python_value - excel_value
        diff_abs = abs(diff)
        if excel_value != 0:
            diff_pct = diff / abs(excel_value)
        else:
            diff_pct = np.nan
        return {
            "metric": metric,
            "subcategory": subcategory,
            "period": int(period),
            "excel_value": float(excel_value),
            "python_value": float(python_value),
            "diff_abs": float(diff_abs),
            "diff_pct": float(diff_pct) if not np.isnan(diff_pct) else np.nan,
        }
