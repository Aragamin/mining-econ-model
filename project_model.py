from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import numpy_financial as npf
import pandas as pd

from model_inputs import ModelInputs


class MineProjectModel:
    """Core project cash-flow model for the mining project."""

    def __init__(self, inputs: ModelInputs, taxes_mode: str = "detailed") -> None:
        """Initialize the model with structured inputs."""

        if taxes_mode not in {"basic", "detailed"}:
            raise ValueError("taxes_mode must be either 'basic' or 'detailed'.")

        self.inputs = inputs
        self.period_index = inputs.timeline.index
        self._working_capital_detail: Optional[pd.DataFrame] = None
        self._tax_detail: Optional[pd.DataFrame] = None
        self.opex_breakdown = inputs.opex_breakdown.copy()
        self.taxes_mode = taxes_mode

    def calc_revenue(self) -> pd.Series:
        """
        Return periodic revenue in RUB before taxes and investments.

        Uses ore mined, grade, and recovery assumptions from `ModelInputs.production`
        together with RUB-per-gram prices from `ModelInputs.prices`. Positive values
        represent cash inflows.
        """

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

        Costs are expressed in RUB and positive values represent cash outflows.
        The sum of the OPEX breakdown is asserted to match the reported operating
        costs series in `ModelInputs.opex`.
        """

        base_costs = (
            self.opex_breakdown.sum(axis=1)
            .reindex(self.period_index)
            .fillna(0.0)
        )
        production_taxes = self.inputs.opex["production_taxes"]
        costs = base_costs.add(production_taxes, fill_value=0.0)
        costs.name = "operating_costs"

        operating_costs_total = (
            self.inputs.opex["operating_costs"]
            .reindex(self.period_index)
            .fillna(0.0)
        )
        if not np.allclose(base_costs.values, operating_costs_total.values, atol=1e-6):
            raise AssertionError(
                "OPEX breakdown does not match total operating costs."
            )
        return costs

    def calc_capex(self) -> pd.Series:
        """
        Return capital expenditures by period in RUB.

        Positive values represent cash outflows for sustaining or growth CAPEX.
        """

        capex = self.inputs.capex.copy()
        capex.name = "capex"
        return capex

    def calc_working_capital(self) -> pd.Series:
        """Return change in working capital (positive = cash outflow) in RUB."""

        working_capital = self.calc_working_capital_detailed()
        return working_capital["working_capital_change"].copy()

    def calc_working_capital_detailed(self) -> pd.DataFrame:
        """
        Return detailed working capital balances by period in RUB.

        The DataFrame includes columns ["inventory", "receivables", "payables",
        "net_working_capital", "working_capital_change"], where positive changes
        represent cash outflows. Working capital levels follow the ratio defined
        in `ModelInputs.working_capital_inputs` and respect the OPEX composition.
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

        Returns a Series in RUB where positive values represent cash paid to the
        treasury.
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
        Compute detailed tax components using EBIT and a single loss pool.

        Taxable profit is derived strictly from Periodic EBIT less the loss pool,
        both computed within the Python model; Excel-derived profit/taxable/loss
        sequences serve only as reconciliation controls in `check_against_excel()`.
        Mineral and property taxes continue to mirror the FEM.xlsx series.
        All tax values are reported in RUB with positive numbers representing cash
        paid to the authorities.
        """

        if self._tax_detail is not None:
            return self._tax_detail.copy()

        revenue = self.calc_revenue()
        opex_total = self.calc_costs()
        depreciation = self.inputs.depreciation.reindex(self.period_index).fillna(0.0)

        ebit = revenue - opex_total - depreciation
        profit_tax_rate = self.inputs.profit_tax_rate

        loss_pool = 0.0
        profit_tax_values: list[float] = []
        taxable_profit_values: list[float] = []
        loss_pool_carry: list[float] = []

        for value in ebit.values:
            taxable_profit = value - loss_pool
            taxable_profit_values.append(taxable_profit)
            if taxable_profit > 0.0:
                profit_tax = taxable_profit * profit_tax_rate
                loss_pool = 0.0
            else:
                profit_tax = 0.0
                loss_pool = -taxable_profit
            profit_tax_values.append(profit_tax)
            loss_pool_carry.append(loss_pool)

        profit_tax_series = pd.Series(
            profit_tax_values, index=self.period_index, name="profit_tax"
        )
        taxable_profit_series = pd.Series(
            taxable_profit_values, index=self.period_index, name="taxable_profit"
        )
        loss_pool_series = pd.Series(
            loss_pool_carry, index=self.period_index, name="loss_pool_carry"
        )

        mineral_tax_series = (
            self.inputs.excel_taxes["mineral_extraction_tax"]
            .reindex(self.period_index)
            .fillna(0.0)
        )  # TODO: replace with mineral tax calculation from production drivers.
        property_tax_series = (
            self.inputs.excel_taxes["property_tax"]
            .reindex(self.period_index)
            .fillna(0.0)
        )  # TODO: replace with property tax calculation from asset base.

        total_tax_series = (
            profit_tax_series
            + mineral_tax_series
            + property_tax_series
        )

        tax_detail = pd.DataFrame(
            {
                "profit_tax": profit_tax_series,
                "mineral_extraction_tax": mineral_tax_series,
                "property_tax": property_tax_series,
                "total_tax": total_tax_series,
                "depreciation": depreciation,
                "ebit": ebit,
                "taxable_profit": taxable_profit_series,
                "loss_pool_carry": loss_pool_series,
            },
            index=self.period_index,
        )
        self._tax_detail = tax_detail
        return tax_detail.copy()

    def calc_taxes(self):
        """
        Return the active tax calculation (basic or detailed).

        "basic" returns a Series of RUB outflows, "detailed" returns a DataFrame
        with the full tax breakdown.
        """

        if self.taxes_mode == "basic":
            return self.calc_taxes_basic()
        return self.calc_taxes_detailed()

    def calc_unlevered_cashflow(self) -> pd.Series:
        """
        Compute unlevered free cash flow by period in RUB.

        Positive values represent net cash inflow after revenue, operating costs,
        taxes, CAPEX, and working-capital movements.
        """

        revenue = self.calc_revenue()
        costs = self.calc_costs()
        capex = self.calc_capex()
        taxes = self.calc_taxes()
        working_capital = self.calc_working_capital()

        if isinstance(taxes, pd.DataFrame):
            tax_series = taxes["total_tax"]
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

        Arguments:
          rate: decimal discount rate (e.g., 0.15).
          cash_flow: Series to discount; defaults to unlevered cash flow.
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
        Return NPV in RUB of unlevered cash flow at the provided discount rate.

        If rate is omitted the model's base discount rate is used.
        """

        discount_rate = rate if rate is not None else self.inputs.discount_rate
        discounted_cf = self.discount_cashflow(discount_rate)
        return float(discounted_cf.sum())

    def calc_irr(self) -> float:
        """Return IRR (decimal) of the unlevered cash flow series."""

        cash_flow = self.calc_unlevered_cashflow()
        irr_value = npf.irr(cash_flow.values)
        return float(irr_value) if irr_value is not None else float("nan")

    def calc_payback(self) -> float:
        """
        Return the simple payback period (in project periods).

        Based on cumulative unlevered cash flow; returns NaN if payback is never reached.
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

        If the discounted cumulative cash flow never turns non-negative, returns NaN.
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
        Build the financing schedule for debt and equity without applying it yet.

        Expected future columns include:
          - opening_debt (RUB balance at period start)
          - drawdown (positive = debt inflow)
          - principal_repayment (positive = cash outflow)
          - interest_expense (positive = cash outflow)
          - equity_drawdown (positive = inflow)
          - closing_debt (RUB balance at period end)

        All amounts would be expressed in RUB. Principal and interest are cash
        outflows (positive = paid), while drawdowns and equity injections are
        inflows. For now this method raises NotImplementedError until a full
        financing engine is defined.
        """

        raise NotImplementedError("TODO: implement financing schedule calculations.")

    def calc_levered_cashflow(self) -> pd.Series:
        """
        Return levered cash flow once financing adjustments are applied.

        Levered cash flow will be derived from the unlevered CF by adding financing
        inflows (drawdowns/equity) and subtracting cash outflows for interest and
        principal as laid out by `calc_financing_schedule`. Returns a Series in RUB
        with positive values representing inflows.
        """

        raise NotImplementedError("TODO: implement levered cash flow calculations.")

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
