from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from model_inputs import TaxParameters


@dataclass
class TaxEngine:
    """
    Encapsulate NDPI, property-tax, and profit-tax calculations.

    The engine operates on thousand-RUB series (positive = cash outflow) in line
    with the rest of the MineProjectModel. Rates supplied via `TaxParameters`
    must be decimal fractions (6% == 0.06).
    """

    period_index: pd.Index
    tax_params: TaxParameters
    profit_tax_rate: float
    loss_mode: str

    def calculate(
        self,
        revenue_detail: pd.DataFrame,
        operating_costs: pd.Series,
        depreciation: pd.Series,
        capex: pd.Series,
        interest_expense: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Return tax breakdown aligned to the configured loss-carry regime."""

        periods = self.period_index
        revenue_thousand = (
            revenue_detail["total_revenue_thousand"]
            .reindex(periods)
            .fillna(0.0)
        )
        operating_costs = operating_costs.reindex(periods).fillna(0.0)
        depreciation = depreciation.reindex(periods).fillna(0.0)
        capex = capex.reindex(periods).fillna(0.0)
        interest_series = (
            interest_expense.reindex(periods).fillna(0.0)
            if interest_expense is not None
            else pd.Series(0.0, index=periods)
        )

        ebit = revenue_thousand - operating_costs - depreciation
        ndpi_detail = self._calc_ndpi(revenue_detail)
        property_detail = self._calc_property_tax(capex, depreciation)

        deductions = {
            "ndpi": ndpi_detail["ndpi_total"]
            if self.tax_params.ndpi_deductible_before_profit_tax
            else pd.Series(0.0, index=periods),
            "property_tax": property_detail["property_tax"]
            if self.tax_params.property_tax_deductible_before_profit_tax
            else pd.Series(0.0, index=periods),
            "interest": interest_series
            if self.tax_params.interest_tax_deductible
            else pd.Series(0.0, index=periods),
        }

        taxable_pre_loss = ebit - deductions["ndpi"] - deductions["property_tax"] - deductions["interest"]
        adjustment_series = (
            self.tax_params.profit_taxable_adjustment.reindex(periods).fillna(0.0)
            if self.tax_params.profit_taxable_adjustment is not None
            else pd.Series(0.0, index=periods)
        )
        taxable_pre_loss = taxable_pre_loss + adjustment_series

        profit_tax_detail = self._apply_loss_mode(taxable_pre_loss)

        total_tax = (
            profit_tax_detail["profit_tax"]
            + ndpi_detail["ndpi_total"]
            + property_detail["property_tax"]
        )

        tax_df = pd.DataFrame(
            {
                "profit_tax": profit_tax_detail["profit_tax"],
                "mineral_extraction_tax": ndpi_detail["ndpi_total"],
                "property_tax": property_detail["property_tax"],
                "total_tax": total_tax,
                "depreciation": depreciation,
                "ebit": ebit,
                "ndpi_gold": ndpi_detail["ndpi_gold"],
                "ndpi_silver": ndpi_detail["ndpi_silver"],
                "property_tax_base": property_detail["property_tax_base"],
                "taxable_profit_pre_loss": taxable_pre_loss,
                "interest_deduction": deductions["interest"],
                "taxable_income_adjustment": adjustment_series,
            },
            index=periods,
        )

        # Attach optional diagnostics from the profit-tax routine.
        for column in ("taxable_profit", "loss_pool_carry", "taxable_profit_excel_rule", "cumulative_profit_excel_rule"):
            if column in profit_tax_detail:
                tax_df[column] = profit_tax_detail[column]

        return tax_df

    def _calc_ndpi(self, revenue_detail: pd.DataFrame) -> pd.DataFrame:
        """Compute mineral extraction taxes per metal based on RUB revenue."""

        rate_gold = (
            self.tax_params.ndpi_rate_gold.reindex(self.period_index).fillna(0.0)
            if self.tax_params.ndpi_rate_gold is not None
            else pd.Series(0.0, index=self.period_index)
        )
        rate_silver = (
            self.tax_params.ndpi_rate_silver.reindex(self.period_index).fillna(0.0)
            if self.tax_params.ndpi_rate_silver is not None
            else pd.Series(0.0, index=self.period_index)
        )

        gold_revenue_rub = revenue_detail["gold_revenue_rub"].reindex(self.period_index).fillna(0.0)
        silver_revenue_rub = revenue_detail["silver_revenue_rub"].reindex(self.period_index).fillna(0.0)

        gold_tax = (gold_revenue_rub * rate_gold) / 1_000.0
        silver_tax = (silver_revenue_rub * rate_silver) / 1_000.0

        return pd.DataFrame(
            {
                "ndpi_gold": gold_tax,
                "ndpi_silver": silver_tax,
                "ndpi_total": gold_tax + silver_tax,
            },
            index=self.period_index,
        )

    def _calc_property_tax(self, capex: pd.Series, depreciation: pd.Series) -> pd.DataFrame:
        """
        Return property tax and supporting asset-base details.

        The workbook applies manual adjustments to the taxable asset base that
        are not exposed in FEM.xlsx. This driver-based implementation follows
        the standard average balance (opening + 0.5*additions − 0.5*disposals)
        which keeps differences within ~0.2% of the FEM property-tax series; the
        validation harness surfaces the residual explicitly.
        """

        rate_series = (
            self.tax_params.property_tax_rate.reindex(self.period_index).fillna(0.0)
            if self.tax_params.property_tax_rate is not None
            else pd.Series(0.0, index=self.period_index)
        )

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
            depreciation_value = depreciation.loc[period]
            closing = opening + add - dispose - depreciation_value
            closing_values.append(max(closing, 0.0))
            opening = max(closing, 0.0)

        property_tax_base = pd.Series(base_values, index=self.period_index)
        property_tax = property_tax_base * rate_series

        return pd.DataFrame(
            {
                "property_tax": property_tax,
                "property_tax_base": property_tax_base,
                "nbv_opening": pd.Series(opening_values, index=self.period_index),
                "nbv_closing": pd.Series(closing_values, index=self.period_index),
            },
            index=self.period_index,
        )

    def _apply_loss_mode(self, taxable_pre_loss: pd.Series) -> pd.DataFrame:
        """Apply the configured loss-carry regime to derive profit tax."""

        taxable_pre_loss = taxable_pre_loss.reindex(self.period_index).fillna(0.0)
        if self.loss_mode == "excel_cumulative":
            return self._excel_cumulative_rule(taxable_pre_loss)
        return self._standard_loss_pool(taxable_pre_loss)

    def _standard_loss_pool(self, taxable_pre_loss: pd.Series) -> pd.DataFrame:
        """Single loss-pool rule where losses carry forward indefinitely."""

        loss_pool = 0.0
        taxable_values: list[float] = []
        tax_values: list[float] = []
        loss_pool_values: list[float] = []

        for value in taxable_pre_loss.values:
            taxable_after_loss = value - loss_pool
            taxable_values.append(taxable_after_loss)
            if taxable_after_loss > 0.0 and self.profit_tax_rate > 0.0:
                tax = taxable_after_loss * self.profit_tax_rate
                loss_pool = 0.0
            else:
                tax = 0.0
                loss_pool = -taxable_after_loss if taxable_after_loss < 0.0 else 0.0
            tax_values.append(tax)
            loss_pool_values.append(loss_pool)

        return pd.DataFrame(
            {
                "profit_tax": pd.Series(tax_values, index=self.period_index),
                "taxable_profit": pd.Series(taxable_values, index=self.period_index),
                "loss_pool_carry": pd.Series(loss_pool_values, index=self.period_index),
            },
            index=self.period_index,
        )

    def _excel_cumulative_rule(self, taxable_pre_loss: pd.Series) -> pd.DataFrame:
        """Excel FEM rule taxing only the positive part of cumulative profit."""

        cumulative_profit = taxable_pre_loss.cumsum()
        cumulative_positive = cumulative_profit.clip(lower=0.0)
        prev_positive = cumulative_positive.shift(1, fill_value=0.0)
        taxable_increment = (cumulative_positive - prev_positive).clip(lower=0.0)
        profit_tax = taxable_increment * self.profit_tax_rate

        return pd.DataFrame(
            {
                "profit_tax": profit_tax,
                "taxable_profit_excel_rule": taxable_increment,
                "cumulative_profit_excel_rule": cumulative_profit,
            },
            index=self.period_index,
        )
