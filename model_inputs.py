from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


DATA_PATH = Path(__file__).resolve().parent / "data" / "FEM.xlsx"

OPEX_CATEGORY_MAP = {
    "Шары для мельниц": "processing_consumables",
    "Конвейерная лента": "processing_consumables",
    "Дизельное топливо (выбранный вариант)": "fuel",
    "Масла и смазки": "lubricants",
    "Аммиачная селитра": "explosives",
    "Ремонт и обслуживание ОС (% от первоначальной ст-ти)": "maintenance",
    "Административно-хозяйственные расходы": "g_and_a",
    "Доставка продукции до аффинажного завода": "logistics",
    "Расходы на доставку вахты": "logistics",
    "Стоимость аффинажа": "refining",
    "Услуги банка при продаже драгоценных металлов": "refining",
    "Административно-управленический персонал": "admin_labor",
    "Производственный персонал": "production_labor",
    "Вспомогательный персонал  (выбранный вариант)": "support_labor",
}

INVENTORY_CATEGORY_KEYS = (
    "processing_consumables",
    "fuel",
    "lubricants",
    "explosives",
)

PAYABLE_CATEGORY_KEYS = (
    "processing_consumables",
    "fuel",
    "lubricants",
    "explosives",
    "maintenance",
    "logistics",
    "refining",
)


@dataclass
class WorkingCapitalInputs:
    """Assumptions needed to derive detailed working-capital components."""

    ratio_input: pd.Series
    ratio_implied: pd.Series
    inventory_share: pd.Series
    payables_share: pd.Series
    excel_target: pd.Series


@dataclass
class ModelInputs:
    """Container with structured inputs required by MineProjectModel."""

    timeline: pd.DataFrame
    production: pd.DataFrame
    prices: pd.DataFrame
    opex: pd.DataFrame
    opex_breakdown: pd.DataFrame
    capex: pd.Series
    working_capital: pd.Series
    working_capital_inputs: WorkingCapitalInputs
    depreciation: pd.Series
    discount_rate: float
    profit_tax_rate: float
    scenario: str
    excel_revenue: pd.Series
    excel_opex_total: pd.Series
    excel_working_capital_change: pd.Series
    excel_taxes: pd.DataFrame
    excel_profit_loss_period: pd.Series
    excel_profit_loss_cumulative: pd.Series
    excel_taxable_profit_base: pd.Series
    excel_loss_pool: pd.Series
    excel_unlevered_cashflow: pd.Series
    metadata: Dict[str, Any]

    def copy_with(self, **updates: Any) -> "ModelInputs":
        """Return a shallow copy with updated fields."""

        params = {
            "timeline": self.timeline.copy(),
            "production": self.production.copy(),
            "prices": self.prices.copy(),
            "opex": self.opex.copy(),
            "opex_breakdown": self.opex_breakdown.copy(),
            "capex": self.capex.copy(),
            "working_capital": self.working_capital.copy(),
            "working_capital_inputs": self.working_capital_inputs,
            "depreciation": self.depreciation.copy(),
            "discount_rate": self.discount_rate,
            "profit_tax_rate": self.profit_tax_rate,
            "scenario": self.scenario,
            "excel_revenue": self.excel_revenue.copy(),
            "excel_opex_total": self.excel_opex_total.copy(),
            "excel_working_capital_change": self.excel_working_capital_change.copy(),
            "excel_taxes": self.excel_taxes.copy(),
            "excel_profit_loss_period": self.excel_profit_loss_period.copy(),
            "excel_profit_loss_cumulative": self.excel_profit_loss_cumulative.copy(),
            "excel_taxable_profit_base": self.excel_taxable_profit_base.copy(),
            "excel_loss_pool": self.excel_loss_pool.copy(),
            "excel_unlevered_cashflow": self.excel_unlevered_cashflow.copy(),
            "metadata": dict(self.metadata),
        }
        params.update(updates)
        return ModelInputs(**params)


def load_model_inputs(excel_path: Optional[Path] = None) -> ModelInputs:
    """Load model inputs from FEM.xlsx."""

    path = Path(excel_path) if excel_path else DATA_PATH
    inputs_sheet = pd.read_excel(path, sheet_name="Исходные данные")
    calendar_sheet = pd.read_excel(path, sheet_name="Горный календарь")
    opex_sheet = pd.read_excel(path, sheet_name="OPEX")
    taxes_sheet = pd.read_excel(path, sheet_name="Налоги")
    cashflow_sheet = pd.read_excel(path, sheet_name="Cash Flow")

    timeline, period_index, value_columns = _load_timeline(inputs_sheet)

    production = _load_production(calendar_sheet, period_index, value_columns)
    prices, discount_rate, profit_tax_rate, scenario = _load_prices_and_rates(
        inputs_sheet, period_index, value_columns
    )

    cashflow_data = _load_cashflow_data(
        cashflow_sheet, period_index, value_columns
    )

    opex_breakdown = _load_opex_breakdown(
        opex_sheet, period_index, value_columns
    )
    working_capital_inputs = _build_working_capital_inputs(
        inputs_sheet, opex_sheet, opex_breakdown, period_index, value_columns
    )
    working_capital_change = _calculate_working_capital_change(
        working_capital_inputs.excel_target
    )
    tax_data = _load_tax_data(
        taxes_sheet, period_index, value_columns
    )
    excel_profit_loss_period = tax_data["profit_loss_period"]
    excel_profit_loss_cumulative = tax_data["profit_loss_cumulative"]

    if profit_tax_rate == 0.0:
        excel_taxable_profit_base = pd.Series(
            0.0, index=period_index, name="excel_taxable_profit_base"
        )
    else:
        excel_taxable_profit_base = (
            tax_data["excel_taxes"]["profit_tax"]
            .reindex(period_index)
            .divide(profit_tax_rate)
            .fillna(0.0)
        )
        excel_taxable_profit_base.name = "excel_taxable_profit_base"

    excel_loss_pool = (-excel_profit_loss_cumulative).clip(lower=0.0)
    excel_loss_pool.name = "excel_loss_pool"

    metadata = {"source_file": str(path)}
    return ModelInputs(
        timeline=timeline,
        production=production,
        prices=prices,
        opex=cashflow_data["opex"],
        opex_breakdown=opex_breakdown,
        capex=cashflow_data["capex"],
        working_capital=working_capital_change,
        working_capital_inputs=working_capital_inputs,
        depreciation=tax_data["depreciation"],
        discount_rate=discount_rate,
        profit_tax_rate=profit_tax_rate,
        scenario=scenario,
        excel_revenue=cashflow_data["revenue"],
        excel_opex_total=cashflow_data["opex_total"],
        excel_working_capital_change=cashflow_data["working_capital_change"],
        excel_taxes=tax_data["excel_taxes"],
        excel_profit_loss_period=excel_profit_loss_period,
        excel_profit_loss_cumulative=excel_profit_loss_cumulative,
        excel_taxable_profit_base=excel_taxable_profit_base,
        excel_loss_pool=excel_loss_pool,
        excel_unlevered_cashflow=cashflow_data["excel_unlevered_cashflow"],
        metadata=metadata,
    )


def _load_timeline(
    inputs_sheet: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Index, Iterable[str]]:
    label_col = _normalized_column(inputs_sheet, "Unnamed: 3")

    period_row = inputs_sheet.loc[label_col.eq("Номер периода")]
    if period_row.empty:
        raise ValueError("Не найден ряд с номерами периодов.")

    raw_periods = pd.to_numeric(period_row.iloc[0, 7:], errors="coerce").dropna()
    period_index = pd.Index(raw_periods.astype(int), name="period")
    value_columns = list(raw_periods.index)

    years_row = inputs_sheet.loc[label_col.eq("Календарный год")]
    if years_row.empty:
        raise ValueError("Не найден ряд с календарными годами.")
    raw_years = pd.to_numeric(
        years_row[value_columns].iloc[0], errors="coerce"
    ).ffill()
    years = raw_years.astype(int).to_numpy()

    timeline = pd.DataFrame(
        {"period": period_index.astype(int), "year": years},
        index=period_index,
    )
    timeline.index.name = "period"
    return timeline, period_index, value_columns


def _load_production(
    calendar_sheet: pd.DataFrame,
    period_index: pd.Index,
    value_columns: Iterable[str],
) -> pd.DataFrame:
    label_col = _normalized_column(calendar_sheet, "Unnamed: 3")

    production = pd.DataFrame(index=period_index)
    production["ore_mined_kton"] = _extract_series(
        calendar_sheet, label_col, value_columns, period_index, "Добыча руды", "тыс. т."
    )
    production["gold_grade_gpt"] = _extract_series(
        calendar_sheet,
        label_col,
        value_columns,
        period_index,
        "Содержание золота в руде",
        "г/т",
    )
    production["silver_grade_gpt"] = _extract_series(
        calendar_sheet,
        label_col,
        value_columns,
        period_index,
        "Содержание серебра в руде",
        "г/т",
    )
    production["gold_recovery"] = _extract_series(
        calendar_sheet,
        label_col,
        value_columns,
        period_index,
        "Извлечение золота на фабрике",
        "%",
    )
    production["silver_recovery"] = _extract_series(
        calendar_sheet,
        label_col,
        value_columns,
        period_index,
        "Извлечение серебра на фабрике",
        "%",
    )

    return production


def _load_prices_and_rates(
    inputs_sheet: pd.DataFrame,
    period_index: pd.Index,
    value_columns: Iterable[str],
) -> tuple[pd.DataFrame, float, float, str]:
    label_col = _normalized_column(inputs_sheet, "Unnamed: 3")

    prices = pd.DataFrame(index=period_index)
    prices["exchange_rate_rub_usd"] = _extract_series(
        inputs_sheet,
        label_col,
        value_columns,
        period_index,
        "Курс национальной валюты к доллару",
        "руб./$",
    )
    prices["gold_price_usd_oz"] = _extract_series(
        inputs_sheet, label_col, value_columns, period_index, "Цена на золото", "$/Oz"
    )
    prices["gold_price_rub_g"] = _extract_series(
        inputs_sheet, label_col, value_columns, period_index, "Цена на золото", "руб/грамм"
    )
    prices["silver_price_usd_oz"] = _extract_series(
        inputs_sheet, label_col, value_columns, period_index, "Цена на серебро", "$/Oz"
    )
    prices["silver_price_rub_g"] = _extract_series(
        inputs_sheet,
        label_col,
        value_columns,
        period_index,
        "Цена на серебро",
        "руб/грамм",
    )

    discount_rate = float(
        _extract_scalar(inputs_sheet, label_col, "Ставка дисконтирования")
    )
    profit_tax_rate = float(
        _extract_scalar(inputs_sheet, label_col, "Налог на прибыль")
    )
    scenario_row = inputs_sheet.loc[label_col.eq("Сценарий"), value_columns]
    if scenario_row.empty:
        scenario = ""
    else:
        scenario_values = scenario_row.iloc[0].dropna()
        scenario = str(scenario_values.iloc[0]) if not scenario_values.empty else ""

    return prices, discount_rate, profit_tax_rate, scenario


def _load_cashflow_data(
    cashflow_sheet: pd.DataFrame,
    period_index: pd.Index,
    value_columns: Iterable[str],
) -> Dict[str, Any]:
    label_col = _normalized_column(cashflow_sheet, "Unnamed: 3")

    revenue = _extract_series(
        cashflow_sheet, label_col, value_columns, period_index, "Доход предприятия"
    ).fillna(0.0)
    operating_costs_raw = _extract_series(
        cashflow_sheet, label_col, value_columns, period_index, "Операционные затраты"
    ).fillna(0.0)
    operating_costs = operating_costs_raw.abs()
    production_taxes_raw = _extract_series(
        cashflow_sheet,
        label_col,
        value_columns,
        period_index,
        "Налоги и отчисления в себестоимости",
    ).fillna(0.0)
    production_taxes = production_taxes_raw.abs()
    profit_tax_raw = _extract_series(
        cashflow_sheet, label_col, value_columns, period_index, "Налог на прибыль"
    ).fillna(0.0)
    profit_tax = profit_tax_raw.abs()

    working_capital_raw = _extract_series(
        cashflow_sheet,
        label_col,
        value_columns,
        period_index,
        "Потребность в оборотных средствах",
    ).fillna(0.0)
    working_capital_change = -working_capital_raw

    capex_raw = _extract_series(
        cashflow_sheet,
        label_col,
        value_columns,
        period_index,
        "Вложения во внеоборотные активы",
    ).fillna(0.0)
    capex = capex_raw.abs()

    excel_unlevered_cashflow = (
        revenue
        + operating_costs_raw
        + production_taxes_raw
        + profit_tax_raw
        + working_capital_raw
        + capex_raw
    )

    opex = pd.DataFrame(
        {
            "operating_costs": operating_costs,
            "production_taxes": production_taxes,
        },
        index=period_index,
    )

    return {
        "opex": opex,
        "capex": capex,
        "working_capital_change": working_capital_change,
        "revenue": revenue.abs(),
        "opex_total": (operating_costs + production_taxes),
        "profit_tax": profit_tax_raw.abs(),
        "excel_unlevered_cashflow": excel_unlevered_cashflow,
    }


def _load_opex_breakdown(
    opex_sheet: pd.DataFrame,
    period_index: pd.Index,
    value_columns: Iterable[str],
) -> pd.DataFrame:
    label_col = _normalized_column(opex_sheet, "Unnamed: 3")
    breakdown: Dict[str, pd.Series] = {}

    for label, category in OPEX_CATEGORY_MAP.items():
        try:
            series = _extract_series(
                opex_sheet, label_col, value_columns, period_index, label
            )
        except KeyError:
            continue
        if category in breakdown:
            breakdown[category] = breakdown[category].add(series, fill_value=0.0)
        else:
            breakdown[category] = series

    # Ensure all configured categories exist in the DataFrame.
    for category in set(OPEX_CATEGORY_MAP.values()):
        breakdown.setdefault(
            category, pd.Series(0.0, index=period_index, name=category)
        )

    opex_breakdown = pd.DataFrame(breakdown, index=period_index).fillna(0.0)
    return opex_breakdown


def _load_tax_data(
    taxes_sheet: pd.DataFrame,
    period_index: pd.Index,
    value_columns: Iterable[str],
) -> Dict[str, Any]:
    label_col = _normalized_column(taxes_sheet, "Unnamed: 3")

    depreciation = _extract_optional_series(
        taxes_sheet,
        label_col,
        value_columns,
        period_index,
        "Амортизация",
        note="TODO: replace placeholder depreciation if tax sheet missing values.",
    )

    profit_tax = _extract_optional_series(
        taxes_sheet,
        label_col,
        value_columns,
        period_index,
        "Налог на прибыль",
    )
    ndpi_gold = _extract_optional_series(
        taxes_sheet,
        label_col,
        value_columns,
        period_index,
        "НДПИ золото",
    )
    ndpi_silver = _extract_optional_series(
        taxes_sheet,
        label_col,
        value_columns,
        period_index,
        "НДПИ серебро",
    )
    property_tax = _extract_optional_series(
        taxes_sheet,
        label_col,
        value_columns,
        period_index,
        "Налог на имущество",
    )

    profit_loss_period = _extract_optional_series(
        taxes_sheet,
        label_col,
        value_columns,
        period_index,
        "Прибыль/убыток периода",
        note="Excel profit/loss reported per period.",
    )
    profit_loss_cumulative = _extract_optional_series(
        taxes_sheet,
        label_col,
        value_columns,
        period_index,
        "Прибыль/убыток нарастающим итогом",
        note="Excel cumulative profit/loss series for taxable tracking.",
    )

    mineral_tax = ndpi_gold.add(ndpi_silver, fill_value=0.0)

    excel_taxes = pd.DataFrame(
        {
            "profit_tax": profit_tax,
            "mineral_extraction_tax": mineral_tax,
            "property_tax": property_tax,
        },
        index=period_index,
    )
    excel_taxes["total_tax"] = excel_taxes.sum(axis=1)

    return {
        "depreciation": depreciation,
        "excel_taxes": excel_taxes,
        "profit_loss_period": profit_loss_period,
        "profit_loss_cumulative": profit_loss_cumulative,
    }


def _build_working_capital_inputs(
    inputs_sheet: pd.DataFrame,
    opex_sheet: pd.DataFrame,
    opex_breakdown: pd.DataFrame,
    period_index: pd.Index,
    value_columns: Iterable[str],
) -> WorkingCapitalInputs:
    label_inputs = _normalized_column(inputs_sheet, "Unnamed: 3")
    ratio_input = _extract_series(
        inputs_sheet,
        label_inputs,
        value_columns,
        period_index,
        "Норма оборотных средств (% от операц. расходов)",
    ).fillna(0.0)

    label_opex = _normalized_column(opex_sheet, "Unnamed: 3")
    excel_target = _extract_series(
        opex_sheet, label_opex, value_columns, period_index, "Норма оборотных средств"
    ).fillna(0.0)

    total_opex = opex_breakdown.sum(axis=1)
    inventory_base = opex_breakdown[
        [key for key in INVENTORY_CATEGORY_KEYS if key in opex_breakdown.columns]
    ].sum(axis=1)
    payables_base = opex_breakdown[
        [key for key in PAYABLE_CATEGORY_KEYS if key in opex_breakdown.columns]
    ].sum(axis=1)

    inventory_share = _safe_divide(inventory_base, total_opex).clip(lower=0.0)
    payables_share = _safe_divide(payables_base, total_opex).clip(lower=0.0)

    ratio_implied = _safe_divide(excel_target, total_opex).fillna(0.0)

    return WorkingCapitalInputs(
        ratio_input=ratio_input,
        ratio_implied=ratio_implied,
        inventory_share=inventory_share,
        payables_share=payables_share,
        excel_target=excel_target,
    )


def _calculate_working_capital_change(excel_target: pd.Series) -> pd.Series:
    net = excel_target.fillna(0.0)
    change = net.diff().fillna(net)
    change.name = "working_capital_change"
    return change


def _extract_series(
    sheet: pd.DataFrame,
    label_col: pd.Series,
    value_columns: Iterable[str],
    period_index: pd.Index,
    label: str,
    unit: Optional[str] = None,
) -> pd.Series:
    mask = label_col.eq(label)
    if unit is not None:
        unit_col = _normalized_column(sheet, "Unnamed: 4")
        mask &= unit_col.eq(unit)

    if not mask.any():
        raise KeyError(f"Не найден ряд '{label}' (unit: {unit}).")

    row = sheet.loc[mask, list(value_columns)]
    if row.empty:
        raise ValueError(f"Ряд '{label}' не содержит данных по периодам.")
    values = pd.to_numeric(row.iloc[0], errors="coerce").fillna(0.0).astype(float)
    return pd.Series(values.values, index=period_index, name=label)


def _extract_scalar(
    sheet: pd.DataFrame,
    label_col: pd.Series,
    label: str,
    column_label: str = "Unnamed: 7",
) -> Any:
    mask = label_col.eq(label)
    if not mask.any():
        raise KeyError(f"Не найден параметр '{label}'.")
    value = sheet.loc[mask, column_label].iloc[0]
    return value


def _normalized_column(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column].astype(str).str.strip().replace({"nan": ""})


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.astype(float).copy()
    denom = denominator.replace(0.0, pd.NA)
    result = result.divide(denom)
    return result.fillna(0.0)


def _extract_optional_series(
    sheet: pd.DataFrame,
    label_col: pd.Series,
    value_columns: Iterable[str],
    period_index: pd.Index,
    label: str,
    note: str = "",
) -> pd.Series:
    try:
        series = _extract_series(
            sheet, label_col, value_columns, period_index, label
        )
    except KeyError:
        series = pd.Series(0.0, index=period_index, name=label)
        if note:
            series.attrs["note"] = note
    return series.fillna(0.0)
