## Tax Modes

- **basic** – simplified rule that taxes positive operating profit without losses. Useful for quick sanity checks only.
- **detailed / standard_loss_pool** – default engine that computes EBIT from Python drivers, tracks a single loss pool, and applies the profit-tax rate to positive EBIT after the pool is depleted. Mineral and property taxes remain direct inputs.
- **excel_cumulative** – replicates the Excel workbook note: cumulative profit is summed each period and only increases in its positive portion are taxed (`tax_base_t = max(0, C⁺_t − C⁺_{t-1})`). Period profit and cumulative profit are computed entirely in Python; Excel outputs stay reconciliation-only.

Both detailed variants share the same interface, mineral/property tax handling, and reconciliation hooks. Choose `excel_cumulative` when validating against FEM.xlsx’s tax sheet; stick with `standard_loss_pool` for generic project analysis. No mode ever feeds Excel output series back into core calculations.
