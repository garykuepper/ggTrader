"""Calculate expected-growth predictors from already aligned accounting inputs."""

import math
from numbers import Real


def investment_change(*, prior_assets: float, assets: float, next_assets: float) -> float:
    """Return the one-year-ahead investment-to-assets change from three fiscal years.

    The caller must establish consecutive fiscal periods, issuer identity and the
    public availability of all three figures before using this as a training target.
    In particular, the next-year figure must never be used before it is available.
    """
    values = (prior_assets, assets, next_assets)
    try:
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value <= 0
            for value in values
        ):
            raise ValueError("Invalid annual assets")
        prior, current, following = (float(value) for value in values)
        change = following / current - current / prior
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError("Invalid annual assets") from exc
    if not math.isfinite(change):
        raise ValueError("Non-finite investment change")
    return change


def predictors(
    *,
    market_equity: float,
    long_debt: float,
    short_debt: float,
    assets: float,
    revenue: float,
    cost_of_goods: float,
    selling_admin: float,
    research_development: float | None,
    receivables_change: float | None,
    inventory_change: float | None,
    prepaid_change: float | None,
    deferred_revenue_change: float | None,
    payables_change: float | None,
    accrued_change: float | None,
    income: float | None,
    prior_equity: float | None,
    year_ago_income: float | None,
    year_ago_prior_equity: float | None,
) -> tuple[float, float, float | None]:
    """Return log q, operating cash flow/assets and year-over-year ROE change.

    For the paper's q predictor, market equity is summed across a firm's share classes;
    the caller must establish point-in-time share-class identity and values. Inputs
    must match Compustat definitions; SEC tags are not automatically equivalent,
    particularly for SG&A, R&D and depreciation. In particular, adding reported R&D
    to a separate reported SG&A line may double-count the R&D add-back unless the
    supplied SG&A has first been mapped to the paper's combined field. Engineering
    and acquired R&D can also differ across definitions. The deferred-revenue change is the
    change in the combined current and noncurrent amounts, not separate imputed changes.
    Caller must establish identity, periods, availability and the four-month annual lag.
    None means a known missing item, not a failed retrieval. Missing R&D and changes
    become zero; missing ROE stays missing (the paper sets it to zero for regressions).
    Quarterly equity inputs must use the paper's book-equity definition, including
    its deferred-tax and preferred-stock adjustments; reported equity alone may differ.
    Nonpositive quarterly equity fails closed; this can exclude otherwise usable firms.
    """
    required = (market_equity, long_debt, short_debt, assets, revenue, cost_of_goods, selling_admin)
    optional = (
        research_development,
        receivables_change,
        inventory_change,
        prepaid_change,
        deferred_revenue_change,
        payables_change,
        accrued_change,
    )
    quarterly = (income, prior_equity, year_ago_income, year_ago_prior_equity)
    try:
        for value in (*required, *optional, *quarterly):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            ):
                raise ValueError("Invalid accounting input")
    except OverflowError as exc:
        raise ValueError("Invalid accounting input") from exc
    if any(value is None for value in required):
        raise ValueError("Missing required accounting input")
    if assets <= 0 or market_equity <= 0 or long_debt < 0 or short_debt < 0:
        raise ValueError("Invalid assets or q numerator")
    if any(value is not None and value <= 0 for value in (prior_equity, year_ago_prior_equity)):
        raise ValueError("Invalid quarterly equity")
    rd, ar, inv, prepaid, deferred, payables, accrued = (
        0 if value is None else value for value in optional
    )
    log_q = math.log((market_equity + long_debt + short_debt) / assets)
    cash_flow = (
        revenue
        - cost_of_goods
        - selling_admin
        + rd
        - ar
        - inv
        - prepaid
        + deferred
        + payables
        + accrued
    ) / assets
    roe_change = None
    if all(value is not None for value in quarterly):
        roe_change = income / prior_equity - year_ago_income / year_ago_prior_equity
    if not all(math.isfinite(v) for v in (log_q, cash_flow)) or (
        roe_change is not None and not math.isfinite(roe_change)
    ):
        raise ValueError("Non-finite predictor")
    return log_q, cash_flow, roe_change
