"""Plan the readable columns of a combined data release."""

import pandas as pd

OUTPUT = [
    "symbol",
    "permno",
    "spell",
    "kind",
    "first_date",
    "last_date",
    "crsp_until",
    "first_yf_date",
    "linked",
    "possible_double",
]


def plan_columns(spells: pd.DataFrame, yahoo: pd.DataFrame) -> pd.DataFrame:  # noqa: PLR0912, PLR0915 — complete naming rules
    """Assign Yahoo names and independent placeholder names to readable spells."""
    y = yahoo.copy()
    for name in ("first_yf_date", "last_yf_date", "common_day"):
        y[name] = pd.to_datetime(y[name])
    s = spells.copy()
    for name in ("first_date", "last_date"):
        s[name] = pd.to_datetime(s[name])
    joined = y.status.eq("linked") & y.common_day.notna()
    if y.loc[joined, "permno"].duplicated().any():
        raise ValueError("multiple linked Yahoo symbols for one stock")
    if y.symbol.duplicated().any():
        raise ValueError("duplicate symbol")
    linked_stocks = set(y.loc[joined, "permno"].dropna())
    candidates = y.status.isin(["unlinked", "ambiguous", "duplicate"])
    flagged = (
        candidates & (~y.permno.isin(linked_stocks) | y.status.eq("duplicate")) & y.permno.notna()
    )
    double_stocks = set(y.loc[flagged, "permno"])
    y["kind"] = "yahoo"
    y.loc[joined, "kind"] = "linked_yahoo"
    y["spell"] = y.common_spell.where(joined)
    y["first_date"] = y.first_yf_date
    y["last_date"] = y.last_yf_date
    y["crsp_until"] = y.first_yf_date.where(joined)
    y["linked"] = joined
    y["possible_double"] = flagged
    matching = s.rename(columns={"spell": "common_spell", "first_date": "spell_first"})
    y = y.merge(
        matching[["permno", "common_spell", "spell_first"]],
        on=["permno", "common_spell"],
        how="left",
        validate="many_to_one",
    )
    if y.loc[joined, "spell_first"].isna().any():
        raise ValueError("common spell not found")
    y.loc[joined, "first_date"] = y.loc[joined, ["first_yf_date", "spell_first"]].min(axis=1)
    names = y.loc[y.status.eq("linked") & y.permno.notna()].set_index("permno")
    if names.index.duplicated().any():
        raise ValueError("multiple linked Yahoo symbols for one stock")
    result = [y[OUTPUT]]
    for permno, group in s.groupby("permno", sort=False):
        group = group.sort_values(["first_date", "spell"]).copy()
        if permno in names.index:
            name = names.loc[permno]
            group = group[group.last_date < name.first_yf_date]
            if pd.notna(name.common_day):
                group = group[group.spell != name.common_spell]
            placeholder = False
        else:
            placeholder = True
        if group.empty:
            continue
        group["kind"] = "spell"
        group["symbol"] = [f"PERMNO-{permno}~{i}" for i in range(1, len(group) + 1)]
        if placeholder:
            group.loc[group.index[-1], ["kind", "symbol"]] = ["placeholder", f"PERMNO-{permno}"]
        group["crsp_until"] = pd.NaT
        group["first_yf_date"] = pd.NaT
        group["linked"] = False
        group["possible_double"] = permno in double_stocks
        result.append(group[OUTPUT])
    output = pd.concat(result, ignore_index=True)
    if output.symbol.duplicated().any():
        raise ValueError("duplicate symbol")
    return output
