"""Data loaders."""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # utils/ -> repository root
RESULTS = ROOT / "code" / "experimentresult"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
for _d in (TABLES, FIGURES):  # ensure output directories exist before any script writes to them
    _d.mkdir(parents=True, exist_ok=True)


def _uci_feature_names():
    names = ["StartYear", "StartQuarter", "CompletionYear", "CompletionQuarter"]
    names += [f"V{i}" for i in range(1, 9)]  # V-1..V-8 physical and financial variables
    econ = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    for lag in range(1, 6):  # 5 time-lagged economic variables
        for v in econ:
            names.append(f"V{v}_L{lag}")
    names += ["OUT_V9_sales", "OUT_V10_cost"]
    return names


def load_uci():
    """UCI Residential Building Data Set #437.
    Returns X (107 inputs), y_cost (V-10 actual construction cost), y_sales (V-9 sale price), feature_names."""
    p = ROOT / "dataset" / "raw" / "uci_437" / "Residential-Building-Data-Set.xlsx"
    raw = pd.read_excel(p, sheet_name="Data", header=None)
    data = raw.iloc[2:].reset_index(drop=True).apply(pd.to_numeric, errors="coerce")
    data = data.dropna().reset_index(drop=True)
    names = _uci_feature_names()
    data.columns = names[: data.shape[1]]
    X = data.iloc[:, :107].values
    y_cost = data["OUT_V10_cost"].values
    y_sales = data["OUT_V9_sales"].values
    feat = names[:107]
    return {"X": X, "y_cost": y_cost, "y_sales": y_sales, "feature_names": feat, "n": len(data)}


def load_nyc_sca_raw():
    """NYC SCA Capital Projects (project-phase level, K-12 school real cost).
    Returns a cleaned DataFrame (with parsed cost/dates/derived features), no filtering applied."""
    p = ROOT / "dataset" / "raw" / "nyc_sca" / "nyc_sca_capital_projects.csv"
    df = pd.read_csv(p, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    def numcol(s):
        return pd.to_numeric(s.astype(str).str.replace(r"[^0-9.]", "", regex=True), errors="coerce")

    df["cost_estimate"] = numcol(df["Final Estimate of Actual Costs Through End of Phase Amount"])
    df["cost_spend"] = numcol(df["Total Phase Actual Spending Amount"])
    df["budget"] = numcol(df["Project Budget Amount"])
    for c in ["Project Phase Actual Start Date", "Project Phase Planned End Date", "Project Phase Actual End Date"]:
        df[c + "_dt"] = pd.to_datetime(df[c], errors="coerce")
    df["start_year"] = df["Project Phase Actual Start Date_dt"].dt.year
    df["planned_dur_days"] = (df["Project Phase Planned End Date_dt"] -
                              df["Project Phase Actual Start Date_dt"]).dt.days
    df["n_phases_bldg"] = df.groupby("Project Building Identifier")["Project Building Identifier"].transform("count")
    df = df.rename(columns={"Project Type": "program_type", "Project Phase Name": "phase",
                            "Project Geographic District": "district", "Project Status Name": "status"})
    return df


SCOPE_KW = ["boiler", "roof", "window", "electr", "ventil", "floor", "ceil", "abate",
            "parapet", "mason", "heat", "cool", "fire", "alarm", "bath", "cafet",
            "gym", "scienc", "lab", "air", "tile", "plaster", "wall", "play"]


def load_nyc_sca_building(min_cost=10000):
    """NYC SCA building-level aggregation: one row per school building.
    Y = max_cost (total cost estimate: the maximum Final Estimate across the building's phases ~= project total cost).
    Features: district/n_phases/n_types/dur_days/start_year + work-scope keywords (from the description).
    """
    df = load_nyc_sca_raw()
    b = df.groupby("Project Building Identifier").agg(
        max_cost=("cost_estimate", "max"), n_phases=("phase", "count"),
        district=("district", "first"), n_types=("program_type", "nunique"),
        start=("Project Phase Actual Start Date_dt", "min"),
        end=("Project Phase Planned End Date_dt", "max"),
        desc=("Project Description", lambda s: " ".join(s.dropna().astype(str))))
    b["dur_days"] = (b["end"] - b["start"]).dt.days
    b["start_year"] = b["start"].dt.year
    b = b[b["max_cost"] >= min_cost].copy()
    desc = b["desc"].str.lower()
    for kw in SCOPE_KW:
        b[f"scope_{kw}"] = desc.str.contains(kw).astype(int)
    for c in ["dur_days", "start_year", "n_phases", "n_types"]:
        b[c] = pd.to_numeric(b[c], errors="coerce")
        b[c] = b[c].fillna(b[c].median())
    return b.reset_index(drop=True)


def load_comstock_public():
    """ComStock public-building subset (real design features, no cost). Returns a DataFrame."""
    p = ROOT / "dataset" / "raw" / "comstock" / "baseline_metadata_only.parquet"
    cs = pd.read_parquet(p)
    PB = ["PrimarySchool", "SecondarySchool", "SmallOffice", "MediumOffice", "LargeOffice",
          "Outpatient", "Hospital", "LargeHotel", "SmallHotel", "RetailStandalone"]
    return cs[cs["in.comstock_building_type"].isin(PB)].copy()


if __name__ == "__main__":
    d = load_uci()
    print(f"UCI: n={d['n']}, X={d['X'].shape}, y_cost range=[{d['y_cost'].min():.1f}, {d['y_cost'].max():.1f}]")
    nyc = load_nyc_sca_raw()
    print(f"NYC SCA: n={len(nyc)}, cost_estimate>0: {(nyc['cost_estimate']>0).sum()}")
    cs = load_comstock_public()
    print(f"ComStock public: n={len(cs)}, cols={len(cs.columns)}")
