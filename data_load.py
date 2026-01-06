import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re



OUTPUT_DIR = "/Users/joshua.stanley/Desktop/Tourism_vs_Trafficking_Graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def safe_filename(name):
    
    return re.sub(r'[^\w\-_. ]', '_', name)

# =======================
# LOAD DATA
# =======================

df_traffic = pd.read_csv(
    "/Users/joshua.stanley/Desktop/TSA/DataScience/finaldataset.csv",
    encoding="latin1",
    engine="python",
    sep=",",
    on_bad_lines="skip"
)

df_tourism = pd.read_csv(
    "/Users/joshua.stanley/Desktop/TSA/DataScience/finaltourism.csv",
    encoding="latin1",
    engine="python",
    sep=",",
    on_bad_lines="skip"
)

TRAFFICKING_COUNTRY_COL = "country"
TRAFFICKING_YEAR_COL = "yearOfRegistration"
TOURISM_COUNTRY_COL = "Country Code"


TOURISM_YEARS = sorted([
    int(c) for c in df_tourism.columns
    if str(c).isdigit() and 1950 <= int(c) <= 2025
])

TRAFFICKING_YEARS = sorted(
    df_traffic[TRAFFICKING_YEAR_COL]
    .dropna()
    .astype(int)
    .unique()
)

YEARS = sorted(set(TOURISM_YEARS) & set(TRAFFICKING_YEARS))

print("Tourism year range:",
      f"{min(TOURISM_YEARS)} → {max(TOURISM_YEARS)}" if TOURISM_YEARS else "NONE")
print("Trafficking year range:",
      f"{min(TRAFFICKING_YEARS)} → {max(TRAFFICKING_YEARS)}")
print("Using overlapping years:", YEARS)

if not YEARS:
    print("\n❌ NO OVERLAPPING YEARS — same-year comparison is impossible.\n")
    exit()

#part 2
def yearly_frequency_tourism(df, country):
    row = df[df[TOURISM_COUNTRY_COL] == country]

    if row.empty:
        return np.zeros(len(YEARS)), np.ones(len(YEARS), dtype=bool)

    values = []
    nan_mask = []

    for y in YEARS:
        col = str(y)
        if col in df.columns:
            v = row[col].iloc[0]
            if pd.isna(v):
                values.append(0)
                nan_mask.append(True)
            else:
                values.append(float(v))
                nan_mask.append(False)
        else:
            values.append(0)
            nan_mask.append(True)

    return np.array(values), np.array(nan_mask)


def yearly_frequency_traffic(df, country):
    subset = df[df[TRAFFICKING_COUNTRY_COL] == country]
    if subset.empty:
        return np.zeros(len(YEARS))
    return np.array([(subset[TRAFFICKING_YEAR_COL] == y).sum() for y in YEARS])


def normalize(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or np.all(arr == 0):
        return arr
    min_v, max_v = np.nanmin(arr), np.nanmax(arr)
    if min_v == max_v:
        return arr
    return (arr - min_v) / (max_v - min_v)


traffic_countries = set(df_traffic[TRAFFICKING_COUNTRY_COL].dropna().unique())
tourism_countries = set(df_tourism[TOURISM_COUNTRY_COL].dropna().unique())
common_countries = sorted(traffic_countries & tourism_countries)

print(f"\nFound {len(common_countries)} countries with both datasets\n")


total_correlation = 0
valid_countries = 0

for country in common_countries:

    tourism_raw, tourism_nan_mask = yearly_frequency_tourism(df_tourism, country)
    trafficking_raw = yearly_frequency_traffic(df_traffic, country)

    if (
        tourism_raw.size != len(YEARS)
        or trafficking_raw.size != len(YEARS)
        or np.all(tourism_raw == 0)
        or np.all(trafficking_raw == 0)
    ):
        continue

    tourism_norm = normalize(tourism_raw)
    trafficking_norm = normalize(trafficking_raw)

    corr = np.corrcoef(tourism_norm, trafficking_norm)[0, 1]

    print("=" * 70)
    print(f"COUNTRY: {country}")
    print("-" * 70)
    print(f"Tourism total: {tourism_raw.sum():.0f}")
    print(f"Trafficking total: {trafficking_raw.sum():.0f}")
    print(f"Pearson correlation: {corr:.3f}")

    total_correlation += corr
    valid_countries += 1

    #pyplot

    plt.figure(figsize=(10, 5))

    plt.plot(YEARS, tourism_norm, label="Tourism (normalized)", linewidth=2)
    plt.plot(YEARS, trafficking_norm, label="Trafficking (normalized)", linewidth=2)

    for year, is_nan in zip(YEARS, tourism_nan_mask):
        if is_nan:
            plt.axvspan(year - 0.5, year + 0.5, color="red", alpha=0.2)

    plt.title(
        f"Tourism vs Trafficking — {country}\n"
        f"Pearson r = {corr:.3f}"
    )
    plt.xlabel("Year")
    plt.ylabel("Relative intensity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    

    filename = f"{safe_filename(country)}_tourism_vs_trafficking.png"
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
#stats 
if valid_countries == 0:
    print("\n❌ No valid country comparisons found.")
else:
    print(
        f"\n✅ Average Pearson correlation across "
        f"{valid_countries} countries: "
        f"{total_correlation / valid_countries:.3f}"
    )
    print(f"\n📁 All graphs saved to:\n{OUTPUT_DIR}")
