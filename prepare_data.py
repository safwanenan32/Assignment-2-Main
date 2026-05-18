from pathlib import Path
import pandas as pd
import numpy as np
import re

# ---------------------------------------------------------
# 1. File paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "raw_data"
DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(exist_ok=True)

WEATHER_FILE = RAW_DIR / "weatherAUS.csv"
CITY_FILE = RAW_DIR / "au.csv"

# ---------------------------------------------------------
# 2. Load datasets
# ---------------------------------------------------------

weather = pd.read_csv(WEATHER_FILE)
cities = pd.read_csv(CITY_FILE)

print("Weather dataset shape:", weather.shape)
print("City dataset shape:", cities.shape)

# ---------------------------------------------------------
# 3. Helper function to clean names for matching
# ---------------------------------------------------------

def clean_name(value):
    """
    Makes names easier to match.
    Example:
    GoldCoast -> goldcoast
    Gold Coast -> goldcoast
    Alice Springs -> alicesprings
    """
    if pd.isna(value):
        return ""
    value = str(value).lower()
    value = re.sub(r"[^a-z0-9]", "", value)
    return value

# ---------------------------------------------------------
# 4. Manual match table
# ---------------------------------------------------------
# Left side = Location name in weatherAUS
# Right side = city name in au.csv
#
# Some weather station names are not written like normal city names.
# Example: GoldCoast in weatherAUS is Gold Coast in au.csv.
# Airport/weather station names are matched to the nearest major city
# for mapping and population context.

manual_match = {
    "AliceSprings": "Alice Springs",
    "CoffsHarbour": "Coffs Harbour",
    "GoldCoast": "Gold Coast",
    "MountGambier": "Mount Gambier",
    "WaggaWagga": "Wagga Wagga",

    # Airport/station names matched to closest major city in au.csv
    "MelbourneAirport": "Melbourne",
    "SydneyAirport": "Sydney",
    "PerthAirport": "Perth",
    "Tuggeranong": "Canberra",
    "Watsonia": "Melbourne",
    "Williamtown": "Newcastle",
    "PearceRAAF": "Perth",

    # Some special/remote stations
    "Uluru": "Yulara",
    "Nhil": "Nhill"
}

# ---------------------------------------------------------
# 5. Prepare city dataset for joining
# ---------------------------------------------------------

cities = cities.copy()

cities["city_clean"] = cities["city"].apply(clean_name)

# Keep only the columns we need from au.csv
cities_small = cities[
    [
        "city",
        "city_clean",
        "lat",
        "lng",
        "admin_name",
        "population",
        "population_proper"
    ]
].copy()

# If duplicate city names exist, keep the first one
cities_small = cities_small.drop_duplicates(subset=["city_clean"], keep="first")

# ---------------------------------------------------------
# 6. Create location match table
# ---------------------------------------------------------

weather_locations = sorted(weather["Location"].dropna().unique())

match_rows = []

for loc in weather_locations:
    # Use manual match if available, otherwise use original location name
    matched_city = manual_match.get(loc, loc)

    match_rows.append({
        "Location": loc,
        "MatchedCity": matched_city,
        "LocationClean": clean_name(loc),
        "MatchedCityClean": clean_name(matched_city),
        "ManualMatchUsed": loc in manual_match
    })

match_table = pd.DataFrame(match_rows)

# Join match table with city coordinate/population data
match_table = match_table.merge(
    cities_small,
    left_on="MatchedCityClean",
    right_on="city_clean",
    how="left"
)

# Rename columns clearly
match_table = match_table.rename(columns={
    "city": "CityFromAU",
    "lat": "Latitude",
    "lng": "Longitude",
    "admin_name": "State",
    "population": "Population",
    "population_proper": "PopulationProper"
})

# Add match status
match_table["MatchStatus"] = np.where(
    match_table["Latitude"].notna() & match_table["Longitude"].notna(),
    "Matched",
    "Needs review"
)

# Save full match table
match_table.to_csv(DATA_DIR / "location_match_table.csv", index=False)

# Save unmatched locations separately
unmatched = match_table[match_table["MatchStatus"] == "Needs review"]
unmatched.to_csv(DATA_DIR / "unmatched_locations_for_review.csv", index=False)

print("Match table saved.")
print("Matched locations:", (match_table["MatchStatus"] == "Matched").sum())
print("Unmatched locations:", (match_table["MatchStatus"] == "Needs review").sum())

# ---------------------------------------------------------
# 7. Clean weather dataset
# ---------------------------------------------------------

weather = weather.copy()

# Convert Date column
weather["Date"] = pd.to_datetime(weather["Date"], errors="coerce")

# Create time fields
weather["Year"] = weather["Date"].dt.year
weather["Month"] = weather["Date"].dt.month
weather["MonthName"] = weather["Date"].dt.strftime("%b")

# Create season field for Australia
def get_season(month):
    if month in [12, 1, 2]:
        return "Summer"
    elif month in [3, 4, 5]:
        return "Autumn"
    elif month in [6, 7, 8]:
        return "Winter"
    elif month in [9, 10, 11]:
        return "Spring"
    return np.nan

weather["Season"] = weather["Month"].apply(get_season)

# Convert rain fields to binary values
weather["RainTodayBinary"] = weather["RainToday"].map({"Yes": 1, "No": 0})
weather["RainTomorrowBinary"] = weather["RainTomorrow"].map({"Yes": 1, "No": 0})

# Rainfall categories
weather["RainyDay"] = np.where(weather["Rainfall"] > 0, 1, 0)
weather["HeavyRainDay"] = np.where(weather["Rainfall"] >= 10, 1, 0)

# Extra derived fields
weather["TempRange"] = weather["MaxTemp"] - weather["MinTemp"]
weather["PressureDrop"] = weather["Pressure9am"] - weather["Pressure3pm"]
weather["HumidityChange"] = weather["Humidity3pm"] - weather["Humidity9am"]

# ---------------------------------------------------------
# 8. Add city/state/coordinate data to weather rows
# ---------------------------------------------------------

weather = weather.merge(
    match_table[
        [
            "Location",
            "MatchedCity",
            "CityFromAU",
            "Latitude",
            "Longitude",
            "State",
            "Population",
            "PopulationProper",
            "MatchStatus"
        ]
    ],
    on="Location",
    how="left"
)

# ---------------------------------------------------------
# 9. Create rainfall_location_summary.csv
# ---------------------------------------------------------

location_summary = weather.groupby(
    [
        "Location",
        "MatchedCity",
        "CityFromAU",
        "State",
        "Latitude",
        "Longitude",
        "Population",
        "PopulationProper",
        "MatchStatus"
    ],
    dropna=False
).agg(
    Records=("Location", "count"),
    AvgRainfall=("Rainfall", "mean"),
    TotalRainfall=("Rainfall", "sum"),
    RainyDays=("RainyDay", "sum"),
    HeavyRainDays=("HeavyRainDay", "sum"),
    RainyDayRate=("RainyDay", "mean"),
    HeavyRainRate=("HeavyRainDay", "mean"),
    RainTomorrowRate=("RainTomorrowBinary", "mean"),
    AvgMinTemp=("MinTemp", "mean"),
    AvgMaxTemp=("MaxTemp", "mean"),
    AvgHumidity3pm=("Humidity3pm", "mean"),
    AvgPressure3pm=("Pressure3pm", "mean"),
    AvgWindGustSpeed=("WindGustSpeed", "mean")
).reset_index()

# Convert rates to percentages
location_summary["RainyDayRate"] = location_summary["RainyDayRate"] * 100
location_summary["HeavyRainRate"] = location_summary["HeavyRainRate"] * 100
location_summary["RainTomorrowRate"] = location_summary["RainTomorrowRate"] * 100

# Round numbers for smaller files and cleaner tooltips
numeric_cols = location_summary.select_dtypes(include=[np.number]).columns
location_summary[numeric_cols] = location_summary[numeric_cols].round(2)

location_summary.to_csv(DATA_DIR / "rainfall_location_summary.csv", index=False)

print("rainfall_location_summary.csv saved.")

# ---------------------------------------------------------
# 10. Create rainfall_monthly_summary.csv
# ---------------------------------------------------------

monthly_summary = weather.groupby(
    ["Location", "State", "Month", "MonthName"],
    dropna=False
).agg(
    AvgRainfall=("Rainfall", "mean"),
    TotalRainfall=("Rainfall", "sum"),
    RainyDayRate=("RainyDay", "mean"),
    RainTomorrowRate=("RainTomorrowBinary", "mean"),
    Records=("Location", "count")
).reset_index()

monthly_summary["RainyDayRate"] = monthly_summary["RainyDayRate"] * 100
monthly_summary["RainTomorrowRate"] = monthly_summary["RainTomorrowRate"] * 100

numeric_cols = monthly_summary.select_dtypes(include=[np.number]).columns
monthly_summary[numeric_cols] = monthly_summary[numeric_cols].round(2)

monthly_summary.to_csv(DATA_DIR / "rainfall_monthly_summary.csv", index=False)

print("rainfall_monthly_summary.csv saved.")

# ---------------------------------------------------------
# 11. Create rainfall_yearly_summary.csv
# ---------------------------------------------------------

yearly_summary = weather.groupby(
    ["Year", "State"],
    dropna=False
).agg(
    AvgRainfall=("Rainfall", "mean"),
    TotalRainfall=("Rainfall", "sum"),
    RainyDayRate=("RainyDay", "mean"),
    HeavyRainRate=("HeavyRainDay", "mean"),
    RainTomorrowRate=("RainTomorrowBinary", "mean"),
    Records=("Location", "count")
).reset_index()

yearly_summary["RainyDayRate"] = yearly_summary["RainyDayRate"] * 100
yearly_summary["HeavyRainRate"] = yearly_summary["HeavyRainRate"] * 100
yearly_summary["RainTomorrowRate"] = yearly_summary["RainTomorrowRate"] * 100

numeric_cols = yearly_summary.select_dtypes(include=[np.number]).columns
yearly_summary[numeric_cols] = yearly_summary[numeric_cols].round(2)

yearly_summary.to_csv(DATA_DIR / "rainfall_yearly_summary.csv", index=False)

print("rainfall_yearly_summary.csv saved.")

# ---------------------------------------------------------
# 12. Create rainfall_predictor_summary.csv
# ---------------------------------------------------------

predictor_frames = []

def make_predictor_summary(df, factor, bins):
    temp = df[[factor, "RainTomorrowBinary"]].dropna().copy()

    temp["Bin"] = pd.cut(
        temp[factor],
        bins=bins,
        include_lowest=True
    )

    summary = temp.groupby("Bin", observed=True).agg(
        RainTomorrowRate=("RainTomorrowBinary", "mean"),
        Count=("RainTomorrowBinary", "count")
    ).reset_index()

    summary["Factor"] = factor
    summary["Bin"] = summary["Bin"].astype(str)
    summary["RainTomorrowRate"] = summary["RainTomorrowRate"] * 100

    return summary[["Factor", "Bin", "RainTomorrowRate", "Count"]]

# Bins chosen to keep charts simple for general audience
predictor_frames.append(make_predictor_summary(weather, "Humidity3pm", [0, 20, 40, 60, 80, 100]))
predictor_frames.append(make_predictor_summary(weather, "Pressure3pm", [980, 990, 1000, 1010, 1020, 1030, 1040]))
predictor_frames.append(make_predictor_summary(weather, "Cloud3pm", [0, 2, 4, 6, 8, 10]))
predictor_frames.append(make_predictor_summary(weather, "WindGustSpeed", [0, 20, 40, 60, 80, 100, 140]))

predictor_summary = pd.concat(predictor_frames, ignore_index=True)

numeric_cols = predictor_summary.select_dtypes(include=[np.number]).columns
predictor_summary[numeric_cols] = predictor_summary[numeric_cols].round(2)

predictor_summary.to_csv(DATA_DIR / "rainfall_predictor_summary.csv", index=False)

print("rainfall_predictor_summary.csv saved.")

# ---------------------------------------------------------
# 13. Create rainfall_wind_direction_summary.csv
# ---------------------------------------------------------

wind_summary = weather.dropna(subset=["WindGustDir"]).groupby(
    "WindGustDir"
).agg(
    AvgRainfall=("Rainfall", "mean"),
    RainTomorrowRate=("RainTomorrowBinary", "mean"),
    Count=("WindGustDir", "count")
).reset_index()

wind_summary["RainTomorrowRate"] = wind_summary["RainTomorrowRate"] * 100

# Direction order for chart sorting
direction_order = {
    "N": 1,
    "NNE": 2,
    "NE": 3,
    "ENE": 4,
    "E": 5,
    "ESE": 6,
    "SE": 7,
    "SSE": 8,
    "S": 9,
    "SSW": 10,
    "SW": 11,
    "WSW": 12,
    "W": 13,
    "WNW": 14,
    "NW": 15,
    "NNW": 16
}

wind_summary["DirectionOrder"] = wind_summary["WindGustDir"].map(direction_order)

numeric_cols = wind_summary.select_dtypes(include=[np.number]).columns
wind_summary[numeric_cols] = wind_summary[numeric_cols].round(2)

wind_summary = wind_summary.sort_values("DirectionOrder")

wind_summary.to_csv(DATA_DIR / "rainfall_wind_direction_summary.csv", index=False)

print("rainfall_wind_direction_summary.csv saved.")

# ---------------------------------------------------------
# 14. Create rainfall_profile_cards.csv
# ---------------------------------------------------------

profile = location_summary.copy()

def assign_rain_category(row):
    if row["RainyDayRate"] >= 35 and row["AvgRainfall"] >= 2:
        return "Frequently wet"
    elif row["HeavyRainRate"] >= 8:
        return "Heavy-rain prone"
    elif row["RainyDayRate"] < 20 and row["AvgRainfall"] < 1.5:
        return "Mostly dry"
    elif row["RainTomorrowRate"] >= 30:
        return "Rain tomorrow likely"
    else:
        return "Moderate rainfall"

profile["RainCategory"] = profile.apply(assign_rain_category, axis=1)

profile["KeyMessage"] = profile.apply(
    lambda row: f"{row['Location']} is classified as {row['RainCategory'].lower()} based on rainfall frequency, average rainfall, and rain tomorrow rate.",
    axis=1
)

profile_cards = profile[
    [
        "Location",
        "State",
        "RainCategory",
        "AvgRainfall",
        "RainyDayRate",
        "HeavyRainRate",
        "RainTomorrowRate",
        "Population",
        "KeyMessage"
    ]
].copy()

# Keep a manageable number for final visual cards
profile_cards = profile_cards.sort_values(
    ["RainTomorrowRate", "AvgRainfall"],
    ascending=False
).head(12)

profile_cards.to_csv(DATA_DIR / "rainfall_profile_cards.csv", index=False)

print("rainfall_profile_cards.csv saved.")

# ---------------------------------------------------------
# 15. Final checks
# ---------------------------------------------------------

print("\nFinished creating datasets.")
print("Files saved inside:", DATA_DIR)

print("\nGenerated files:")
for file in DATA_DIR.glob("*.csv"):
    size_kb = file.stat().st_size / 1024
    print(f"- {file.name}: {size_kb:.1f} KB")