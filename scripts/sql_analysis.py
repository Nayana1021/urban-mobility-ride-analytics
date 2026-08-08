"""
Bangalore Ride Booking Analysis
--------------------------------
End-to-end Python + SQLite SQL analysis script.

This script:
1. Loads the raw Excel dataset.
2. Audits data quality.
3. Cleans and feature-engineers the dataset.
4. Saves the cleaned dataset.
5. Creates a SQLite database and bookings table.
6. Runs the SQL analyses used in the project.
7. Performs Python-side calculations based on SQL/Pandas results.
8. Generates selected analytical charts.

Recommended project structure:

urban-mobility-ride-analytics/
├── data/
│   ├── raw/
│   │   └── bookings_raw.xlsx
│   └── processed/
├── scripts/
│   └── sql_analysis.py
├── notebooks/
│   └── ride_booking_analysis.ipynb
└── powerbi/
    └── Bangalore_Ride_Booking_Analysis.pbix
"""

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 0. PROJECT PATHS & SETTINGS
# ============================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "bookings_raw.xlsx"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_FILE = PROCESSED_DIR / "bookings_clean.csv"
DATABASE_FILE = PROCESSED_DIR / "urban_mobility.db"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("BANGALORE RIDE BOOKING ANALYSIS")
print("=" * 70)
print(f"Project root: {PROJECT_ROOT}")
print(f"Raw file:     {RAW_FILE}")


# ============================================================
# 1. LOAD DATA
# ============================================================

if not RAW_FILE.exists():
    raise FileNotFoundError(
        f"Raw dataset not found: {RAW_FILE}\n"
        "Place bookings_raw.xlsx inside data/raw/."
    )

df = pd.read_excel(RAW_FILE)

print("\nDataset loaded successfully.")
print(f"Shape: {df.shape}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")


# ============================================================
# 2. DATASET AUDIT
# ============================================================

print("\n" + "=" * 70)
print("DATASET AUDIT")
print("=" * 70)

print("\nColumn names:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2}. {col}")

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicate rows:")
print(f"{df.duplicated().sum():,}")

print("\nFirst 5 rows:")
print(df.head())

audit_table = pd.DataFrame(
    {
        "Column": df.columns,
        "Data_Type": df.dtypes.astype(str).values,
        "Missing_Count": df.isna().sum().values,
        "Missing_%": (df.isna().mean() * 100).round(2).values,
        "Unique_Values": df.nunique().values,
    }
)

print("\nColumn-level audit:")
print(audit_table.to_string(index=False))

print("\nNumerical summary:")
print(df.describe().T)

if "Booking_Status" in df.columns:
    print("\nBooking status:")
    print(df["Booking_Status"].value_counts(dropna=False))

if "Vehicle_Type" in df.columns:
    print("\nVehicle type:")
    print(df["Vehicle_Type"].value_counts(dropna=False))

if "Payment_Method" in df.columns:
    print("\nPayment method:")
    print(df["Payment_Method"].value_counts(dropna=False))

if "Date" in df.columns:
    print("\nDate range:")
    print("Minimum date:", df["Date"].min())
    print("Maximum date:", df["Date"].max())


# ============================================================
# 3. DATA QUALITY INVESTIGATION
# ============================================================

print("\n" + "=" * 70)
print("DATA QUALITY INVESTIGATION")
print("=" * 70)

print("\nBooking status distribution:")
print(df["Booking_Status"].value_counts(dropna=False))

payment_missing = (
    df.assign(Payment_Missing=df["Payment_Method"].isna())
    .groupby("Booking_Status")["Payment_Missing"]
    .agg(["count", "sum", "mean"])
)

payment_missing["Missing_%"] = (payment_missing["mean"] * 100).round(2)

print("\nPayment method missingness by booking status:")
print(payment_missing)

print("\nCancellation and incomplete-ride fields:")
for col in [
    "Canceled_Rides_by_Customer",
    "Canceled_Rides_by_Driver",
    "Incomplete_Rides",
    "Incomplete_Rides_Reason",
]:
    if col in df.columns:
        print(f"\n{col}")
        print(df[col].value_counts(dropna=False).head(15))

numeric_cols = df.select_dtypes(include=np.number).columns

missing_numeric = pd.DataFrame(
    {
        "Missing_Count": df[numeric_cols].isna().sum(),
        "Missing_%": (df[numeric_cols].isna().mean() * 100).round(2),
    }
).sort_values("Missing_Count", ascending=False)

print("\nNumerical/KPI missingness:")
print(missing_numeric)

print("\nLocation coverage:")
for col in ["Pickup_Location", "Drop_Location"]:
    if col in df.columns:
        print(f"\n{col}: {df[col].nunique()} unique locations")
        print(df[col].value_counts().head(15))


# ============================================================
# 4. DATA CLEANING & FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 70)
print("DATA CLEANING & FEATURE ENGINEERING")
print("=" * 70)

clean_df = df.copy()

# Standardise column names.
clean_df.columns = (
    clean_df.columns
    .str.strip()
    .str.replace(" ", "_", regex=False)
)

# Standardise text values by removing surrounding whitespace.
text_columns = clean_df.select_dtypes(include="object").columns

for col in text_columns:
    clean_df[col] = clean_df[col].apply(
        lambda x: x.strip() if isinstance(x, str) else x
    )

# Convert date and time fields.
clean_df["Date"] = pd.to_datetime(
    clean_df["Date"],
    errors="coerce",
)

clean_df["Time"] = pd.to_datetime(
    clean_df["Time"].astype(str),
    errors="coerce",
).dt.time

# Create date/time dimensions.
clean_df["Day"] = clean_df["Date"].dt.day
clean_df["Day_Name"] = clean_df["Date"].dt.day_name()
clean_df["Week"] = clean_df["Date"].dt.isocalendar().week.astype(int)
clean_df["Month"] = clean_df["Date"].dt.month

clean_df["Hour"] = pd.to_datetime(
    clean_df["Time"].astype(str),
    errors="coerce",
).dt.hour

# Convert analytical numeric columns.
numeric_columns = [
    "V_TAT",
    "C_TAT",
    "Booking_Value",
    "Ride_Distance",
    "Driver_Ratings",
    "Customer_Rating",
]

for col in numeric_columns:
    if col in clean_df.columns:
        clean_df[col] = pd.to_numeric(
            clean_df[col],
            errors="coerce",
        )

# Booking outcome flags.
clean_df["Is_Successful_Ride"] = (
    clean_df["Booking_Status"]
    .eq("Success")
    .astype(int)
)

clean_df["Is_Customer_Cancelled"] = (
    clean_df["Booking_Status"]
    .eq("Canceled by Customer")
    .astype(int)
)

clean_df["Is_Driver_Cancelled"] = (
    clean_df["Booking_Status"]
    .eq("Canceled by Driver")
    .astype(int)
)

clean_df["Is_Incomplete_Ride"] = (
    clean_df["Booking_Status"]
    .eq("Incomplete")
    .astype(int)
)

# Unified ride outcome.
clean_df["Ride_Outcome"] = np.select(
    [
        clean_df["Is_Successful_Ride"].eq(1),
        clean_df["Is_Customer_Cancelled"].eq(1),
        clean_df["Is_Driver_Cancelled"].eq(1),
        clean_df["Is_Incomplete_Ride"].eq(1),
    ],
    [
        "Completed",
        "Customer Cancelled",
        "Driver Cancelled",
        "Incomplete",
    ],
    default="Other",
)

# Revenue is realised only for successfully completed rides.
clean_df["Realised_Revenue"] = np.where(
    clean_df["Is_Successful_Ride"].eq(1),
    clean_df["Booking_Value"],
    0,
)

# Data-quality flags.
clean_df["Payment_Method_Missing"] = (
    clean_df["Payment_Method"].isna().astype(int)
)

clean_df["Customer_Rating_Missing"] = (
    clean_df["Customer_Rating"].isna().astype(int)
)

clean_df["Driver_Rating_Missing"] = (
    clean_df["Driver_Ratings"].isna().astype(int)
)

clean_df["Ride_Distance_Missing"] = (
    clean_df["Ride_Distance"].isna().astype(int)
)

clean_df["Ride_Distance_Valid"] = (
    clean_df["Ride_Distance"].gt(0).astype(int)
)

print("\nCleaning completed.")
print(f"Rows: {len(clean_df):,}")
print(f"Columns: {len(clean_df.columns):,}")


# ============================================================
# 5. CLEANING VALIDATION & EXPORT
# ============================================================

validation = pd.DataFrame(
    {
        "Metric": [
            "Raw Rows",
            "Clean Rows",
            "Raw Columns",
            "Clean Columns",
            "Duplicate Rows",
        ],
        "Value": [
            len(df),
            len(clean_df),
            len(df.columns),
            len(clean_df.columns),
            clean_df.duplicated().sum(),
        ],
    }
)

print("\nCleaning validation:")
print(validation.to_string(index=False))

print("\nRide outcome:")
print(clean_df["Ride_Outcome"].value_counts(dropna=False))

print("\nSuccessful ride rate:")
print(round(clean_df["Is_Successful_Ride"].mean() * 100, 2), "%")

print("\nRealised revenue:")
print(f"₹{clean_df['Realised_Revenue'].sum():,.0f}")

clean_df.to_csv(CLEAN_FILE, index=False)

print(f"\nCleaned dataset saved to: {CLEAN_FILE}")
print(
    f"File size: {CLEAN_FILE.stat().st_size / (1024 * 1024):.2f} MB"
)


# ============================================================
# 6. CREATE SQLITE DATABASE
# ============================================================

print("\n" + "=" * 70)
print("CREATE SQLITE DATABASE")
print("=" * 70)

conn = sqlite3.connect(DATABASE_FILE)

clean_df.to_sql(
    "bookings",
    conn,
    if_exists="replace",
    index=False,
)

print(f"SQLite database created: {DATABASE_FILE}")

tables = pd.read_sql_query(
    """
    SELECT name
    FROM sqlite_master
    WHERE type = 'table';
    """,
    conn,
)

print("\nDatabase tables:")
print(tables)

count = pd.read_sql_query(
    """
    SELECT COUNT(*) AS total_rows
    FROM bookings;
    """,
    conn,
)

print("\nSQL row count:")
print(count)


# ============================================================
# 7. SQL HELPER FUNCTION
# ============================================================

def run_query(query, name):
    """Run a SQL query against SQLite and return a DataFrame."""
    result = pd.read_sql_query(query, conn)
    print(f"\n--- {name} ---")
    print(result.to_string(index=False))
    return result


# ============================================================
# 8. OVERALL PERFORMANCE
# ============================================================

kpi_query = """
SELECT
    COUNT(*) AS total_bookings,
    SUM(Is_Successful_Ride) AS successful_rides,
    SUM(Is_Customer_Cancelled) AS customer_cancellations,
    SUM(Is_Driver_Cancelled) AS driver_cancellations,
    SUM(Is_Incomplete_Ride) AS incomplete_rides,

    ROUND(
        100.0 * SUM(Is_Successful_Ride) / COUNT(*),
        2
    ) AS success_rate

FROM bookings;
"""

kpi_result = run_query(
    kpi_query,
    "Overall Booking Performance",
)


# ============================================================
# 9. REVENUE ANALYSIS
# ============================================================

revenue_query = """
SELECT
    COUNT(*) AS total_bookings,
    SUM(Is_Successful_Ride) AS successful_rides,
    SUM(Realised_Revenue) AS realised_revenue,

    ROUND(
        AVG(
            CASE
                WHEN Is_Successful_Ride = 1
                THEN Booking_Value
            END
        ),
        2
    ) AS average_successful_booking_value

FROM bookings;
"""

revenue_result = run_query(
    revenue_query,
    "Revenue Analysis",
)


# ============================================================
# 10. VEHICLE TYPE PERFORMANCE
# ============================================================

vehicle_query = """
SELECT
    Vehicle_Type,
    COUNT(*) AS total_bookings,
    SUM(Is_Successful_Ride) AS successful_rides,

    ROUND(
        100.0 * SUM(Is_Successful_Ride) / COUNT(*),
        2
    ) AS success_rate,

    SUM(Realised_Revenue) AS realised_revenue,

    ROUND(
        AVG(
            CASE
                WHEN Is_Successful_Ride = 1
                THEN Booking_Value
            END
        ),
        2
    ) AS avg_booking_value

FROM bookings

GROUP BY Vehicle_Type

ORDER BY realised_revenue DESC;
"""

vehicle_result = run_query(
    vehicle_query,
    "Vehicle Type Performance",
)


# ============================================================
# 11. BOOKING STATUS DISTRIBUTION
# ============================================================

status_query = """
SELECT
    Booking_Status,
    COUNT(*) AS total_bookings,

    ROUND(
        100.0 * COUNT(*) /
        (SELECT COUNT(*) FROM bookings),
        2
    ) AS percentage

FROM bookings

GROUP BY Booking_Status

ORDER BY total_bookings DESC;
"""

cancellation_result = run_query(
    status_query,
    "Booking Status Distribution",
)


# ============================================================
# 12. DRIVER CANCELLATION REASONS
# ============================================================

driver_cancel_query = """
SELECT
    Canceled_Rides_by_Driver AS cancellation_reason,
    COUNT(*) AS total_cancellations

FROM bookings

WHERE Is_Driver_Cancelled = 1
  AND Canceled_Rides_by_Driver IS NOT NULL

GROUP BY Canceled_Rides_by_Driver

ORDER BY total_cancellations DESC;
"""

driver_cancel_result = run_query(
    driver_cancel_query,
    "Driver Cancellation Reasons",
)


# ============================================================
# 13. CUSTOMER CANCELLATION REASONS
# ============================================================

customer_cancel_query = """
SELECT
    Canceled_Rides_by_Customer AS cancellation_reason,
    COUNT(*) AS total_cancellations

FROM bookings

WHERE Is_Customer_Cancelled = 1
  AND Canceled_Rides_by_Customer IS NOT NULL

GROUP BY Canceled_Rides_by_Customer

ORDER BY total_cancellations DESC;
"""

customer_cancel_result = run_query(
    customer_cancel_query,
    "Customer Cancellation Reasons",
)


# ============================================================
# 14. TOP PICKUP LOCATIONS BY BOOKING VOLUME
# ============================================================

pickup_query = """
SELECT
    Pickup_Location,
    COUNT(*) AS total_bookings,
    SUM(Is_Successful_Ride) AS successful_rides,

    ROUND(
        100.0 * SUM(Is_Successful_Ride) / COUNT(*),
        2
    ) AS success_rate,

    SUM(Realised_Revenue) AS realised_revenue

FROM bookings

GROUP BY Pickup_Location

ORDER BY total_bookings DESC

LIMIT 15;
"""

pickup_result = run_query(
    pickup_query,
    "Top Pickup Locations by Booking Volume",
)


# ============================================================
# 15. HOURLY DEMAND
# ============================================================

hour_query = """
SELECT
    Hour,
    COUNT(*) AS total_bookings,
    SUM(Is_Successful_Ride) AS successful_rides,

    ROUND(
        100.0 * SUM(Is_Successful_Ride) / COUNT(*),
        2
    ) AS success_rate,

    SUM(Realised_Revenue) AS realised_revenue

FROM bookings

GROUP BY Hour

ORDER BY Hour;
"""

hour_result = run_query(
    hour_query,
    "Hourly Booking Demand",
)


# ============================================================
# 16. PAYMENT METHOD PERFORMANCE
# ============================================================

payment_query = """
SELECT
    Payment_Method,
    COUNT(*) AS total_bookings,
    SUM(Is_Successful_Ride) AS successful_rides,

    ROUND(
        100.0 * SUM(Is_Successful_Ride) / COUNT(*),
        2
    ) AS success_rate,

    SUM(Realised_Revenue) AS realised_revenue

FROM bookings

WHERE Payment_Method IS NOT NULL

GROUP BY Payment_Method

ORDER BY total_bookings DESC;
"""

payment_result = run_query(
    payment_query,
    "Payment Method Performance",
)


# ============================================================
# 17. LOCATION PERFORMANCE
# ============================================================

location_query = """
SELECT
    Pickup_Location,
    COUNT(*) AS total_bookings,
    SUM(Is_Successful_Ride) AS successful_rides,
    SUM(Is_Customer_Cancelled) AS customer_cancellations,
    SUM(Is_Driver_Cancelled) AS driver_cancellations,
    SUM(Is_Incomplete_Ride) AS incomplete_rides,

    ROUND(
        100.0 * SUM(Is_Successful_Ride) / COUNT(*),
        2
    ) AS success_rate

FROM bookings

GROUP BY Pickup_Location

ORDER BY total_bookings DESC

LIMIT 15;
"""

location_result = run_query(
    location_query,
    "Location Performance",
)


# ============================================================
# 18. CANCELLATION RATE BY PICKUP LOCATION
# ============================================================

cancellation_location_query = """
SELECT
    Pickup_Location,
    COUNT(*) AS total_bookings,

    ROUND(
        100.0 * SUM(Is_Customer_Cancelled) / COUNT(*),
        2
    ) AS customer_cancellation_rate,

    ROUND(
        100.0 * SUM(Is_Driver_Cancelled) / COUNT(*),
        2
    ) AS driver_cancellation_rate,

    ROUND(
        100.0 * SUM(Is_Successful_Ride) / COUNT(*),
        2
    ) AS success_rate

FROM bookings

GROUP BY Pickup_Location

ORDER BY driver_cancellation_rate DESC

LIMIT 15;
"""

cancellation_location_result = run_query(
    cancellation_location_query,
    "Cancellation Rate by Pickup Location",
)


# ============================================================
# 19. HOURLY RIDE PERFORMANCE
# ============================================================

hour_performance_query = """
SELECT
    Hour,
    COUNT(*) AS total_bookings,
    SUM(Is_Successful_Ride) AS successful_rides,

    ROUND(
        100.0 * SUM(Is_Customer_Cancelled) / COUNT(*),
        2
    ) AS customer_cancellation_rate,

    ROUND(
        100.0 * SUM(Is_Driver_Cancelled) / COUNT(*),
        2
    ) AS driver_cancellation_rate,

    ROUND(
        100.0 * SUM(Is_Successful_Ride) / COUNT(*),
        2
    ) AS success_rate

FROM bookings

GROUP BY Hour

ORDER BY Hour;
"""

hourly_performance_result = run_query(
    hour_performance_query,
    "Hourly Ride Performance",
)


# ============================================================
# 20. DAY-OF-WEEK PERFORMANCE
# ============================================================

day_query = """
SELECT
    Day_Name,
    COUNT(*) AS total_bookings,
    SUM(Is_Successful_Ride) AS successful_rides,

    ROUND(
        100.0 * SUM(Is_Successful_Ride) / COUNT(*),
        2
    ) AS success_rate,

    SUM(Is_Driver_Cancelled) AS driver_cancellations,
    SUM(Is_Customer_Cancelled) AS customer_cancellations,

    ROUND(
        100.0 * SUM(Is_Driver_Cancelled) / COUNT(*),
        2
    ) AS driver_cancellation_rate,

    ROUND(
        100.0 * SUM(Is_Customer_Cancelled) / COUNT(*),
        2
    ) AS customer_cancellation_rate,

    SUM(Realised_Revenue) AS realised_revenue,

    ROUND(
        AVG(
            CASE
                WHEN Is_Successful_Ride = 1
                THEN Realised_Revenue
            END
        ),
        2
    ) AS avg_revenue_per_successful_ride

FROM bookings

GROUP BY Day_Name

ORDER BY
    CASE Day_Name
        WHEN 'Monday' THEN 1
        WHEN 'Tuesday' THEN 2
        WHEN 'Wednesday' THEN 3
        WHEN 'Thursday' THEN 4
        WHEN 'Friday' THEN 5
        WHEN 'Saturday' THEN 6
        WHEN 'Sunday' THEN 7
    END;
"""

day_result = run_query(
    day_query,
    "Day-of-Week Performance",
)


# ============================================================
# 21. VEHICLE × LOCATION PERFORMANCE
# ============================================================

vehicle_location_query = """
SELECT
    Vehicle_Type,
    Pickup_Location,
    COUNT(*) AS total_bookings,
    SUM(Is_Successful_Ride) AS successful_rides,
    SUM(Realised_Revenue) AS realised_revenue,

    ROUND(
        100.0 * SUM(Is_Successful_Ride) / COUNT(*),
        2
    ) AS success_rate

FROM bookings

GROUP BY
    Vehicle_Type,
    Pickup_Location

ORDER BY success_rate DESC;
"""

vehicle_location_result = run_query(
    vehicle_location_query,
    "Vehicle × Pickup Location Performance",
)

top_vehicle_location = (
    vehicle_location_result[
        vehicle_location_result["total_bookings"] >= 250
    ]
    .sort_values("success_rate", ascending=False)
    .head(10)
)

bottom_vehicle_location = (
    vehicle_location_result[
        vehicle_location_result["total_bookings"] >= 250
    ]
    .sort_values("success_rate", ascending=True)
    .head(10)
)

top_vehicle_location_avg = top_vehicle_location["success_rate"].mean()
bottom_vehicle_location_avg = bottom_vehicle_location["success_rate"].mean()

print("\nVehicle × Location comparison:")
print(
    f"Top 10 average success rate: "
    f"{top_vehicle_location_avg:.2f}%"
)
print(
    f"Bottom 10 average success rate: "
    f"{bottom_vehicle_location_avg:.2f}%"
)
print(
    f"Difference: "
    f"{top_vehicle_location_avg - bottom_vehicle_location_avg:.2f} "
    "percentage points"
)


# ============================================================
# 22. RIDE DISTANCE SUMMARY
# ============================================================

distance_summary_query = """
SELECT
    COUNT(Ride_Distance) AS rides_with_distance,

    ROUND(
        AVG(Ride_Distance),
        2
    ) AS avg_ride_distance,

    MIN(Ride_Distance) AS min_ride_distance,
    MAX(Ride_Distance) AS max_ride_distance,

    ROUND(
        AVG(
            CASE
                WHEN Is_Successful_Ride = 1
                THEN Ride_Distance
            END
        ),
        2
    ) AS avg_successful_ride_distance

FROM bookings

WHERE Ride_Distance_Valid = 1;
"""

distance_summary = run_query(
    distance_summary_query,
    "Overall Ride Distance Performance",
)


# ============================================================
# 23. RIDE DISTANCE BY RANGE
# ============================================================

distance_range_query = """
SELECT
    CASE
        WHEN Ride_Distance <= 5 THEN '0-5 km'
        WHEN Ride_Distance <= 10 THEN '5-10 km'
        WHEN Ride_Distance <= 20 THEN '10-20 km'
        WHEN Ride_Distance <= 30 THEN '20-30 km'
        ELSE '30+ km'
    END AS distance_range,

    COUNT(*) AS successful_rides,

    ROUND(
        AVG(Ride_Distance),
        2
    ) AS avg_distance,

    ROUND(
        AVG(Booking_Value),
        2
    ) AS avg_booking_value,

    SUM(Realised_Revenue) AS realised_revenue,

    ROUND(
        AVG(Realised_Revenue),
        2
    ) AS avg_revenue_per_ride

FROM bookings

WHERE Is_Successful_Ride = 1
  AND Ride_Distance_Valid = 1

GROUP BY distance_range

ORDER BY
    CASE distance_range
        WHEN '0-5 km' THEN 1
        WHEN '5-10 km' THEN 2
        WHEN '10-20 km' THEN 3
        WHEN '20-30 km' THEN 4
        ELSE 5
    END;
"""

distance_range_result = run_query(
    distance_range_query,
    "Ride Distance Range Analysis",
)


# ============================================================
# 24. RIDE DISTANCE VALIDITY CHECK
# ============================================================

distance_validity_query = """
SELECT
    Is_Successful_Ride,
    Ride_Distance_Valid,
    COUNT(*) AS total_bookings

FROM bookings

GROUP BY
    Is_Successful_Ride,
    Ride_Distance_Valid

ORDER BY
    Is_Successful_Ride,
    Ride_Distance_Valid;
"""

distance_validity_check = run_query(
    distance_validity_query,
    "Ride Distance Validity Check",
)


# ============================================================
# 25. RIDE DISTANCE BY VEHICLE TYPE
# ============================================================

vehicle_distance_query = """
SELECT
    Vehicle_Type,
    COUNT(*) AS successful_rides,

    ROUND(
        AVG(Ride_Distance),
        2
    ) AS avg_distance,

    ROUND(
        AVG(Booking_Value),
        2
    ) AS avg_booking_value,

    SUM(Realised_Revenue) AS realised_revenue,

    ROUND(
        AVG(Realised_Revenue),
        2
    ) AS avg_revenue_per_ride

FROM bookings

WHERE Is_Successful_Ride = 1
  AND Ride_Distance_Valid = 1

GROUP BY Vehicle_Type

ORDER BY realised_revenue DESC;
"""

vehicle_distance_result = run_query(
    vehicle_distance_query,
    "Ride Distance Performance by Vehicle Type",
)


# ============================================================
# 26. REVENUE EFFICIENCY BY VEHICLE TYPE
# ============================================================

revenue_efficiency_query = """
SELECT
    Vehicle_Type,
    COUNT(*) AS successful_rides,

    ROUND(
        AVG(Ride_Distance),
        2
    ) AS avg_distance,

    ROUND(
        AVG(Realised_Revenue),
        2
    ) AS avg_revenue_per_ride,

    ROUND(
        AVG(
            Realised_Revenue /
            NULLIF(Ride_Distance, 0)
        ),
        2
    ) AS avg_revenue_per_km,

    SUM(Realised_Revenue) AS total_revenue

FROM bookings

WHERE Is_Successful_Ride = 1
  AND Ride_Distance_Valid = 1
  AND Ride_Distance > 0

GROUP BY Vehicle_Type

ORDER BY avg_revenue_per_km DESC;
"""

revenue_efficiency_result = run_query(
    revenue_efficiency_query,
    "Revenue Efficiency by Vehicle Type",
)


# ============================================================
# 27. RIDE DISTANCE VS REVENUE
# ============================================================

distance_revenue_query = """
SELECT
    Ride_Distance,
    AVG(Realised_Revenue) AS avg_revenue,
    COUNT(*) AS total_rides

FROM bookings

WHERE Is_Successful_Ride = 1
  AND Ride_Distance_Valid = 1
  AND Ride_Distance > 0

GROUP BY Ride_Distance

ORDER BY Ride_Distance;
"""

distance_revenue_result = run_query(
    distance_revenue_query,
    "Ride Distance vs Revenue",
)


# ============================================================
# 28. DISTANCE RANGE REVENUE EFFICIENCY
# ============================================================

distance_efficiency_query = """
SELECT
    CASE
        WHEN Ride_Distance <= 5 THEN '0-5 km'
        WHEN Ride_Distance <= 10 THEN '5-10 km'
        WHEN Ride_Distance <= 20 THEN '10-20 km'
        WHEN Ride_Distance <= 30 THEN '20-30 km'
        ELSE '30+ km'
    END AS distance_range,

    COUNT(*) AS successful_rides,

    ROUND(
        SUM(Realised_Revenue),
        2
    ) AS total_revenue,

    SUM(Ride_Distance) AS total_distance,

    ROUND(
        SUM(Realised_Revenue) * 1.0 /
        NULLIF(SUM(Ride_Distance), 0),
        2
    ) AS revenue_per_km

FROM bookings

WHERE Is_Successful_Ride = 1
  AND Ride_Distance_Valid = 1
  AND Ride_Distance > 0

GROUP BY distance_range

ORDER BY
    CASE distance_range
        WHEN '0-5 km' THEN 1
        WHEN '5-10 km' THEN 2
        WHEN '10-20 km' THEN 3
        WHEN '20-30 km' THEN 4
        ELSE 5
    END;
"""

distance_efficiency_result = run_query(
    distance_efficiency_query,
    "Revenue Efficiency by Distance Range",
)


# ============================================================
# 29. RIDE OUTCOME ANALYSIS
# ============================================================

ride_outcome_query = """
SELECT
    Ride_Outcome,
    COUNT(*) AS total_bookings,

    ROUND(
        100.0 * COUNT(*) /
        (SELECT COUNT(*) FROM bookings),
        2
    ) AS percentage

FROM bookings

GROUP BY Ride_Outcome

ORDER BY total_bookings DESC;
"""

ride_outcome_result = run_query(
    ride_outcome_query,
    "Ride Outcome Analysis",
)


# ============================================================
# 30. FAILURE SUMMARY
# ============================================================

failure_summary_query = """
SELECT
    COUNT(*) AS total_bookings,
    SUM(Is_Successful_Ride) AS successful_rides,
    SUM(Is_Driver_Cancelled) AS driver_cancelled,
    SUM(Is_Customer_Cancelled) AS customer_cancelled,

    SUM(
        CASE
            WHEN Ride_Outcome = 'Other'
            THEN 1
            ELSE 0
        END
    ) AS other_failed_rides,

    ROUND(
        100.0 * SUM(Is_Successful_Ride) / COUNT(*),
        2
    ) AS success_rate,

    ROUND(
        100.0 * SUM(Is_Driver_Cancelled) / COUNT(*),
        2
    ) AS driver_cancellation_rate,

    ROUND(
        100.0 * SUM(Is_Customer_Cancelled) / COUNT(*),
        2
    ) AS customer_cancellation_rate,

    ROUND(
        100.0 * SUM(
            CASE
                WHEN Ride_Outcome = 'Other'
                THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS other_failure_rate

FROM bookings;
"""

failure_summary_result = run_query(
    failure_summary_query,
    "Cancellation and Ride Failure Summary",
)


# ============================================================
# 31. CANCELLATION REASONS WITH PERCENTAGES
# ============================================================

driver_cancel_reason_query = """
SELECT
    Canceled_Rides_by_Driver AS cancellation_reason,
    COUNT(*) AS total_cancellations,

    ROUND(
        100.0 * COUNT(*) /
        (
            SELECT COUNT(*)
            FROM bookings
            WHERE Is_Driver_Cancelled = 1
              AND Canceled_Rides_by_Driver IS NOT NULL
        ),
        2
    ) AS percentage

FROM bookings

WHERE Is_Driver_Cancelled = 1
  AND Canceled_Rides_by_Driver IS NOT NULL

GROUP BY Canceled_Rides_by_Driver

ORDER BY total_cancellations DESC;
"""

driver_cancel_reason_result = run_query(
    driver_cancel_reason_query,
    "Driver Cancellation Reasons with Percentages",
)


customer_cancel_reason_query = """
SELECT
    Canceled_Rides_by_Customer AS cancellation_reason,
    COUNT(*) AS total_cancellations,

    ROUND(
        100.0 * COUNT(*) /
        (
            SELECT COUNT(*)
            FROM bookings
            WHERE Is_Customer_Cancelled = 1
              AND Canceled_Rides_by_Customer IS NOT NULL
        ),
        2
    ) AS percentage

FROM bookings

WHERE Is_Customer_Cancelled = 1
  AND Canceled_Rides_by_Customer IS NOT NULL

GROUP BY Canceled_Rides_by_Customer

ORDER BY total_cancellations DESC;
"""

customer_cancel_reason_result = run_query(
    customer_cancel_reason_query,
    "Customer Cancellation Reasons with Percentages",
)


# ============================================================
# 32. DRIVER VS CUSTOMER CANCELLATION
# ============================================================

cancellation_comparison_query = """
SELECT
    ROUND(
        100.0 * SUM(Is_Driver_Cancelled) / COUNT(*),
        2
    ) AS driver_cancellation_rate,

    ROUND(
        100.0 * SUM(Is_Customer_Cancelled) / COUNT(*),
        2
    ) AS customer_cancellation_rate,

    ROUND(
        100.0 * SUM(
            CASE
                WHEN Is_Driver_Cancelled = 1
                  OR Is_Customer_Cancelled = 1
                THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS overall_cancellation_rate

FROM bookings;
"""

cancellation_comparison_result = run_query(
    cancellation_comparison_query,
    "Driver vs Customer Cancellation",
)


# ============================================================
# 33. CANCELLATION RATE BY PICKUP LOCATION
# ============================================================

location_cancellation_query = """
SELECT
    Pickup_Location,
    COUNT(*) AS total_bookings,

    SUM(Is_Driver_Cancelled) AS driver_cancellations,
    SUM(Is_Customer_Cancelled) AS customer_cancellations,

    ROUND(
        100.0 * SUM(
            CASE
                WHEN Is_Driver_Cancelled = 1
                  OR Is_Customer_Cancelled = 1
                THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS cancellation_rate

FROM bookings

GROUP BY Pickup_Location

HAVING COUNT(*) >= 1000

ORDER BY cancellation_rate DESC;
"""

location_cancellation_result = run_query(
    location_cancellation_query,
    "Cancellation Rate by Pickup Location",
)


# ============================================================
# 34. HIGH-IMPACT CANCELLATION LOCATIONS
# ============================================================

high_impact_cancellation_query = """
SELECT
    Pickup_Location,
    COUNT(*) AS total_bookings,

    SUM(
        CASE
            WHEN Is_Driver_Cancelled = 1
              OR Is_Customer_Cancelled = 1
            THEN 1
            ELSE 0
        END
    ) AS total_cancellations,

    ROUND(
        100.0 * SUM(
            CASE
                WHEN Is_Driver_Cancelled = 1
                  OR Is_Customer_Cancelled = 1
                THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS cancellation_rate

FROM bookings

GROUP BY Pickup_Location

ORDER BY total_cancellations DESC

LIMIT 10;
"""

high_impact_cancellation_result = run_query(
    high_impact_cancellation_query,
    "High-Impact Cancellation Locations",
)


# ============================================================
# 35. CANCELLATION RATE BY VEHICLE TYPE
# ============================================================

vehicle_cancellation_query = """
SELECT
    Vehicle_Type,
    COUNT(*) AS total_bookings,

    SUM(Is_Driver_Cancelled) AS driver_cancellations,
    SUM(Is_Customer_Cancelled) AS customer_cancellations,

    SUM(
        CASE
            WHEN Is_Driver_Cancelled = 1
              OR Is_Customer_Cancelled = 1
            THEN 1
            ELSE 0
        END
    ) AS total_cancellations,

    ROUND(
        100.0 * SUM(
            CASE
                WHEN Is_Driver_Cancelled = 1
                  OR Is_Customer_Cancelled = 1
                THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS cancellation_rate

FROM bookings

GROUP BY Vehicle_Type

ORDER BY cancellation_rate DESC;
"""

vehicle_cancellation_result = run_query(
    vehicle_cancellation_query,
    "Cancellation Rate by Vehicle Type",
)


# ============================================================
# 36. MONTHLY PERFORMANCE
# ============================================================

monthly_query = """
SELECT
    Month,
    COUNT(*) AS total_bookings,
    SUM(Is_Successful_Ride) AS successful_rides,

    ROUND(
        100.0 * SUM(Is_Successful_Ride) / COUNT(*),
        2
    ) AS success_rate,

    SUM(Realised_Revenue) AS realised_revenue

FROM bookings

GROUP BY Month

ORDER BY Month;
"""

monthly_result = run_query(
    monthly_query,
    "Monthly Booking and Revenue Performance",
)


# ============================================================
# 37. HOURLY REVENUE PERFORMANCE
# ============================================================

hourly_revenue_query = """
SELECT
    Hour,
    COUNT(*) AS total_bookings,
    SUM(Is_Successful_Ride) AS successful_rides,

    ROUND(
        100.0 * SUM(Is_Successful_Ride) / COUNT(*),
        2
    ) AS success_rate,

    SUM(Realised_Revenue) AS realised_revenue,

    ROUND(
        AVG(
            CASE
                WHEN Is_Successful_Ride = 1
                THEN Realised_Revenue
            END
        ),
        2
    ) AS avg_revenue_per_successful_ride

FROM bookings

GROUP BY Hour

ORDER BY Hour;
"""

hourly_result = run_query(
    hourly_revenue_query,
    "Hourly Revenue Performance",
)


# ============================================================
# 38. PEAK VS OFF-PEAK ANALYSIS
# ============================================================

peak_hours = (
    hourly_result
    .sort_values("total_bookings", ascending=False)
    .head(5)
)

off_peak_hours = (
    hourly_result
    .sort_values("total_bookings", ascending=True)
    .head(5)
)

peak_avg = peak_hours["total_bookings"].mean()
off_peak_avg = off_peak_hours["total_bookings"].mean()

print("\nPeak hours:")
print(peak_hours.to_string(index=False))

print("\nOff-peak hours:")
print(off_peak_hours.to_string(index=False))

print(f"\nAverage bookings during peak hours: {peak_avg:.0f}")
print(f"Average bookings during off-peak hours: {off_peak_avg:.0f}")
print(f"Difference: {peak_avg - off_peak_avg:.0f} bookings")


# ============================================================
# 39. VEHICLE PERFORMANCE WITH CANCELLATION METRICS
# ============================================================

vehicle_detail_query = """
SELECT
    Vehicle_Type,
    COUNT(*) AS total_bookings,

    SUM(
        CASE
            WHEN Booking_Status = 'Success'
            THEN 1
            ELSE 0
        END
    ) AS successful_rides,

    ROUND(
        100.0 * SUM(
            CASE
                WHEN Booking_Status = 'Success'
                THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS success_rate,

    SUM(Is_Driver_Cancelled) AS driver_cancellations,
    SUM(Is_Customer_Cancelled) AS customer_cancellations,

    ROUND(
        100.0 * SUM(Is_Driver_Cancelled) / COUNT(*),
        2
    ) AS driver_cancellation_rate,

    ROUND(
        100.0 * SUM(Is_Customer_Cancelled) / COUNT(*),
        2
    ) AS customer_cancellation_rate,

    SUM(
        CASE
            WHEN Booking_Status = 'Success'
            THEN Booking_Value
            ELSE 0
        END
    ) AS realised_revenue,

    ROUND(
        AVG(
            CASE
                WHEN Booking_Status = 'Success'
                THEN Booking_Value
            END
        ),
        2
    ) AS avg_revenue_per_successful_ride

FROM bookings

GROUP BY Vehicle_Type

ORDER BY total_bookings DESC;
"""

vehicle_detail_result = run_query(
    vehicle_detail_query,
    "Detailed Vehicle Performance",
)


# ============================================================
# 40. LOCATION PERFORMANCE DETAIL
# ============================================================

location_performance_query = """
SELECT
    Pickup_Location,
    COUNT(*) AS total_bookings,

    SUM(
        CASE
            WHEN Booking_Status = 'Success'
            THEN 1
            ELSE 0
        END
    ) AS successful_rides,

    ROUND(
        100.0 * SUM(
            CASE
                WHEN Booking_Status = 'Success'
                THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS success_rate,

    SUM(Is_Driver_Cancelled) AS driver_cancellations,
    SUM(Is_Customer_Cancelled) AS customer_cancellations,

    SUM(
        CASE
            WHEN Booking_Status = 'Success'
            THEN Booking_Value
            ELSE 0
        END
    ) AS realised_revenue

FROM bookings

GROUP BY Pickup_Location

ORDER BY successful_rides DESC;
"""

location_performance_result = run_query(
    location_performance_query,
    "Detailed Location Performance",
)


# ============================================================
# 41. LOCATION SUCCESS-RATE COMPARISON
# ============================================================

top_success_locations = (
    location_performance_result
    .sort_values("success_rate", ascending=False)
    .head(10)
)

bottom_success_locations = (
    location_performance_result
    .sort_values("success_rate", ascending=True)
    .head(10)
)

top_avg_success = top_success_locations["success_rate"].mean()
bottom_avg_success = bottom_success_locations["success_rate"].mean()

print("\nLocation success-rate comparison:")
print(
    f"Average success rate - Top 10 locations: "
    f"{top_avg_success:.2f}%"
)
print(
    f"Average success rate - Bottom 10 locations: "
    f"{bottom_avg_success:.2f}%"
)
print(
    f"Difference: "
    f"{top_avg_success - bottom_avg_success:.2f} "
    "percentage points"
)


# ============================================================
# 42. ROUTE PERFORMANCE
# ============================================================

route_query = """
SELECT
    Pickup_Location,
    Drop_Location,
    COUNT(*) AS total_bookings,

    SUM(
        CASE
            WHEN Booking_Status = 'Success'
            THEN 1
            ELSE 0
        END
    ) AS successful_rides,

    ROUND(
        100.0 * SUM(
            CASE
                WHEN Booking_Status = 'Success'
                THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS success_rate,

    SUM(
        CASE
            WHEN Booking_Status = 'Success'
            THEN Booking_Value
            ELSE 0
        END
    ) AS realised_revenue

FROM bookings

GROUP BY
    Pickup_Location,
    Drop_Location

HAVING COUNT(*) >= 30

ORDER BY success_rate DESC;
"""

route_result = run_query(
    route_query,
    "Route Performance",
)

print(
    "\nMaximum bookings on any qualifying route:",
    route_result["total_bookings"].max(),
)

print("\nRoute booking distribution:")
print(route_result["total_bookings"].describe())


# ============================================================
# 43. HIGH-VOLUME ROUTES
# ============================================================

high_volume_routes = (
    route_result[
        route_result["total_bookings"] >= 50
    ]
    .sort_values("total_bookings", ascending=False)
    .head(10)
)

high_volume_avg_success = (
    high_volume_routes["success_rate"].mean()
)

overall_route_avg_success = (
    route_result["success_rate"].mean()
)

print("\nHigh-volume routes:")
print(high_volume_routes.to_string(index=False))

print(
    f"\nAverage success rate - Top 10 high-volume routes: "
    f"{high_volume_avg_success:.2f}%"
)

print(
    f"Average success rate - All routes: "
    f"{overall_route_avg_success:.2f}%"
)

print(
    f"Difference: "
    f"{high_volume_avg_success - overall_route_avg_success:.2f} "
    "percentage points"
)


# ============================================================
# 44. RAW CANCELLATION FIELD VALUES
# ============================================================

print("\n" + "=" * 70)
print("RAW CANCELLATION FIELD VALUES")
print("=" * 70)

print("\nDriver cancellation values:")
print(
    clean_df["Canceled_Rides_by_Driver"]
    .value_counts(dropna=False)
)

print("\nCustomer cancellation values:")
print(
    clean_df["Canceled_Rides_by_Customer"]
    .value_counts(dropna=False)
)

print("\nIncomplete ride values:")
print(
    clean_df["Incomplete_Rides"]
    .value_counts(dropna=False)
)


# ============================================================
# 45. CANCELLATION SUMMARY USING PANDAS
# ============================================================

total_bookings = len(clean_df)

driver_cancellations = (
    clean_df["Canceled_Rides_by_Driver"].notna().sum()
)

customer_cancellations = (
    clean_df["Canceled_Rides_by_Customer"].notna().sum()
)

incomplete_rides = (
    clean_df["Incomplete_Rides"].eq("Yes").sum()
)

driver_cancel_rate = (
    driver_cancellations / total_bookings * 100
)

customer_cancel_rate = (
    customer_cancellations / total_bookings * 100
)

incomplete_rate = (
    incomplete_rides / total_bookings * 100
)

print("\nCancellation summary:")
print(f"Total Bookings: {total_bookings:,}")
print(
    f"Driver Cancellations: "
    f"{driver_cancellations:,} "
    f"({driver_cancel_rate:.2f}%)"
)
print(
    f"Customer Cancellations: "
    f"{customer_cancellations:,} "
    f"({customer_cancel_rate:.2f}%)"
)
print(
    f"Incomplete Rides: "
    f"{incomplete_rides:,} "
    f"({incomplete_rate:.2f}%)"
)


# ============================================================
# 46. CANCELLATION REASON PERCENTAGES USING PANDAS
# ============================================================

driver_cancel_reasons = (
    clean_df["Canceled_Rides_by_Driver"]
    .dropna()
    .value_counts()
    .reset_index()
)

driver_cancel_reasons.columns = [
    "Cancellation_Reason",
    "Count",
]

driver_cancel_reasons["Percentage"] = (
    driver_cancel_reasons["Count"]
    / driver_cancel_reasons["Count"].sum()
    * 100
)

customer_cancel_reasons = (
    clean_df["Canceled_Rides_by_Customer"]
    .dropna()
    .value_counts()
    .reset_index()
)

customer_cancel_reasons.columns = [
    "Cancellation_Reason",
    "Count",
]

customer_cancel_reasons["Percentage"] = (
    customer_cancel_reasons["Count"]
    / customer_cancel_reasons["Count"].sum()
    * 100
)

print("\nDriver cancellation reasons:")
print(driver_cancel_reasons.to_string(index=False))

print("\nCustomer cancellation reasons:")
print(customer_cancel_reasons.to_string(index=False))


# ============================================================
# 47. VEHICLE CANCELLATION ANALYSIS USING PANDAS
# ============================================================

vehicle_cancellation = (
    clean_df.groupby("Vehicle_Type")
    .agg(
        total_bookings=("Booking_ID", "count"),
        driver_cancellations=(
            "Canceled_Rides_by_Driver",
            "count",
        ),
        customer_cancellations=(
            "Canceled_Rides_by_Customer",
            "count",
        ),
        incomplete_rides=(
            "Incomplete_Rides",
            lambda x: (x == "Yes").sum(),
        ),
    )
    .reset_index()
)

vehicle_cancellation["driver_cancel_rate"] = (
    vehicle_cancellation["driver_cancellations"]
    / vehicle_cancellation["total_bookings"]
    * 100
)

vehicle_cancellation["customer_cancel_rate"] = (
    vehicle_cancellation["customer_cancellations"]
    / vehicle_cancellation["total_bookings"]
    * 100
)

vehicle_cancellation["incomplete_rate"] = (
    vehicle_cancellation["incomplete_rides"]
    / vehicle_cancellation["total_bookings"]
    * 100
)

print("\nVehicle cancellation analysis:")
print(vehicle_cancellation.round(2).to_string(index=False))


# ============================================================
# 48. VISUALISATIONS
# ============================================================

# The notebook contains exploratory charts. The script keeps the
# main project charts here so the analysis remains reproducible.

plt.figure(figsize=(10, 6))
plt.bar(
    revenue_efficiency_result["Vehicle_Type"],
    revenue_efficiency_result["avg_revenue_per_km"],
)
plt.title("Average Revenue per Kilometre by Vehicle Type")
plt.xlabel("Vehicle Type")
plt.ylabel("Average Revenue per Kilometre")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
plt.bar(
    driver_cancel_reason_result["cancellation_reason"],
    driver_cancel_reason_result["percentage"],
)
plt.title("Driver Cancellation Reasons")
plt.xlabel("Cancellation Reason")
plt.ylabel("Percentage of Driver Cancellations")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
plt.bar(
    customer_cancel_reason_result["cancellation_reason"],
    customer_cancel_reason_result["percentage"],
)
plt.title("Customer Cancellation Reasons")
plt.xlabel("Cancellation Reason")
plt.ylabel("Percentage of Customer Cancellations")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(
    day_result["Day_Name"],
    day_result["total_bookings"],
)
plt.title("Booking Volume by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Total Bookings")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(
    day_result["Day_Name"],
    day_result["realised_revenue"],
)
plt.title("Realised Revenue by Day")
plt.xlabel("Day")
plt.ylabel("Realised Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ============================================================
# 49. FINAL PROJECT SUMMARY
# ============================================================

final_summary = {
    "Total Bookings": int(len(clean_df)),
    "Successful Rides": int(
        clean_df["Is_Successful_Ride"].sum()
    ),
    "Success Rate": round(
        clean_df["Is_Successful_Ride"].mean() * 100,
        2,
    ),
    "Driver Cancellation Rate": round(
        clean_df["Canceled_Rides_by_Driver"].notna().mean()
        * 100,
        2,
    ),
    "Customer Cancellation Rate": round(
        clean_df["Canceled_Rides_by_Customer"].notna().mean()
        * 100,
        2,
    ),
    "Incomplete Ride Rate": round(
        clean_df["Incomplete_Rides"].eq("Yes").mean()
        * 100,
        2,
    ),
    "Total Revenue": round(
        clean_df["Realised_Revenue"].sum(),
        2,
    ),
    "Average Ride Distance": round(
        clean_df["Ride_Distance"].mean(),
        2,
    ),
    "Average Driver Rating": round(
        clean_df["Driver_Ratings"].mean(),
        2,
    ),
    "Average Customer Rating": round(
        clean_df["Customer_Rating"].mean(),
        2,
    ),
}

final_summary_df = pd.DataFrame(
    final_summary.items(),
    columns=["Metric", "Value"],
)

print("\n" + "=" * 70)
print("FINAL PROJECT KPIs")
print("=" * 70)
print(final_summary_df.to_string(index=False))


# ============================================================
# 50. FINAL BUSINESS INSIGHTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL BUSINESS INSIGHTS")
print("=" * 70)

print(
    "1. Driver cancellations are the primary operational challenge, "
    "at approximately 17.89% of bookings."
)

print(
    "2. Customer cancellations account for approximately 10.19% "
    "of bookings."
)

print(
    "3. Vehicle type alone produces relatively small differences "
    "in ride success, so vehicle-level decisions should be "
    "considered alongside location and route."
)

print(
    "4. Vehicle-location combinations show a substantially wider "
    "performance gap than vehicle type alone."
)

print(
    "5. The top high-volume routes perform above the overall "
    "route-level success average."
)

print(
    "6. Pickup locations such as Vijayanagar and Tumkur Road "
    "show elevated cancellation/performance concerns."
)

print(
    "7. Day-of-week success rates are relatively stable, suggesting "
    "that operational improvements should focus more heavily on "
    "driver-side issues, locations, routes and vehicle allocation."
)


# ============================================================
# 51. CLOSE DATABASE
# ============================================================

conn.close()

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)
