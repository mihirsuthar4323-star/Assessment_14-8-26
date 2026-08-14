import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------
# 1. GENERATE SYNTHETIC DATA USING NUMPY
# ---------------------------------------------------

np.random.seed(7)

n = 200

order_value = np.random.uniform(100, 800, n)
distance_km = np.random.uniform(1, 20, n)
delivery_time_mins = np.random.normal(30, 8, n)
rating = np.random.uniform(1.0, 5.0, n)
discount_pct = np.random.uniform(0, 30, n)



# 2. LOAD DATA INTO PANDAS DATAFRAME


df = pd.DataFrame({
    "order_value": order_value,
    "distance_km": distance_km,
    "delivery_time_mins": delivery_time_mins,
    "rating": rating,
    "discount_pct": discount_pct
})


print("Original Dataset")
print("=" * 70)
print(df.head())


# ---------------------------------------------------
# 3. INTRODUCE 5% NULL VALUES
# ---------------------------------------------------

# 5% of 200 = 10 rows
null_count = int(0.05 * n)

# Random indexes for delivery time
delivery_null_indices = np.random.choice(
    df.index,
    size=null_count,
    replace=False
)

# Random indexes for rating
rating_null_indices = np.random.choice(
    df.index,
    size=null_count,
    replace=False
)

# Introduce null values
df.loc[delivery_null_indices, "delivery_time_mins"] = np.nan
df.loc[rating_null_indices, "rating"] = np.nan


print("\nNull Values Before Cleaning")
print("=" * 70)
print(df.isnull().sum())


# ---------------------------------------------------
# 4. FILL NULL VALUES WITH COLUMN MEDIANS
# ---------------------------------------------------

delivery_median = df["delivery_time_mins"].median()
rating_median = df["rating"].median()

df["delivery_time_mins"] = df["delivery_time_mins"].fillna(
    delivery_median
)

df["rating"] = df["rating"].fillna(
    rating_median
)


print("\nNull Values After Cleaning")
print("=" * 70)
print(df.isnull().sum())


# ---------------------------------------------------
# 5. CREATE DERIVED COLUMN
# DELIVERY SPEED IN KM/H
# ---------------------------------------------------

df["delivery_speed_kmph"] = (
    df["distance_km"] /
    (df["delivery_time_mins"] / 60)
)


# ---------------------------------------------------
# 6. CREATE SPEED BAND USING pd.qcut()
# ---------------------------------------------------

df["speed_band"] = pd.qcut(
    df["delivery_speed_kmph"],
    q=3,
    labels=["Slow", "Normal", "Fast"]
)


print("\nSpeed Band Distribution")
print("=" * 70)
print(df["speed_band"].value_counts())


# ---------------------------------------------------
# 7. CORRELATION HEATMAP
# ---------------------------------------------------

# Select numeric columns
numeric_df = df.select_dtypes(include=np.number)

# Pearson correlation matrix
correlation_matrix = numeric_df.corr(method="pearson")

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Food Delivery Numeric Feature Correlation")

plt.tight_layout()

# Save heatmap
plt.savefig(
    "correlation_heatmap.png",
    dpi=150
)

plt.close()


# ---------------------------------------------------
# 8. SEABORN PAIRPLOT
# ---------------------------------------------------

pairplot_data = df[
    [
        "order_value",
        "distance_km",
        "delivery_time_mins",
        "rating",
        "speed_band"
    ]
]

pair_plot = sns.pairplot(
    pairplot_data,
    hue="speed_band"
)

pair_plot.fig.suptitle(
    "Food Delivery Feature Pairplot",
    y=1.02
)

# Save pairplot
pair_plot.savefig(
    "pairplot.png",
    dpi=150
)

plt.close("all")


# ---------------------------------------------------
# 9. FINAL DATASET PREVIEW
# ---------------------------------------------------

print("\nFinal Dataset")
print("=" * 70)
print(df.head(10))

print("\nEDA Pipeline Completed Successfully!")

print("\nFiles Created:")
print("1. correlation_heatmap.png")
print("2. pairplot.png")