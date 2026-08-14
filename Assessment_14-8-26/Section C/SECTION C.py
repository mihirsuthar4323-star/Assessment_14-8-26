import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. GENERATE DATASET
# ============================================================

np.random.seed(42)

n = 250

restaurants = [
    "Food Palace",
    "Spice Hub",
    "Pizza Point",
    "Tasty Bites",
    "Urban Kitchen"
]

cities = [
    "Ahmedabad",
    "Surat",
    "Vadodara"
]

cuisines = [
    "Indian",
    "Chinese",
    "Fast Food"
]


df = pd.DataFrame({
    "restaurant_name": np.random.choice(restaurants, n),
    "city": np.random.choice(cities, n),
    "cuisine_type": np.random.choice(cuisines, n),
    "order_value": np.random.uniform(100, 1000, n),
    "distance_km": np.random.uniform(1, 20, n),
    "delivery_time_mins": np.random.normal(30, 8, n),
    "rating": np.random.uniform(1, 5, n),
    "discount_pct": np.random.uniform(0, 30, n)
})


# ============================================================
# 2. INTRODUCE SOME NULL VALUES
# ============================================================

null_indices = np.random.choice(df.index, size=10, replace=False)

df.loc[null_indices[:5], "delivery_time_mins"] = np.nan
df.loc[null_indices[5:], "rating"] = np.nan


# ============================================================
# 3. CLEAN NUMERIC NULL VALUES
# ============================================================

numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:

    if df[column].isnull().any():

        median_value = df[column].median()

        df[column] = df[column].fillna(median_value)


print("Dataset generated and cleaned successfully.")
print(f"Total rows: {len(df)}")


# ============================================================
# OPTION 1
# SUMMARY STATISTICS
# Primary Technique: NumPy statistical functions
# ============================================================

def summary_statistics():

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    numeric_data = df.select_dtypes(include=np.number)

    for column in numeric_data.columns:

        values = numeric_data[column].to_numpy()

        print(f"\n{column}")

        print(f"Mean       : {np.mean(values):.2f}")
        print(f"Minimum    : {np.min(values):.2f}")
        print(f"Maximum    : {np.max(values):.2f}")
        print(f"Std Dev    : {np.std(values):.2f}")


# ============================================================
# OPTION 2
# DISTRIBUTION ANALYSIS
# Primary Technique: Matplotlib subplot
# ============================================================

def distribution_analysis():

    print("\n" + "=" * 60)
    print("DISTRIBUTION ANALYSIS")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram 1 - Order Value
    axes[0].hist(
        df["order_value"],
        bins=15
    )

    axes[0].set_title("Order Value Distribution")
    axes[0].set_xlabel("Order Value (Rs)")
    axes[0].set_ylabel("Frequency")


    # Histogram 2 - Delivery Time
    axes[1].hist(
        df["delivery_time_mins"],
        bins=15
    )

    axes[1].set_title("Delivery Time Distribution")
    axes[1].set_xlabel("Delivery Time (Minutes)")
    axes[1].set_ylabel("Frequency")


    fig.suptitle("Food Delivery Distribution Analysis")

    plt.tight_layout()

    plt.savefig(
        "distribution_analysis.png",
        dpi=150
    )

    plt.close()

    print("Chart saved as: distribution_analysis.png")


# ============================================================
# OPTION 3
# CORRELATION HEATMAP
# Primary Technique: Seaborn
# ============================================================

def correlation_heatmap():

    print("\n" + "=" * 60)
    print("CORRELATION HEATMAP")
    print("=" * 60)

    numeric_data = df.select_dtypes(include=np.number)

    correlation = numeric_data.corr()

    plt.figure(figsize=(10, 7))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm"
    )

    plt.title("Food Delivery Feature Correlation")

    plt.tight_layout()

    plt.savefig(
        "correlation_heatmap_console.png",
        dpi=150
    )

    plt.close()

    print("Heatmap saved as: correlation_heatmap_console.png")


# ============================================================
# OPTION 4
# RESTAURANT PERFORMANCE REPORT
# Primary Technique: Pandas groupby
# ============================================================

def restaurant_performance():

    print("\n" + "=" * 60)
    print("RESTAURANT PERFORMANCE REPORT")
    print("=" * 60)

    performance = (
        df.groupby("restaurant_name")
        .agg(
            mean_order_value=("order_value", "mean"),
            mean_delivery_time=("delivery_time_mins", "mean"),
            mean_rating=("rating", "mean"),
            total_orders=("order_value", "count")
        )
        .sort_values(
            "mean_rating",
            ascending=False
        )
        .reset_index()
    )

    performance["mean_order_value"] = (
        performance["mean_order_value"].round(2)
    )

    performance["mean_delivery_time"] = (
        performance["mean_delivery_time"].round(2)
    )

    performance["mean_rating"] = (
        performance["mean_rating"].round(2)
    )

    print(performance.to_string(index=False))

    # Save restaurant performance chart
    plt.figure(figsize=(10, 6))

    plt.bar(
        performance["restaurant_name"],
        performance["mean_rating"]
    )

    plt.title("Restaurant Mean Rating")
    plt.xlabel("Restaurant")
    plt.ylabel("Mean Rating")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(
        "restaurant_performance.png",
        dpi=150
    )

    plt.close()

    print("\nChart saved as: restaurant_performance.png")


# ============================================================
# FINAL SUMMARY REPORT
# ============================================================

def final_summary_report():

    print("\n")
    print("=" * 70)
    print("FINAL FOOD DELIVERY ANALYTICS SUMMARY REPORT")
    print("=" * 70)


    # --------------------------------------------------------
    # Top 3 restaurants by mean rating
    # --------------------------------------------------------

    restaurant_ratings = (
        df.groupby("restaurant_name")["rating"]
        .mean()
        .sort_values(ascending=False)
        .head(3)
    )

    print("\nTop 3 Restaurants by Mean Rating:")
    print("-" * 40)

    for restaurant, rating in restaurant_ratings.items():

        print(
            f"{restaurant}: {rating:.2f}"
        )


    # --------------------------------------------------------
    # Highest absolute Pearson correlation
    # --------------------------------------------------------

    numeric_data = df.select_dtypes(include=np.number)

    correlation_matrix = numeric_data.corr()

    correlation_values = correlation_matrix.abs()

    # Remove diagonal values
    np.fill_diagonal(
        correlation_values.values,
        0
    )

    max_position = np.unravel_index(
        np.argmax(correlation_values.values),
        correlation_values.shape
    )

    column_1 = correlation_values.index[max_position[0]]
    column_2 = correlation_values.columns[max_position[1]]

    correlation_value = correlation_matrix.loc[
        column_1,
        column_2
    ]

    print("\nNumeric Column Pair with Highest Absolute Pearson Correlation:")
    print("-" * 65)

    print(
        f"{column_1} and {column_2}: "
        f"{correlation_value:.4f}"
    )


    # --------------------------------------------------------
    # Overall delivery time statistics
    # --------------------------------------------------------

    delivery_times = df["delivery_time_mins"].to_numpy()

    mean_delivery_time = np.mean(delivery_times)

    std_delivery_time = np.std(delivery_times)


    print("\nOverall Delivery Time:")
    print("-" * 40)

    print(
        f"Mean       : {mean_delivery_time:.2f} minutes"
    )

    print(
        f"Std Dev    : {std_delivery_time:.2f} minutes"
    )


# ============================================================
# MAIN MENU
# ============================================================

while True:

    print("\n")
    print("=" * 60)
    print("FOOD DELIVERY ANALYTICS CONSOLE")
    print("=" * 60)

    print("1. Summary Statistics")
    print("2. Distribution Analysis")
    print("3. Correlation Heatmap")
    print("4. Restaurant Performance Report")
    print("5. Exit")

    choice = input("\nEnter your choice (1-5): ")


    if choice == "1":

        summary_statistics()


    elif choice == "2":

        distribution_analysis()


    elif choice == "3":

        correlation_heatmap()


    elif choice == "4":

        restaurant_performance()


    elif choice == "5":

        print("\nExiting the program...")

        final_summary_report()

        print("\nThank you for using Food Delivery Analytics Console!")

        break


    else:

        print("\nInvalid choice! Please enter a number from 1 to 5.")