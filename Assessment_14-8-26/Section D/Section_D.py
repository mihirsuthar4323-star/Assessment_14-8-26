# Create a Python 3 program for a food delivery analytics dataset using NumPy, Pandas, Matplotlib and Seaborn.

# Requirements:
# 1. Generate a synthetic dataset with at least 200 rows.
# 2. Include at least 5 numeric columns such as order_value, distance_km, delivery_time_mins, rating and discount_pct.
# 3. Include one categorical column called cuisine_type.
# 4. Generate the data using NumPy and load it into a Pandas DataFrame.
# 5. Produce a Seaborn pairplot of all numeric columns, coloured by cuisine_type.
# 6. Save the pairplot as a PNG file.
# 7. Calculate the Pearson correlation matrix of all numeric columns.
# 8. Display the correlation matrix using a Seaborn heatmap.
# 9. The heatmap must show annotated correlation values.
# 10. Use a diverging colour palette for the heatmap.
# 11. Save the heatmap as a separate PNG file at DPI 200.
# 12. Provide clean and executable Python code.

# AI PYTHON PROGRAM:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


np.random.seed(42)

n = 200

df = pd.DataFrame({
    "order_value": np.random.uniform(100, 1000, n),
    "distance_km": np.random.uniform(1, 20, n),
    "delivery_time_mins": np.random.normal(30, 8, n),
    "rating": np.random.uniform(1, 5, n),
    "discount_pct": np.random.uniform(0, 30, n),
    "cuisine_type": np.random.choice(
        ["Indian", "Chinese", "Fast Food"],
        n
    )
})

sns.pairplot(
    df,
    hue="cuisine_type"
)

plt.savefig("food_delivery_pairplot.png")
plt.close()

correlation_matrix = df.select_dtypes(
    include=np.number
).corr()

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm"
)

plt.title("Food Delivery Correlation Heatmap")

plt.savefig(
    "food_delivery_correlation_heatmap.png"
)

plt.show()








# Problem 1 — Heatmap DPI
# Original code mein:

plt.savefig("food_delivery_correlation_heatmap.png")

# Solution:

plt.savefig(
    "food_delivery_correlation_heatmap.png",
    dpi=200)


# While testing the AI-generated code,
# I found that the heatmap was saved without specifying DPI 200,
# which was required by the assessment. 
# I corrected the savefig() statement by adding dpi=200.
# I also improved the pairplot saving by using the returned pairplot 
# figure object and added null-value handling for numeric and categorical columns.
# These changes make the program more reliable and satisfy the required output specifications.