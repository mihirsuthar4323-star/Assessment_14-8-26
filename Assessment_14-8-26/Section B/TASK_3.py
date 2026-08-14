import numpy as np
import matplotlib.pyplot as plt

# Fixed random seed
np.random.seed(42)

# 12 months
months = np.arange(1, 13)

# Generate total orders between 1,000 and 5,000
orders = np.random.randint(1000, 5001, 12)

# Generate random average order value between Rs 200 and Rs 400
average_order_value = np.random.uniform(200, 400, 12)

# Calculate monthly revenue
revenue = orders * average_order_value

# Generate delivery times
delivery_mean = 28
delivery_std = 4

delivery_times = np.random.normal(
    delivery_mean,
    delivery_std,
    500
)

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))


# ---------------------------------------------------
# SUBPLOT 1 — LINE CHART: TOTAL ORDERS
# ---------------------------------------------------

axes[0].plot(
    months,
    orders,
    marker='o'
)

axes[0].set_title("Monthly Total Orders")
axes[0].set_xlabel("Month")
axes[0].set_ylabel("Number of Orders")

# Annotate every data point
for x, y in zip(months, orders):
    axes[0].annotate(
        str(y),
        xy=(x, y),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center"
    )


# ---------------------------------------------------
# SUBPLOT 2 — BAR CHART: MONTHLY REVENUE
# ---------------------------------------------------

# Green if revenue > Rs 8,00,000
# Red otherwise
bar_colors = np.where(
    revenue > 800000,
    "green",
    "red"
)

axes[1].bar(
    months,
    revenue,
    color=bar_colors
)

# Draw Rs 8,00,000 threshold line
axes[1].axhline(
    y=800000,
    linestyle="--",
    label="Rs 8,00,000 Threshold"
)

axes[1].set_title("Monthly Revenue")
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Revenue (Rs)")

axes[1].legend()


# ---------------------------------------------------
# SUBPLOT 3 — HISTOGRAM: DELIVERY TIMES
# ---------------------------------------------------

axes[2].hist(
    delivery_times,
    bins=15
)

# Calculate sample mean
sample_mean = np.mean(delivery_times)

# Draw vertical dashed mean line
axes[2].axvline(
    sample_mean,
    linestyle="--",
    label="Mean"
)

axes[2].set_title("Delivery Time Distribution")
axes[2].set_xlabel("Delivery Time (Minutes)")
axes[2].set_ylabel("Frequency")

axes[2].legend()

fig.suptitle(
    "Food Delivery Monthly Performance Dashboard",
    fontsize=16
)

plt.tight_layout()

# Save figure as PNG with DPI 150
plt.savefig(
    "food_delivery_dashboard.png",
    dpi=150
)

# Show chart
plt.show()