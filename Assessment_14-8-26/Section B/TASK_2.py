import numpy as np
import pandas as pd

np.random.seed(55)

n = 50

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
    "order_value": np.random.randint(200, 1501, n),
    "delivery_time_mins": np.random.randint(20, 51, n),
    "rating": np.round(np.random.uniform(1.0, 5.0, n), 1),
    "cuisine_type": np.random.choice(cuisines, n)
})

print("Food Delivery Dataset")
print("=" * 80)
print(df.to_string(index=False))

summary = (
    df.groupby("restaurant_name")
      .agg(
          mean_order_value=("order_value", "mean"),
          mean_delivery_time=("delivery_time_mins", "mean"),
          mean_rating=("rating", "mean")
      )
)

filtered = summary[
    (summary["mean_rating"] > 4.0) &
    (summary["mean_delivery_time"] < 35)
]

final_result = (
    filtered
    .sort_values("mean_order_value", ascending=False)
    .reset_index()
)

final_result["mean_order_value"] = final_result["mean_order_value"].round(2)
final_result["mean_delivery_time"] = final_result["mean_delivery_time"].round(2)
final_result["mean_rating"] = final_result["mean_rating"].round(2)

print("\n\nRestaurant Performance Report")
print("=" * 80)

if final_result.empty:
    print("No restaurant satisfies the given conditions.")
else:
    print(final_result.to_string(index=False))