import numpy as np

np.random.seed(42)

distances = np.random.uniform(1.0, 15.0, 25)

fees = 20 + (5 * distances)

condition = fees > 60

print("Orders where delivery fee exceeds Rs 60:")
print("-" * 45)
print("Distance (km)    Fee (Rs)")
print("-" * 45)

print(np.column_stack((distances[condition], fees[condition])))

print("\nDelivery Fee Statistics")
print("-" * 30)

print(f"Minimum Fee       : Rs {np.min(fees):.2f}")
print(f"Maximum Fee       : Rs {np.max(fees):.2f}")
print(f"Mean Fee          : Rs {np.mean(fees):.2f}")
print(f"Standard Deviation: Rs {np.std(fees):.2f}")