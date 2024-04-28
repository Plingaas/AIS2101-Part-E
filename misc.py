import pandas as pd
import matplotlib.pyplot as plt

### PERFECT PLOT EXAMPLE FOR PART 3
# Read the CSV file
df = pd.read_csv("Housing_categorized2.csv")

# Assign colors based on price categories
colors = []
for price in df["price"]:
    if price < 3500000:
        colors.append("red")
    elif price < 5000000:
        colors.append("blue")
    else:
        colors.append("green")

# Plot the points
plt.figure(figsize=(10, 6))
plt.scatter(df["price_category"], df["price"], c=colors)

# Customize plot
plt.title("Perfect separation of classes example")
plt.xlabel("Price Category")
plt.ylabel("True Price")

# Show plot
plt.show()
##############
