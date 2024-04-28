import pandas as pd

# Load your dataset
data = pd.read_csv("Housing_reformatted.csv")

# Assume 'price' is not in the dataset and needs to be added
# Generate 'price' column based on some condition (you may need to replace this with actual data)

# Add 'price_category' column based on 'price'
data["price_category"] = data["price"].apply(
    lambda x: 0 if x < 3500000 else (1 if x < 5000000 else 2)
)

# Save the updated dataset with the new column
data.to_csv("Housing_categorized2.csv", index=False)
