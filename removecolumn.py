import pandas as pd

# Load your dataset
data = pd.read_csv("Housing_categorized.csv")

# Remove the column you want to drop (e.g., "column_to_remove")
column_to_remove = "hotwaterheating"
data_modified = data.drop(column_to_remove, axis=1)

# Save the modified dataset
data_modified.to_csv("Housing_categorized.csv", index=False)
