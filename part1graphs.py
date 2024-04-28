import pandas as pd
import matplotlib.pyplot as plt
import os

# Read the CSV file
df = pd.read_csv("Housing_categorized.csv")


def plot_feature_frequency(feature, save=False, subplot=(1, 1, 1)):
    # Count the frequency of each value of the specified feature
    value_counts = df[feature].value_counts().sort_index()

    # Plot the frequency of each value
    plt.subplot(subplot[0], subplot[1], subplot[2])
    plt.bar(value_counts.index, value_counts.values)
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.title(f"Frequency of {feature}")
    if save:
        plt.savefig(os.path.join("images", f"{feature}_frequency.png"))


plot_feature_frequency("price", False, (4, 3, 1))
plot_feature_frequency("bedrooms", False, (4, 3, 2))
plot_feature_frequency("bathrooms", False, (4, 3, 3))
plot_feature_frequency("stories", False, (4, 3, 4))
plot_feature_frequency("mainroad", False, (4, 3, 5))
plot_feature_frequency("guestroom", False, (4, 3, 6))
plot_feature_frequency("basement", False, (4, 3, 7))
plot_feature_frequency("airconditioning", False, (4, 3, 9))
plot_feature_frequency("parking", False, (4, 3, 10))
plot_feature_frequency("prefarea", False, (4, 3, 11))
plot_feature_frequency("furnishingstatus", False, (4, 3, 12))
plt.show()
