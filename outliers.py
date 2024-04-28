import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def xy_variable_plot(x, y):
    # Read the CSV file
    df = pd.read_csv("Housing.csv")

    # Extract prices and areas
    prices = df[f"{y}"]
    areas = df[f"{x}"]

    # Plot prices vs. areas
    plt.figure(figsize=(8, 6))
    plt.scatter(areas, prices, color="blue", alpha=0.5)
    plt.title(f"{y} vs. {x}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True)
    plt.show()


def xy_variable_plot_color(x, y, color_feature):
    """
    Plot points with colors based on the value of another feature.

    Args:
    - x: Name of the feature for the x-axis
    - y: Name of the feature for the y-axis
    - color_feature: Name of the feature to use for coloring the points

    Returns:
    - None
    """
    # Read the CSV file
    df = pd.read_csv("Housing_reformatted.csv")

    # Extract values of x, y, and color_feature
    x_values = df[x]
    y_values = df[y]
    color_values = df[color_feature]

    # Get unique values of the color feature and corresponding colors
    unique_color_values = color_values.unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_color_values)))

    # Plot points with colors based on the value of the color feature
    plt.figure(figsize=(8, 6))
    for i, value in enumerate(unique_color_values):
        subset_x = x_values[color_values == value]
        subset_y = y_values[color_values == value]
        plt.scatter(subset_x, subset_y, color=colors[i], label=value, alpha=0.5)

    # Add labels and legend
    plt.title(f"{y} vs. {x}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend(title=color_feature, loc="upper right")
    plt.grid(True)
    plt.show()


xy_variable_plot("area", "price")
