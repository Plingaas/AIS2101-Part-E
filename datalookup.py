import pandas as pd
import matplotlib.pyplot as plt
import os
# Read the CSV file
df = pd.read_csv("Housing_categorized.csv")


# Specify the label and value you want to check
def get_value_frequency(feature, value):
    # Count the number of objects with the specific value for the specific label
    count = df[df[feature] == value].shape[0]
    print(f"Number of objects with {feature} = {value}: {count}")

# Read the CSV file
# Specify the feature for which you want to plot the frequency
def plot_feature_frequency(feature, save=False):
    # Count the frequency of each value of the specified feature
    value_counts = df[feature].value_counts().sort_index()

    # Plot the frequency of each value
    plt.bar(value_counts.index, value_counts.values)
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.title(f"Frequency of {feature}")
    if save:
        plt.savefig(os.path.join("images", f"{feature}_frequency.png"))
    plt.show()

plot_feature_frequency("price", True)
plot_feature_frequency("furnishingstatus", True)

def calculate_median_by_feature(df, feature1, feature2):
    """
    Calculate the median value of feature2 for each unique value of feature1 in the DataFrame df.
    
    Args:
    - df: DataFrame containing the dataset
    - feature1: Name of the first feature
    - feature2: Name of the second feature
    
    Returns:
    - A dictionary where keys are unique values of feature1 and values are the median values of feature2
    """
    median_values = {}
    unique_values = df[feature1].unique()
    for value in unique_values:
        median_values[value] = df[df[feature1] == value][feature2].median()
    plt.bar(median_values.keys(), median_values.values())
    plt.xlabel(feature1)
    plt.ylabel(f"Median {feature2}")
    plt.title(f"Median {feature2} by {feature1}")
    plt.show()

def calculate_average_by_feature(df, feature1, feature2, save = True):
    """
    Calculate the average (mean) value of feature2 for each unique value of feature1 in the DataFrame df.
    
    Args:
    - df: DataFrame containing the dataset
    - feature1: Name of the first feature
    - feature2: Name of the second feature
    
    Returns:
    - A dictionary where keys are unique values of feature1 and values are the average (mean) values of feature2
    """
    average_values = {}
    unique_values = df[feature1].unique()
    for value in unique_values:
        average_values[value] = df[df[feature1] == value][feature2].mean()
    plt.bar(average_values.keys(), average_values.values())
    plt.xlabel(feature1)
    plt.ylabel(f"Average {feature2}")
    plt.title(f"Average {feature2} by {feature1}")
    if save:
        plt.savefig(os.path.join("images/averages", f"{feature2}_vs_{feature1}_average.png"))
    plt.show()

def calculate_statistics(df, feature):
    """
    Calculate mode, first quartile (Q1), third quartile (Q3), and standard deviation for a given feature.
    
    Args:
    - df: DataFrame containing the dataset
    - feature: Name of the feature for which statistics are to be calculated
    
    Returns:
    - A dictionary containing mode, Q1, Q3, and standard deviation of the given feature
    """

    # Mean
    mean_value = df[feature].mean()

    # Mode
    mode_value = df[feature].mode()[0]
    
    # Median
    median_value = df[feature].median()

    # Q1
    q1_value = df[feature].quantile(0.25)
    
    # Q3
    q3_value = df[feature].quantile(0.75)
    
    # Standard deviation
    std_value = df[feature].std()
    
    # Store the results in a dictionary
    statistics = {
        "mean": mean_value,
        "mode": mode_value,
        "median": median_value,
        "Q1": q1_value,
        "Q3": q3_value,
        "standard_deviation": std_value
    }
    
    print(f"feature: {feature} - {statistics}")

