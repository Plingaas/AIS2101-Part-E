import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Housing_categorized.csv")
data = data.astype(float)


# e. Identifying Data Groupings
# Use box plots or swarm plots to identify separable groupings within each feature
def boxplots():
    for column in data.columns:
        if column != "price":  # Exclude the target variable
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=column, y="price", data=data)
            plt.title(f"Boxplot of {column} by Price")
            plt.xlabel("Price")
            plt.ylabel(column)
            plt.xticks(rotation=45)
            plt.show()


# a. Class Balance
def counts():
    class_counts = data["price"].value_counts()
    print(class_counts)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title("Class Distribution")
    plt.xlabel("Price")
    plt.ylabel("Count")
    plt.show()


# b. Relationships Amongst Data Elements
def relations():
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()


# c. Calculating Statistics
# Since most features are categorical, calculate counts or proportions for each category within each feature
statistics = data["area"].describe()
print(statistics)

# d. Handling Potential Issues in Data
# Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:")
print(missing_values)


# Visualize the distribution of each feature
def distributions():
    for column in data.columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(x=column, data=data)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()


# f. Proximity of Identified Data Groupings
# Measure distance between groupings using statistical tests like chi-square test of independence

# Additional visualizations and analysis as needed


# Set up the plot
def grouping(feature, subplot=(1, 1, 1)):

    plt.subplot(subplot[0], subplot[1], subplot[2])
    # Create a count plot to visualize groupings for binary data
    sns.countplot(x=feature, hue="price", data=data)

    # Add labels and title
    plt.title(f"Count Plot of {feature} by Price")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.legend(title="Price")


grouping("mainroad", (3, 2, 1))
grouping("guestroom", (3, 2, 2))
grouping("basement", (3, 2, 3))
grouping("airconditioning", (3, 2, 4))
grouping("prefarea", (3, 2, 5))

plt.show()
# price, area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, airconditioning, parking, prefarea, furnishingstatus
