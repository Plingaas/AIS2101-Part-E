import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    roc_curve,
    auc,
)


# Function to load and preprocess the data
def load_data(file_path):
    data = pd.read_csv(file_path)
    # Convert categorical variables into dummy/indicator variables
    data = pd.get_dummies(
        data,
        columns=[
            "area",
            "bathrooms",
            "bedrooms",
            "stories",
            "mainroad",
            "guestroom",
            "basement",
            "airconditioning",
            "parking",
            "prefarea",
            "furnishingstatus",
        ],
        drop_first=True,
    )
    return data


# Function to split data into features and target
def split_data(data):
    X = data.drop("price", axis=1)
    y = data["price"]
    return X, y


def random_forest_classification(X_train, y_train, n_estimators):
    rf = RandomForestClassifier(
        n_estimators=142, random_state=35, max_depth=46, max_features=6
    )
    rf.fit(X_train, y_train)
    return rf


# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm = (
        cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    )  # Normalize confusion matrix

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_true)))
    plt.xticks(tick_marks, ["Cheap", "Medium", "Expensive"], rotation=45)
    plt.yticks(tick_marks, ["Cheap", "Medium", "Expensive"])

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                "{:.2f}%".format(cm[i, j] * 100),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True Category")
    plt.xlabel("Predicted Category")
    plt.tight_layout()
    plt.show()


# Function to plot accuracy and classification report
def plot_performance_metrics(y_true, y_pred, plot=False):
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    plot_confusion_matrix(y_true, y_pred)


def compute_accuracy(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


# Function to perform logistic regression
def logistic_regression(X_train, y_train):
    lr = LogisticRegression(max_iter=50, solver="lbfgs", C=0.1)
    lr.fit(X_train, y_train)
    return lr


def plot_roc():
    file_path = "Housing_categorized.csv"
    data = load_data(file_path)
    X, y = split_data(data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=35
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Perform logistic regression
    lr_model = logistic_regression(X_train, y_train)

    # Predict probabilities for test set
    y_probs = lr_model.predict_proba(X_test)

    # Calculate ROC curve for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):  # Three classes
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve
    plt.figure()
    plt.plot(
        fpr[0], tpr[0], color="blue", lw=2, label="Cheap (AUC = %0.2f)" % roc_auc[0]
    )
    plt.plot(
        fpr[1], tpr[1], color="green", lw=2, label="Medium (AUC = %0.2f)" % roc_auc[1]
    )
    plt.plot(
        fpr[2], tpr[2], color="red", lw=2, label="Expensive (AUC = %0.2f)" % roc_auc[2]
    )
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Logistic Regression")
    plt.legend(loc="lower right")
    plt.show()


# Main function
def main():
    file_path = "Housing_categorized.csv"
    data = load_data(file_path)
    X, y = split_data(data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=35
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Perform logistic regression
    lr_model = logistic_regression(X_train, y_train)

    # Predictions
    y_pred_test = lr_model.predict(X_test)

    plot_performance_metrics(y_test, y_pred_test)

    accuracies = []
    x = []
    for i in range(1, 100):
        lr = LogisticRegression(max_iter=i)
        lr.fit(X_train, y_train)
        y_pred_test = lr.predict(X_test)  # Predictions were missing
        accuracies.append(accuracy_score(y_test, y_pred_test))
        x.append(i)  # Changed x[i - 1] to x.append(i)

    plt.plot(x, accuracies)  # Changed the width to a fixed value
    plt.xlabel("max_iter")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. max_iter")
    plt.show()


if __name__ == "__main__":
    plot_roc()
    main()
