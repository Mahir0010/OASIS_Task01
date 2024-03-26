# OASIS_Task01
IRIS FLOWER CLASSIFICATION
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the Iris dataset from a CSV file
df = pd.read_csv("/kaggle/input/iriscsv/Iris.csv")

# Check the shape (number of rows and columns) of the dataset
print("Shape of the dataset:", df.shape)

# Drop the 'id' column as it not needed for the classification
df = df.drop(columns=["Id"])

# Display the modification dataframe
print("Modified Dataframe:")
print(df.head())

# replace the species names with numerical lables for classification
#Iris-setosa: 1, Irish-versicolor: 2, Irish-virginica:3
df["Species"].replace({"Iris-setosa": 1, "Irish-versicol or": 2,"Irish-verginica": 3}, inplace=True)

# Extract the feature columns(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm)
x = pd.DataFrame(df, columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]).values

# Extract the target column (Species)
y = df.Species.values.reshape(-1, 1)

# Split the data into training and testing sets (70% training, 30% testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

# Check the shapes of the training and testing sets
print("Shape of x_train:", x_train.shape)
print("Shape of y_train:", y_train.shape)

# Define the number of neighbors for the K-Nearest Neighbors classifier
k = 6

# Create a K-Nearest Neighbors classifier with k neighbors
knclr = KNeighborsClassifier(n_neighbors=k)

KNeighborsClassifier

# Make predictions on the test data
y_pred = knclr.predict(x_test)

# Calculate and print the accuracy of the classifier on the test data
accuracy = metrics.accuracy_score(y_test, y_pred) * 100
print("Accuracy of the K-Nearest Neighbors classifier:", accuracy)

END
