# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset (cancer tumour data, the first column being the diagnosis class)
dataset = pd.read_csv("data/cancer_data.csv")

# Split the data into training and test set. This prevents over fitting (high bias).
# Ref. https://stackoverflow.com/questions/13411544/delete-a-column-from-a-pandas-dataframe
X = dataset.drop(columns=["diagnosis(1=m, 0=b)"], axis=1)  # Egenskaper (features)
y = dataset["diagnosis(1=m, 0=b)"]  # Target label, 1=malign, 0=benign

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
# To center the data (make it have zero mean and unit standard error), you subtract the mean and then divide the result by the standard deviation:
# x′=x−μσ
# Ref. https://datascience.stackexchange.com/questions/12321/whats-the-difference-between-fit-and-fit-transform-in-scikit-learn-models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a Support Vector Machine (SVM) classification model.
model = SVC(kernel="linear")
model.fit(X_train_scaled, y_train)


# Make predictions based on the test set
y_predModelSVC = model.predict(X_test_scaled)

# Evaluate the model
modelAccuracy = accuracy_score(y_test, y_predModelSVC)
modelReport = classification_report(y_test, y_predModelSVC)

print(f"Model accuracy (SVC): {modelAccuracy:.2f}")
print("Classification report (SVC):\n", modelReport)
