import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data.csv")

# Features & target
X = df.drop("performance", axis=1)
y = df["performance"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Results
print("\n=== Model Evaluation ===\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ---------------------------
# 🔥 User Input Prediction
# ---------------------------
print("\n--- Predict New Employee ---")

hours = int(input("Enter working hours: "))
projects = int(input("Enter number of projects: "))
experience = int(input("Enter experience: "))

new_data = [[hours, projects, experience]]

prediction = model.predict(new_data)

print("\nPredicted Performance:", prediction[0])