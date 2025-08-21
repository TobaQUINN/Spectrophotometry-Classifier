import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load dataset
print("Loading dataset...")
df = pd.read_csv("data/preprocessed_spectrophotometer_data.csv")

# Features & target
X = df.drop("Class", axis=1)
y = df["Class"]

# Class names
class_names = ["Carbohydrate", "Lipid", "Protein"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Data split into training and testing sets.")


# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    class_weight="balanced"
)
rf.fit(X_train, y_train)
print("Model training completed.")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/Spectrophotometer-Classifier.pkl")
print("Saved trained model to models/Spectrophotometer-Classifier.pkl")

# Evaluation
y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.savefig("reports/confusion_matrix.png")
plt.show()

# Feature Importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

top_n = 15
plt.figure(figsize=(10, 6))
plt.title(f"Top {top_n} Feature Importances (Wavelengths)")
plt.bar(range(top_n), importances[indices][:top_n], align="center")
plt.xticks(range(top_n), X.columns[indices][:top_n], rotation=45)
plt.tight_layout()
plt.savefig("reports/feature_importances.png")
plt.show()

# Learning Curve
train_acc, test_acc = [], []
n_estimators_range = range(10, 201, 20)

for n in n_estimators_range:
    rf_temp = RandomForestClassifier(
        n_estimators=n,
        max_depth=None,
        random_state=42,
        class_weight="balanced"
    )
    rf_temp.fit(X_train, y_train)
    train_acc.append(rf_temp.score(X_train, y_train))
    test_acc.append(rf_temp.score(X_test, y_test))

plt.figure(figsize=(8, 5))
plt.plot(n_estimators_range, train_acc, label="Train Accuracy", marker="o")
plt.plot(n_estimators_range, test_acc, label="Test Accuracy", marker="o")
plt.xlabel("Number of Trees")
plt.ylabel("Accuracy")
plt.title("Learning Curve for Random Forest")
plt.legend()
plt.savefig("reports/learning_curve.png")
plt.show()


