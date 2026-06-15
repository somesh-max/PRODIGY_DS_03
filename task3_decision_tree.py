
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv(
    r"C:\Users\somes\Documents\pridogy\task-3\bank.csv",
    sep=";"
)

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())

# ==========================
# Prepare Features and Target
# ==========================

# Convert target column to numeric
y = df["y"].map({"yes": 1, "no": 0})

# Convert all categorical columns to numeric
X = pd.get_dummies(df.drop("y", axis=1), drop_first=True)

print("\nShape of Features:", X.shape)
print("Shape of Target:", y.shape)

# ==========================
# Split Dataset
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Train Decision Tree Model
# ==========================
model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)

model.fit(X_train, y_train)

# ==========================
# Make Predictions
# ==========================
y_pred = model.predict(X_test)

# ==========================
# Evaluate Model
# ==========================
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================
# Visualize Decision Tree
# ==========================
plt.figure(figsize=(20, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree Classifier - Bank Marketing Dataset")
plt.show()
