import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# 1. Load the dataset
# -------------------------------
df = pd.read_csv("dataset.csv")
print("Dataset loaded successfully!\n")
print("Columns in dataset:", df.columns.tolist())

# -------------------------------
# 2. Ask user for target column
# -------------------------------
target = input("\nEnter the target column name (example: 0, 1, 2, 3, or 4): ")

if target not in df.columns.astype(str):
    print("\n❌ Invalid column name! Exiting...")
    exit()

print(f"\n✔ Target column set to: {target}")

# -------------------------------
# 3. Define features and target
# -------------------------------
X = df.drop(columns=[target])
y = df[target]

# -------------------------------
# 4. Split data
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 5. Scale features
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 6. Train Logistic Regression model
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# -------------------------------
# 7. Predictions
# -------------------------------
y_pred = model.predict(X_test_scaled)

# -------------------------------
# 8. Evaluation
# -------------------------------
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
