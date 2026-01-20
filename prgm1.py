import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

housing_df = pd.read_csv('housing.csv')
cancer_df = pd.read_csv('Breast_Cancer.csv')

# 1. regression (housing data)
# Preprocessing
housing_df['total_bedrooms'] = housing_df['total_bedrooms'].fillna(
    housing_df['total_bedrooms'].median())
housing_processed = pd.get_dummies(
    housing_df, columns=['ocean_proximity'], drop_first=True)
X_reg = housing_processed.drop('median_house_value', axis=1)
y_reg = housing_processed['median_house_value']
scaler_reg = StandardScaler()
X_reg_scaled = scaler_reg.fit_transform(X_reg)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg_scaled, y_reg, test_size=0.2, random_state=42)

# Models
reg_models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "ElasticNet": ElasticNet(),
    "KNN Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "AdaBoost": AdaBoostRegressor(random_state=42),
    "Extra Trees": ExtraTreesRegressor(n_estimators=50, random_state=42)
}

# training & evaluation
reg_results = []
for name, model in reg_models.items():
    model.fit(X_train_reg, y_train_reg)
    y_pred = model.predict(X_test_reg)

    mse = mean_squared_error(y_test_reg, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_reg, y_pred)

    reg_results.append(
        {"Model": name, "MSE": mse, "RMSE": rmse, "R2 Score": r2})

reg_df = pd.DataFrame(reg_results)

# 2. classification (breast cancer data)
# Preprocessing
le = LabelEncoder()
cancer_df['Status'] = le.fit_transform(cancer_df['Status'])
categorical_cols = cancer_df.select_dtypes(include=['object']).columns
cancer_processed = pd.get_dummies(
    cancer_df, columns=categorical_cols, drop_first=True)
X_clf = cancer_processed.drop('Status', axis=1)
y_clf = cancer_processed['Status']
scaler_clf = StandardScaler()
X_clf_scaled = scaler_clf.fit_transform(X_clf)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf_scaled, y_clf, test_size=0.2, random_state=42)

# Models
clf_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN Classifier": KNeighborsClassifier(),
    "SVC": SVC(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Gaussian NB": GaussianNB(),
    "LDA": LinearDiscriminantAnalysis(),
    "Extra Trees": ExtraTreesClassifier(n_estimators=50, random_state=42)
}

# training & evaluation
clf_results = []
for name, model in clf_models.items():
    model.fit(X_train_clf, y_train_clf)
    y_pred = model.predict(X_test_clf)

    acc = accuracy_score(y_test_clf, y_pred)
    prec = precision_score(y_test_clf, y_pred, average='weighted')
    rec = recall_score(y_test_clf, y_pred, average='weighted')

    clf_results.append({"Model": name, "Accuracy": acc,
                       "Precision": prec, "Recall": rec})

clf_df = pd.DataFrame(clf_results)

# 3. PLOTTING SEPARATELY


def remove_legend(ax):
    if ax.legend_:
        ax.legend_.remove()


# Plot 1: Regression R2 Score
plt.figure(figsize=(10, 6))
ax1 = sns.barplot(x="R2 Score", y="Model", data=reg_df.sort_values("R2 Score", ascending=False),
                  hue="Model", palette="viridis", dodge=False)
remove_legend(ax1)
plt.title("Regression Models - R2 Score")
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig("regression_r2_score.png")
plt.close()

# Plot 2: Regression RMSE
plt.figure(figsize=(10, 6))
ax2 = sns.barplot(x="RMSE", y="Model", data=reg_df.sort_values("RMSE"),
                  hue="Model", palette="magma", dodge=False)
remove_legend(ax2)
plt.title("Regression Models - RMSE (Lower is Better)")
plt.tight_layout()
plt.savefig("regression_rmse.png")
plt.close()

# plot 3: classification metrics (grouped bar)
plt.figure(figsize=(12, 8))
clf_melted = clf_df.melt(
    id_vars="Model", var_name="Metric", value_name="Score")
sns.barplot(x="Score", y="Model", hue="Metric",
            data=clf_melted, palette="deep")
plt.title("Classification Models - Accuracy, Precision, Recall")
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig("classification_metrics_bar.png")
plt.close()

# plot 4: classification heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(clf_df.set_index("Model"), annot=True, cmap="Blues", fmt=".3f")
plt.title("Classification Metrics Heatmap")
plt.tight_layout()
plt.savefig("classification_metrics_heatmap.png")
plt.close()
