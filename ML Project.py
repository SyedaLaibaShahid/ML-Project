
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np

# Step 1: Dataset Selection
diabetes = datasets.load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)  
df['Target'] = (diabetes.target > np.median(diabetes.target)).astype(int)
X = df.drop(columns=["Target"])
y = df["Target"]

# Step 2: Data Preprocessing
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
pca = PCA(n_components=10)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Step 3: Model Implementation and Optimization
results = []
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(random_state=42,  eval_metric="logloss"),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000)
}
param_grids = {
    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10]
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 6, 10]
    },
    "Logistic Regression": {
        "C": [0.1, 1, 10]
    }
}
for name, model in models.items():
    start_time = time.time()
    if name == "XGBoost":
        search = RandomizedSearchCV(model, param_grids[name], scoring="f1_macro", cv=3, n_iter=10, random_state=42)
    else:
        search = GridSearchCV(model, param_grids[name], scoring="f1_macro", cv=3)
    
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)
    exec_time = time.time() - start_time
    
    results.append({
        "Algorithm": name,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": "N/A",  # Not required as per user preference
        "Best Hyperparameters": search.best_params_ if hasattr(search, 'best_params_') else "N/A",
        "Execution Time (in sec)": exec_time,
        "Remarks": f"{name} executed successfully."
    })
    plt.figure(figsize=(6, 6))
    plt.matshow(cm, cmap="Blues", fignum=1)
    plt.colorbar()
    plt.xlabel("Predicted", fontsize= 16)
    plt.ylabel("Actual", fontsize= 16)
    plt.title(f"Confusion Matrix for {name}")
    plt.show()

# Step 4: Comparison and Insights
results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv("model_comparison_results.csv", index=False)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))  
fig.suptitle("Model Evaluation", fontsize=21)
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

for i, metric in enumerate(metrics):

    row = i // 2  # Determine the row (0 or 1)
    col = i % 2 
    axs[row, col].bar(results_df["Algorithm"], results_df[metric], color="yellowgreen", width=0.5)
    axs[row, col].set_title(f"{metric} Comparison", fontsize=16)
    axs[row, col].set_xlabel("Algorithm", fontsize=16)
    axs[row, col].set_ylabel(metric, fontsize=16)

plt.tight_layout()
plt.subplots_adjust(top=0.9)  # To give space for the main title
plt.show()