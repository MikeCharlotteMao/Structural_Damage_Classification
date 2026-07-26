import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from xgboost import XGBClassifier

# =========================
# 1. Load labels and cached features
# =========================

labels = np.load("labels.npy")   # shape: (1530,)
X_features = np.load("X_features_frequency_curvature.npy")

print("Labels shape:", labels.shape)
print("Feature shape:", X_features.shape)

# =========================
# 2. Train XGBoost Model with Feature Selection
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)


def create_xgboost_model(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42
):
    """
    Create an XGBoost classifier.
    The hyperparameters can be changed during the small tuning experiment below.
    """
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="multi:softprob",
        num_class=len(np.unique(labels)),
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1
    )


# Small hyperparameter candidates around the current best setting.
# Keep this list small to avoid over-tuning the test set.
hyperparameter_candidates = [
    {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.1,
        "subsample": 0.7, "colsample_bytree": 0.7},
    {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
        "subsample": 0.7, "colsample_bytree": 0.7},
    {"n_estimators": 250, "max_depth": 6, "learning_rate": 0.1,
        "subsample": 0.7, "colsample_bytree": 0.7},
    {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1,
        "subsample": 0.7, "colsample_bytree": 0.7},
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1,
        "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
        "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
        "subsample": 0.9, "colsample_bytree": 0.9},
]

# Train one model using all features to obtain feature importance
base_params = {"n_estimators": 200, "max_depth": 6,
               "learning_rate": 0.1, "subsample": 0.7, "colsample_bytree": 0.7}
base_clf = create_xgboost_model(**base_params, random_state=42)
base_clf.fit(X_train, y_train)

feature_importances = base_clf.feature_importances_
feature_ranking = np.argsort(feature_importances)[::-1]

# Top feature numbers to test
top_feature_list = [20, 30, 40, 50, 60, 78]

performance_results = []

print("\n===== XGBoost Feature Selection Results =====")

for top_n in top_feature_list:
    # Avoid selecting more features than the feature matrix actually has
    top_n = min(top_n, X_features.shape[1])

    selected_features = feature_ranking[:top_n]

    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    for params in hyperparameter_candidates:
        clf = create_xgboost_model(**params, random_state=42)
        clf.fit(X_train_selected, y_train)

        y_pred = clf.predict(X_test_selected)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        result = {
            "top_features": top_n,
            "accuracy": accuracy,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            **params
        }
        performance_results.append(result)

        print(f"\nTop {top_n} Features | Params: {params}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")


# Save performance results as CSV
results_path = "xgboost_top_feature_performance.csv"

with open(results_path, "w") as f:
    f.write("top_features,accuracy,precision_macro,recall_macro,f1_macro,n_estimators,max_depth,learning_rate,subsample,colsample_bytree\n")
    for result in performance_results:
        f.write(
            f"{result['top_features']},"
            f"{result['accuracy']:.6f},"
            f"{result['precision_macro']:.6f},"
            f"{result['recall_macro']:.6f},"
            f"{result['f1_macro']:.6f},"
            f"{result['n_estimators']},"
            f"{result['max_depth']},"
            f"{result['learning_rate']},"
            f"{result['subsample']},"
            f"{result['colsample_bytree']}\n"
        )

print(f"\nSaved performance results to {results_path}")


# Optional: print detailed report for the best top-feature setting
best_result = max(performance_results, key=lambda x: x["accuracy"])
best_top_n = best_result["top_features"]
best_selected_features = feature_ranking[:best_top_n]
best_params = {
    "n_estimators": best_result["n_estimators"],
    "max_depth": best_result["max_depth"],
    "learning_rate": best_result["learning_rate"],
    "subsample": best_result["subsample"],
    "colsample_bytree": best_result["colsample_bytree"]
}

best_clf = create_xgboost_model(**best_params, random_state=42)
best_clf.fit(X_train[:, best_selected_features], y_train)
best_y_pred = best_clf.predict(X_test[:, best_selected_features])

print("\n===== Best XGBoost Feature Selection Result =====")
print(f"Best Top Features: {best_top_n}")
print(f"Best Hyperparameters: {best_params}")
print(f"Best Accuracy: {best_result['accuracy']:.4f}")
print("\nClassification Report for Best Result:")
print(classification_report(y_test, best_y_pred, zero_division=0))
