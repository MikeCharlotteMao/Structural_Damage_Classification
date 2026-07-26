import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.ensemble import ExtraTreesClassifier

# =========================
# 1. Load labels and cached features
# =========================

labels = np.load("labels.npy")
X_features = np.load("X_features_frequency_curvature.npy")

print("Labels shape:", labels.shape)
print("Feature shape:", X_features.shape)

# =========================
# 2. Train/Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels,
)

# =========================
# 3. Extra Trees Model
# =========================


def create_extra_trees_model(
    n_estimators=1000,
    max_depth=None,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
):
    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )

# =========================
# 4. Hyperparameter Candidates
# =========================


hyperparameter_candidates = [

    {
        "n_estimators": 500,
        "max_depth": None,
        "max_features": "sqrt",
        "class_weight": "balanced",
    },

    {
        "n_estimators": 1000,
        "max_depth": None,
        "max_features": "sqrt",
        "class_weight": "balanced",
    },

    {
        "n_estimators": 500,
        "max_depth": 20,
        "max_features": "sqrt",
        "class_weight": "balanced",
    },

    {
        "n_estimators": 1000,
        "max_depth": 20,
        "max_features": "sqrt",
        "class_weight": "balanced",
    },

    {
        "n_estimators": 500,
        "max_depth": None,
        "max_features": "log2",
        "class_weight": "balanced",
    },

    {
        "n_estimators": 1000,
        "max_depth": None,
        "max_features": "log2",
        "class_weight": "balanced",
    },

]

# =========================
# 5. Feature Importance
# =========================

base_model = create_extra_trees_model(
    n_estimators=1000,
    max_depth=None,
    max_features="sqrt",
)

base_model.fit(X_train, y_train)

feature_importances = base_model.feature_importances_
feature_ranking = np.argsort(feature_importances)[::-1]

# =========================
# 6. Feature Selection
# =========================

top_feature_list = [20, 30, 40, 50, 60, 78]

performance_results = []

print("\n===== Extra Trees Feature Selection Results =====")

for top_n in top_feature_list:

    top_n = min(top_n, X_features.shape[1])

    selected_features = feature_ranking[:top_n]

    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    for params in hyperparameter_candidates:

        clf = create_extra_trees_model(**params)

        clf.fit(X_train_selected, y_train)

        y_pred = clf.predict(X_test_selected)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )
        recall = recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )
        f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        result = {
            "top_features": top_n,
            "accuracy": accuracy,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            **params,
        }

        performance_results.append(result)

        print(f"\nTop {top_n} Features | Params: {params}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")

# =========================
# 7. Save Results
# =========================

results_path = "extra_trees_top_feature_performance.csv"

with open(results_path, "w") as f:

    f.write(
        "top_features,accuracy,precision_macro,"
        "recall_macro,f1_macro,"
        "n_estimators,max_depth,max_features,class_weight\n"
    )

    for result in performance_results:

        f.write(
            f"{result['top_features']},"
            f"{result['accuracy']:.6f},"
            f"{result['precision_macro']:.6f},"
            f"{result['recall_macro']:.6f},"
            f"{result['f1_macro']:.6f},"
            f"{result['n_estimators']},"
            f"{result['max_depth']},"
            f"{result['max_features']},"
            f"{result['class_weight']}\n"
        )

print(f"\nSaved performance results to {results_path}")

# =========================
# 8. Best Model
# =========================

best_result = max(
    performance_results,
    key=lambda x: x["accuracy"],
)

best_top_n = best_result["top_features"]

best_features = feature_ranking[:best_top_n]

best_params = {
    "n_estimators": best_result["n_estimators"],
    "max_depth": best_result["max_depth"],
    "max_features": best_result["max_features"],
    "class_weight": best_result["class_weight"],
}

best_model = create_extra_trees_model(**best_params)

best_model.fit(
    X_train[:, best_features],
    y_train,
)

best_prediction = best_model.predict(
    X_test[:, best_features]
)

print("\n===== Best Extra Trees Result =====")

print(f"Best Top Features: {best_top_n}")
print(f"Best Hyperparameters: {best_params}")
print(f"Best Accuracy: {best_result['accuracy']:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        best_prediction,
        zero_division=0,
    )
)
