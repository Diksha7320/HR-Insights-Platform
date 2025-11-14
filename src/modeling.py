import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# Utility to safely pull values from classification report
def safe_get(report, label, metric):
    try:
        return round(report[str(label)][metric], 3)
    except KeyError:
        return "N/A"


def get_text_explanations(feature_importances, df, target_col):
    """
    Creates human-readable SHAP-like contributions based on:
    feature importance * mean difference between classes.
    """

    df0 = df[df[target_col] == 0]  # Non Attrition
    df1 = df[df[target_col] == 1]  # Attrition

    explanations = []

    for _, row in feature_importances.iterrows():
        feature = row["feature"]
        importance = row["importance"]

        if feature in df.columns:
            diff = df1[feature].mean() - df0[feature].mean()

            score = round(abs(diff * importance), 3)

            if diff > 0:
                explanations.append(f"+ {feature} (increases attrition risk by {score})")
            else:
                explanations.append(f"- {feature} (decreases attrition risk by {score})")

    return explanations[:10]  # Top 10 explanatory features


def train_model(df, target_col):

    df = df.copy()

    # 1️⃣ Split into X, y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2️⃣ Safe SMOTE
    min_class = y.value_counts().min()

    if min_class > 1:
        safe_k = min(5, min_class - 1)
        sm = SMOTE(k_neighbors=safe_k)
        X_res, y_res = sm.fit_resample(X, y)

        # Convert back to DF
        X_res = pd.DataFrame(X_res, columns=X.columns)
        y_res = pd.Series(y_res, name=target_col)
    else:
        X_res, y_res = X, y

    # 3️⃣ Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.25, random_state=42
    )

    # 4️⃣ Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5️⃣ Predictions
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    # 6️⃣ Clean metrics
    clean_metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 3),
        "precision_yes": safe_get(report, 1, "precision"),
        "recall_yes": safe_get(report, 1, "recall"),
        "f1_yes": safe_get(report, 1, "f1-score"),
        "precision_no": safe_get(report, 0, "precision"),
        "recall_no": safe_get(report, 0, "recall"),
        "f1_no": safe_get(report, 0, "f1-score"),
    }

    # 7️⃣ Feature importance
    feature_importances = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    # 8️⃣ Text explanations
    explanations = get_text_explanations(feature_importances, df, target_col)

    return model, clean_metrics, feature_importances, explanations


