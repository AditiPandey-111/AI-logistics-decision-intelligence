import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# import pipeline functions
from data_pipeline import load_and_prepare_data , add_feature_engineering, prepare_training_data


def train_delay_model():

    # 1. load + merge + derive delay/cost metrics
    df = load_and_prepare_data()

    # 2. add route efficiency + anomaly + severity scores
    df = add_feature_engineering(df)

    # 3. extract ML-ready features + target
    X, y = prepare_training_data(df)

    # 4. split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # 5. initialize model
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        random_state=42
    )

    # 6. train model
    model.fit(X_train, y_train)

    # 7. predict
    y_pred = model.predict(X_test)

    # 8. evaluation
    print("\nModel Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 9. save model + feature list
    joblib.dump(model, "models/delay_predictor.pkl")
    joblib.dump(list(X.columns), "models/features.pkl")


    # save processed dataset for dashboard
    df.to_csv("data/processed_orders_full.csv", index=False)
    print("\nSaved processed dataset for dashboard use.")


    return model


if __name__ == "__main__":
    train_delay_model()
