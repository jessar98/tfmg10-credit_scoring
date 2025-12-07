from xgboost import XGBClassifier
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # Cargar dataset
    df = pd.read_csv("data/g1tfm-test-dataset-complete.csv")

    X = df.drop("pago", axis=1)
    y = df["pago"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Definir el modelo (misma configuración que tu notebook)
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    # Entrenar
    model.fit(X_train, y_train)

    # Guardar modelo como artifact
    joblib.dump(model, "artifacts/g10tfm_model_test1.pkl")

if __name__ == "__main__":
    main()
