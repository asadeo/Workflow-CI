import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    # Load Data
    df = pd.read_csv("data_siap_latih.csv")
    X = df.drop('Target', axis=1)
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Start Run & Training
    with mlflow.start_run() as run:
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)

        mlflow.sklearn.log_model(model, "model")

        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)
            
        print(f"Pelatihan selesai. Run ID: {run.info.run_id}")