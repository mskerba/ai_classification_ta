# ml_logic.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

# === Train model at startup ===
df_train = pd.read_excel("test01.xlsx")

df_train["text"] = (
    df_train["Systeme"].astype(str) + " | " +
    df_train["Description"].astype(str) + " | " +
    df_train["Description de l'équipement"].astype(str)
)

df_train = df_train.dropna(subset=["Fiabilité Intégrité", "Disponibilté", "Process Safety"])

X_train = df_train["text"]
y_train = df_train[["Fiabilité Intégrité", "Disponibilté", "Process Safety"]]

model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
    ("regressor", MultiOutputRegressor(Ridge()))
])
model.fit(X_train, y_train)


# === Predict and return JSON-ready records ===
def score_new_file_as_json(input_path: str):
    df_new = pd.read_excel(input_path)

    df_new["text"] = (
        df_new["Systeme"].astype(str) + " | " +
        df_new["Description"].astype(str) + " | " +
        df_new["Description de l'équipement"].astype(str)
    )

    X_new = df_new["text"]
    predictions = model.predict(X_new)

    predicted_scores = pd.DataFrame(predictions, columns=["Fiabilité Intégrité", "Disponibilté", "Process Safety"])
    predicted_scores = predicted_scores.round().clip(1, 5).astype(int)

    df_result = pd.concat([df_new, predicted_scores], axis=1)
    return df_result.to_dict(orient="records")
