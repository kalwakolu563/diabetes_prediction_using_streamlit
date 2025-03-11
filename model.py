import pandas as pd
import numpy as np
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


df=pd.read_csv("diabetes.csv",encoding="ISO-8859-1")

df.columns

df.fillna(df.mean(), inplace=True)

# Verify no missing values exist now
print("\nMissing Values After Handling:\n", df.isnull().sum())

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Standardize numerical features (except categorical)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
}

# Train and evaluate models
results = []
for name, model in models.items():
    try:
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        end_time = time.time()

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Training Time": round(end_time - start_time, 2)
        })
    except Exception as e:
        print(f"Error in {name}: {str(e)}")

# Convert results to DataFrame and sort by accuracy
results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)

# Print results
print(results_df)

# Print classification reports for the top model
best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)

print(f"\nBest Model: {best_model_name}")
print(classification_report(y_test, y_pred_best))

# Save the best model
model_filename = "best_model.sav"
pickle.dump(best_model, open(model_filename, "wb"))

# Save the StandardScaler used for feature scaling
scaler_filename = "scaler.sav"
pickle.dump(scaler, open(scaler_filename, "wb"))

print(f"Best model saved as {model_filename}")
print(f"Scaler saved as {scaler_filename}")

