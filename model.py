from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from preprocessing import clean_data, split_X_y

clean_data()

df = pd.read_csv('assets/cleaned_data.csv')

X, y = split_X_y()

X_res, y_res = SMOTE().fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print('-----')
print("Classification Report:\n", classification_report(y_test, y_pred))
print('-----')
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the model
joblib.dump(model, 'model.joblib')
