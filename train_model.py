import joblib
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score

X_train = joblib.load('X_train.pkl')
y_train = joblib.load('y_train.pkl')

model = xgb.XGBClassifier(random_state=42, scale_pos_weight=1)  # Balanced after SMOTE
model.fit(X_train, y_train)

joblib.dump(model, 'attrition_model.pkl')
print("Model trained and saved. AUC:", roc_auc_score(y_train, model.predict_proba(X_train)[:,1]))
