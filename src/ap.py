# your code here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

local_path = "C:\4geeks\machine-learning-python-template\data\raw\bank-marketing-campaign-data.csv"
df = pd.read_csv(local_path, sep=';')

df['was_previously_contacted'] = df['pdays'].apply(lambda x: 0 if x == 999 else 1)
df['y'] = df['y'].map({'yes': 1, 'no': 0})

features = [col for col in df.columns if col != 'y']
numeric_features = df[features].select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df[features].select_dtypes(include='object').columns.tolist()

if 'was_previously_contacted' in numeric_features:
    numeric_features.remove('was_previously_contacted')
    categorical_features.append('was_previously_contacted')

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X = df.drop('y', axis=1)
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print(f"\nDatos divididos en entrenamiento y prueba:")
print(f"X_train original shape: {X_train.shape}")
print(f"y_train original shape: {y_train.shape}")
print(f"X_test original shape: {X_test.shape}")
print(f"y_test original shape: {y_test.shape}")

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
print(f"\nDatos preprocesados y listos para el modelado:")
print(f"X_train_processed shape: {X_train_processed.shape}")
print(f"X_test_processed shape: {X_test_processed.shape}")

print("-"*200)

model = LogisticRegression(random_state=42, solver='liblinear')

model.fit(X_train_processed, y_train)

y_pred = model.predict(X_test_processed)
y_pred_proba = model.predict_proba(X_test_processed)[:, 1]

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))

print("ROC AUC Score:")
print(roc_auc_score(y_test, y_pred_proba))

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='Logistic Regression')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

print("-"*200)

pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('logistic', LogisticRegression(random_state=42, solver='liblinear', max_iter=1000))
])

param_grid = {
    'logistic__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'logistic__penalty': ['l1', 'l2']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
grid_search.fit(X_train_processed, y_train)
print("Best Parameters:", grid_search.best_params_)
print("Best ROC AUC Score:", grid_search.best_score_)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test_processed)
y_pred_proba = best_model.predict_proba(X_test_processed)[:, 1]

cm_optimized = confusion_matrix(y_test, y_pred)
print("Confusion Matrix (Optimized):")
print(cm_optimized)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_optimized, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Optimized)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("Classification Report (Optimized):")
print(classification_report(y_test, y_pred))

print("ROC AUC Score (Optimized):")
print(roc_auc_score(y_test, y_pred_proba))

print("ROC Curve (Optimized):")
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='Logistic Regression')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

