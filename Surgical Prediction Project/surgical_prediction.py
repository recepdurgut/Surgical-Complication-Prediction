import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

# ---------------- Data Loading ----------------
try:
    data = pd.read_csv("Surgical-deepnet.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    raise FileNotFoundError("Error: 'Surgical-deepnet.csv' not found. Please ensure the file is in the directory.")

# ---------------- Preprocessing ----------------
X = data.drop('complication', axis=1)
y = data['complication']

# Encode target
y = LabelEncoder().fit_transform(y)

# Train-test split (stratified to preserve class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

# ---------------- Pipeline: StandardScaler + PCA + RF ----------------
# Determine number of PCA components to explain ~95% variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
pca_temp = PCA(0.95)
pca_temp.fit(X_train_scaled)
n_components = pca_temp.n_components_
print(f"PCA components selected to explain 95% variance: {n_components}")

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=n_components)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=0, class_weight='balanced'))
])

# ---------------- Model Training ----------------
pipeline.fit(X_train, y_train)

# ---------------- Prediction ----------------
y_pred = pipeline.predict(X_test)

# ---------------- Evaluation ----------------
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='g', cmap="YlGnBu",
            xticklabels=['No Complication','Complication'],
            yticklabels=['No Complication','Complication'])
plt.title("Confusion Matrix: Surgical Complication Prediction", fontsize=14)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ---------------- Cross-validation ----------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
print(f"5-Fold CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# ---------------- Feature Importance ----------------
# Extract RF from pipeline
rf_model = pipeline.named_steps['rf']
pca_model = pipeline.named_steps['pca']

# Feature importance in original feature space approximation
importances = rf_model.feature_importances_
component_indices = np.arange(len(importances))

plt.figure(figsize=(10,6))
sns.barplot(x=component_indices, y=importances, palette='viridis')
plt.title("Feature Importance by Random Forest Components")
plt.xlabel("PCA Component Index")
plt.ylabel("Importance")
plt.show()
