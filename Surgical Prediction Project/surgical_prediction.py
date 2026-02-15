import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ---------------- Data Loading ----------------
try:
    # Ensure the file exists
    data = pd.read_csv("Surgical-deepnet.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    raise FileNotFoundError("Error: 'Surgical-deepnet.csv' not found.")

# ---------------- Preprocessing ----------------
# Separate features and target
X = data.drop('complication', axis=1)
y = data['complication']

# Encode target (Categorical to Numerical)
y = LabelEncoder().fit_transform(y)

# Train-test split (Stratified to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ---------------- Pipeline Construction ----------------
# Pipeline handles Imputation -> Scaling -> PCA -> Model
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Handle missing data
    ('scaler', StandardScaler()),                   # Normalize features
    ('pca', PCA(n_components=0.95)),                # Keep 95% variance
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

# ---------------- Model Training ----------------
print("Training model...")
pipeline.fit(X_train, y_train)

# ---------------- Prediction & Evaluation ----------------
y_pred = pipeline.predict(X_test)

# Metrics (Focus on F1 and Recall for medical data)
f1 = f1_score(y_test, y_pred)
print(f"\nModel F1 Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='g', cmap="Blues",
            xticklabels=['No Comp', 'Comp'],
            yticklabels=['No Comp', 'Comp'])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.show()

# ---------------- Cross-Validation ----------------
# Validate on the whole dataset to check stability
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1')
print(f"5-Fold CV F1 Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# ---------------- Component Importance ----------------
# We visualize PCA Component Importance, not Feature Importance
rf_model = pipeline.named_steps['rf']
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=np.arange(len(importances)), y=importances[indices], palette='viridis')
plt.title("Importance of PCA Components (Not Original Features)")
plt.xlabel("PCA Component Index")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()
