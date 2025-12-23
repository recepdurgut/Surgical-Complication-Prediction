# Surgical Complication Prediction using Random Forest and PCA

Python-based machine learning workflow for predicting surgical complications using patient features. The pipeline includes preprocessing, dimensionality reduction (PCA), Random Forest classification, evaluation, and visualization.

This workflow handles:
- Data preprocessing and scaling
- Dimensionality reduction with PCA (variance-based selection)
- Handling class imbalance with `class_weight='balanced'`
- Random Forest model training and evaluation
- Confusion matrix visualization
- Cross-validation for robust performance estimation
- Feature importance visualization

---

## Author

**Name:** Recep Durgut  
**Program:** M.Sc. Biotechnologie, 1st Semester  
**Student Number:** 0514594  

---

## Dependencies

- Python ≥ 3.8
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

Install Python packages via pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
````

---

## File Structure

```text
project_directory/
│
├── Surgical-deepnet.csv      # Input dataset with features and target
├── surgical_complications.py # Main Python workflow
├── README.md
└── results/                  # Output folder for plots and CSVs (optional)
```

---

## Usage

1. Ensure your dataset CSV is named `Surgical-deepnet.csv` or update the script accordingly.
2. Run the workflow:

```bash
python surgical_complications.py
```

3. Outputs:

* Console prints: model accuracy, classification report
* Confusion matrix heatmap displayed
* Feature importance barplot displayed
* Cross-validation accuracy printed (5-fold stratified)

4. Optional customization:

* Adjust PCA explained variance (currently 95%)
* Adjust Random Forest parameters (`n_estimators`, `max_depth`)
* Tune cross-validation folds or scoring metric

---

## Workflow Overview

1. **Data Loading:** Reads CSV and separates features (X) and target (y).
2. **Preprocessing:** StandardScaler and LabelEncoder; stratified train-test split.
3. **Dimensionality Reduction:** PCA components chosen to explain ~95% variance.
4. **Modeling:** Random Forest classifier with balanced class weights.
5. **Evaluation:** Accuracy, classification report, confusion matrix.
6. **Cross-validation:** Stratified 5-fold CV for model robustness.
7. **Visualization:** Confusion matrix and feature importance by PCA components.

---

## Notes

* Stratified split + class weighting addresses imbalance in complication class.
* PCA ensures dimensionality reduction without losing significant variance.
* Feature importance is reported in PCA component space (approximation of original features).
* All results are reproducible with random_state=0.

---

## References

1. Scikit-learn: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
2. PCA in machine learning: Jolliffe, I. T. *Principal Component Analysis*, Springer, 2002
3. Random Forest Classifier: Breiman, L. *Random Forests*, Machine Learning, 2001
4. Surgical complication prediction: [Dataset source / publication if available]

---

## License

For academic and research use only.
