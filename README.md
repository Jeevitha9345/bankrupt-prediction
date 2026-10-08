# 4-Class Corporate Bankruptcy Prediction Pipeline for Indian Companies

A machine learning system for predicting bankruptcy risk among Indian companies across four financial health states:
- **Bankrupt** (`Class 0`)
- **Financial Distress** (`Class 1`)
- **Probably Bankrupt** (`Class 2`)
- **Healthy** (`Class 3`)

---

## 1. How the Data Was Loaded

The project combines two raw external datasets without modifying the source files:
- **Dataset 1** (`Dataset1.xlsx` / `latest_bankrupt_dataset`): 1,026 rows containing distressed, bankrupt, and boundary Indian companies.
- **Dataset 2** (`Dataset2.xlsx` / `latest_healthy_dataset`): 2,643 rows primarily comprising healthy listed Indian companies.

Both datasets share the exact 66 financial features (`x1` through `x66`) and identification attributes (`Folder_name` for Company, `Feature_name` for Year, and `Label` for the target status).
- The raw datasets were converted and preserved as `data/dataset1_raw.csv` and `data/dataset2_raw.csv`.
- They were merged into `data/combined_raw.csv` (3,669 rows).
- 9 unlabeled rows (`Label` was NaN) were excluded, yielding 3,660 valid labeled observations saved into `data/cleaned_dataset.csv`.

---

## 2. How Preprocessing Works

1. **Target Encoding**:
   - `Bankrupt` $\rightarrow$ 0
   - `Financial Distress` $\rightarrow$ 1
   - `Probably Bankrupt` $\rightarrow$ 2
   - `Healthy` $\rightarrow$ 3
2. **Missing Value Imputation**:
   - `SimpleImputer(strategy='median')` was fitted **strictly on the training partition** and saved as `models/imputer.pkl`.
   - Applied to the test partition and future inputs to avoid data leakage.
3. **Feature Scaling**:
   - `StandardScaler()` was fitted strictly on the training partition and saved as `models/scaler.pkl` (used for Logistic Regression).
4. **Data Leakage Mitigation (Company-Level Split)**:
   - 116 companies appear across multiple financial years.
   - To prevent severe temporal data leakage, a **Stratified Group Split** (`StratifiedGroupKFold`, 80/20, `random_state=42`) was performed on `Company`.
   - **Zero companies overlap** between the 349 training companies (2,888 rows) and the 91 test companies (772 rows).
5. **Class Imbalance Mitigation**:
   - SMOTE (`imbalanced-learn`) was applied **strictly to the training data**, equalizing representation across classes while leaving the test data untouched.

---

## 3. How the Models Were Trained

Five multiclass algorithms were tuned and evaluated via 5-fold cross-validation on the training set:
1. **Logistic Regression**: Multinomial with L2 regularization (`C=10.0`, `solver='lbfgs'`).
2. **Random Forest**: 200 trees, unconstrained depth, `min_samples_split=2`, `min_samples_leaf=1`.
3. **XGBoost**: `objective='multi:softprob'`, 200 estimators, max depth 6, learning rate 0.1, subsample 0.8, colsample 0.8.
4. **CatBoost**: `loss_function='MultiClass'`, 250 iterations, depth 6, learning rate 0.1, L2 regularization 3.
5. **LightGBM**: `objective='multiclass'`, 200 estimators, num leaves 31, max depth 10, learning rate 0.1.

### Final Test Evaluation Comparison Table (Untouched Test Set):
| Model | Accuracy | Precision | Recall | Macro F1 | Weighted F1 | Bankrupt Recall | Distress Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Best Overall)** | **0.8446** | **0.6540** | **0.7277** | **0.6817** | **0.8546** | 0.6703 | **0.5517** |
| **LightGBM** | **0.8446** | 0.6487 | 0.6933 | 0.6664 | 0.8515 | 0.6374 | 0.4828 |
| **Random Forest** | 0.8303 | 0.6228 | 0.6565 | 0.6346 | 0.8424 | **0.6923** | 0.2759 |
| **CatBoost** | 0.8225 | 0.6212 | 0.7039 | 0.6504 | 0.8377 | 0.6703 | 0.4828 |
| **Logistic Regression** | 0.7150 | 0.5236 | 0.6346 | 0.5500 | 0.7507 | 0.6044 | 0.4483 |

---

## 4. How to Reproduce the Training

1. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the pipeline**:
   ```bash
   python scripts/step1_process_data.py
   python scripts/step2_preprocess_and_correlations.py
   python scripts/step3_train_tune_models.py
   ```
   All outputs, models, reports, and plots will be generated in `data/`, `models/`, and `results/`.

---

## 5. How to Load the Saved Models

In Python:
```python
import joblib
import pandas as pd

# Load pipeline components
imputer = joblib.load("models/imputer.pkl")
model = joblib.load("models/xgboost.pkl")
feature_names = joblib.load("models/feature_list.pkl")

# Prepare new raw company data (DataFrame with columns x1 to x66)
new_company_data = ... 

# Preprocess & predict
imputed_data = imputer.transform(new_company_data[feature_names])
probabilities = model.predict_proba(pd.DataFrame(imputed_data, columns=feature_names))
```

---

## 6. How to Make Predictions on a New Company

Use the standalone CLI script `predict.py`:

### Option A: Using Demo Mode
```bash
python predict.py --demo --model xgboost
```

### Option B: Providing a JSON string
```bash
python predict.py --model xgboost --json "{\"x1\": 0.41, \"x2\": 0.17, \"x3\": 0.0135, ...}"
```

### Option C: Providing a CSV file
```bash
python predict.py --model xgboost --csv path/to/company_features.csv
```

### Output Format:
```text
Prediction:
Probably Bankrupt

Probabilities:
Bankrupt: 0.1200
Financial Distress: 0.1800
Probably Bankrupt: 0.6300
Healthy: 0.0700
```
