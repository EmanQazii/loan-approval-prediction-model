# Loan Approval Prediction

## Overview

This project develops a machine learning classification system for predicting whether a loan application should be **Approved** or **Rejected** based on applicant financial and demographic information.

The project goes beyond simply training multiple models. It includes exploratory data analysis, data quality investigation, feature engineering, preprocessing, class-distribution analysis, model comparison, cross-validation, feature importance analysis, and final model selection.

The objective was to build a reliable and interpretable machine learning solution while following a realistic end-to-end ML workflow.

---

## Problem Statement

Financial institutions need to assess loan applications efficiently and consistently. The decision depends on several factors such as:

- Applicant income
- CIBIL/credit score
- Loan amount
- Loan term
- Number of dependents
- Asset values
- Education
- Self-employment status

This project uses historical loan application data to predict the loan approval status of new applicants.

### Target

- `Approved` → 1
- `Rejected` → 0

---

## Dataset

**Dataset:** Loan Approval Prediction Dataset

The dataset contains **4,269 loan applications** and **13 original columns**.

### Original Features

| Feature | Description |
|---|---|
| `loan_id` | Unique application identifier |
| `no_of_dependents` | Number of dependents |
| `education` | Applicant education status |
| `self_employed` | Whether the applicant is self-employed |
| `income_annum` | Annual income |
| `loan_amount` | Requested loan amount |
| `loan_term` | Loan repayment term |
| `cibil_score` | Applicant credit score |
| `residential_assets_value` | Value of residential assets |
| `commercial_assets_value` | Value of commercial assets |
| `luxury_assets_value` | Value of luxury assets |
| `bank_asset_value` | Value of bank assets |
| `loan_status` | Target variable |

---

## Project Workflow

1. Data loading and initial inspection
2. Exploratory Data Analysis
3. Data quality and anomaly investigation
4. Feature engineering
5. Data preprocessing
6. Train-test split
7. Model training
8. Model evaluation
9. Random Forest stability validation
10. Feature importance analysis
11. Model comparison
12. Final model selection
13. Model serialization

---

## 1. Data Exploration

The dataset was initially examined using:

- Dataset shape
- Column names
- Data types
- Missing-value analysis
- Unique-value analysis
- Descriptive statistics
- Target distribution

The dataset contains:

- **4,269 rows**
- **13 original columns**
- **No missing values**

The target distribution was:

| Loan Status | Count | Percentage |
|---|---:|---:|
| Approved | 2,656 | 62.2% |
| Rejected | 1,613 | 37.8% |

The classes are moderately imbalanced but not severely skewed.

---

## 2. Data Quality Investigation

Several data-quality checks were performed before modeling.

### Negative Values

A negative value was found in:

`residential_assets_value`

There were **28 records** containing `-100000`.

These records were investigated and determined to represent an invalid/anomalous asset value rather than a meaningful negative residential asset value.

The anomaly was handled during data cleaning.

### Zero Values

Zero values were also investigated.

They were found in:

- `no_of_dependents`
- `residential_assets_value`
- `commercial_assets_value`
- `bank_asset_value`

These were not automatically treated as missing values because zero can represent a legitimate value for some of these variables.

---

## 3. Exploratory Data Analysis

EDA included:

- Numerical feature distributions
- Categorical feature distributions
- Target class distribution
- Boxplots by loan status
- Correlation matrix
- Skewness analysis
- CIBIL score analysis
- Loan-to-income analysis

### Important Findings

The CIBIL score showed the strongest relationship with loan approval.

Approval rates by CIBIL band showed a very strong separation between approved and rejected applications.

Applicants with CIBIL scores above 600 were overwhelmingly approved in this dataset, while lower CIBIL scores were associated with significantly higher rejection rates.

---

## 4. Correlation Analysis

Several strong correlations were identified between financial features.

Examples include:

- Income and luxury assets: approximately 0.93
- Income and loan amount: approximately 0.93
- Loan amount and luxury assets: approximately 0.86
- Income and bank assets: approximately 0.85

These relationships indicate that several asset-related features are strongly associated with applicant income and loan size.

Because of these relationships, feature importance was interpreted carefully rather than assuming that every correlated variable independently contributes the same amount of information.

---

## 5. Feature Engineering

A `loan_to_income_ratio` feature was created:

```text
loan_to_income_ratio = loan_amount / income_annum
```

This feature represents the size of the requested loan relative to the applicant's annual income.

It provides a more meaningful financial relationship than considering loan amount and income independently.

A CIBIL score band was also explored during EDA for understanding approval patterns. It was not used as an additional predictive feature because the original numerical CIBIL score already contains the underlying information.

---

## 6. Data Preprocessing

The following preprocessing steps were applied:

### Numerical Features

Numerical features were passed through the numerical preprocessing pipeline.

### Categorical Features

Categorical variables such as:

- Education
- Self-employment status

were encoded using One-Hot Encoding.

### Target Encoding

The target was encoded as:

```text
Rejected → 0
Approved → 1
```

### Identifier Removal

`loan_id` was removed before modeling because it is an identifier and does not contain meaningful predictive information.

---

## 7. Train-Test Split

The dataset was divided into:

- **80% training data**
- **20% testing data**

A stratified split was used to maintain the original target-class distribution in both sets.

Final sizes:

- Training samples: **3,415**
- Testing samples: **854**

---

## 8. Models

Three classification algorithms were evaluated:

### Logistic Regression

Used as the linear baseline model.

### Random Forest

Used to capture nonlinear relationships and interactions between applicant features.

### XGBoost

Used as a gradient-boosting model for comparison against Random Forest.

---

## 9. Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

These metrics provide a more complete evaluation than accuracy alone.

---

## 10. Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 91.33% | 92.08% | 94.16% | 93.11% | 97.31% |
| Random Forest | **99.53%** | **99.25%** | **100.00%** | **99.62%** | **99.91%** |
| XGBoost | **99.53%** | **99.25%** | **100.00%** | **99.62%** | 99.85% |

Random Forest was selected as the final model because it matched XGBoost on the main classification metrics while achieving the highest ROC-AUC.

---

## 11. Random Forest Stability Validation

Because the Random Forest achieved unusually high performance, 5-fold Stratified Cross-Validation was performed on the training data.

The test set was kept completely separate during this validation.

| Metric | Mean | Standard Deviation |
|---|---:|---:|
| Accuracy | 99.59% | 0.28% |
| Precision | 99.44% | 0.43% |
| Recall | 99.91% | 0.19% |
| F1 | 99.67% | 0.23% |
| ROC-AUC | 99.80% | 0.18% |

The low standard deviations indicate that the Random Forest maintained highly consistent performance across different training subsets.

The cross-validation results were also consistent with the held-out test-set results.

---

## 12. CIBIL-Only Experiment

An additional experiment was performed using only the CIBIL score to understand how much of the predictive performance was driven by credit score.

### CIBIL-only Random Forest

| Metric | Score |
|---|---:|
| Accuracy | 96.02% |
| Precision | 98.25% |
| Recall | 95.29% |
| F1 | 96.75% |
| ROC-AUC | 96.96% |

The full feature model improved accuracy from **96.02% to 99.53%**.

This demonstrates that although CIBIL score is the dominant predictor, the remaining applicant and loan characteristics provide meaningful additional information.

---

## 13. Feature Importance

Random Forest feature importance showed that CIBIL score was the dominant predictive feature.

| Feature | Importance |
|---|---:|
| CIBIL Score | 80.00% |
| Loan Term | 7.29% |
| Loan-to-Income Ratio | 4.49% |
| Loan Amount | 1.60% |
| Luxury Assets | 1.26% |
| Commercial Assets | 1.21% |
| Residential Assets | 1.15% |
| Income | 1.09% |
| Bank Assets | 1.06% |
| Dependents | 0.54% |
| Education | 0.17% |
| Self-employed | 0.16% |

CIBIL score was therefore the dominant predictive feature in the trained Random Forest.

These values represent impurity-based feature importance and should not be interpreted as causal effects.

---

## 14. Class Imbalance Handling

The target distribution was approximately:

- 62.2% Approved
- 37.8% Rejected

This represents moderate rather than severe class imbalance.

Aggressive oversampling techniques such as SMOTE were not used because the baseline Random Forest already achieved excellent performance for both classes.

In particular, the model achieved approximately 99%+ recall for rejected applications and 100% recall for approved applications on the test set.

Therefore, additional synthetic oversampling was not considered necessary for this dataset.

---

## 15. Model Optimization Decision

Extensive hyperparameter optimization was not pursued because the baseline Random Forest already demonstrated near-perfect performance and highly consistent cross-validation results.

The model achieved:

- 99.53% test accuracy
- 99.62% test F1 score
- 99.91% test ROC-AUC
- 99.59% mean cross-validation accuracy
- 0.28% cross-validation accuracy standard deviation

Further optimization was therefore unlikely to provide a meaningful practical improvement relative to the additional complexity and computational cost.

---

## 16. Final Model

The final selected model is:

**Random Forest Classifier**

Reasons for selection:

- Highest overall baseline performance
- Highest ROC-AUC among the evaluated models
- Excellent precision and recall
- Excellent F1 score
- Highly stable cross-validation performance
- Strong performance without aggressive resampling
- No extensive hyperparameter tuning required

The complete preprocessing and Random Forest pipeline has been saved as:

```text
models/best_random_forest.pkl
```

---

## 17. Project Outputs

The project stores trained models in:

```text
models/
```

and generated results in:

```text
outputs/
```

These include:

- Model comparison results
- Random Forest cross-validation results
- Feature importance results
- EDA visualizations
- Model evaluation visualizations

---

## 18. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib
- Jupyter Notebook

---

## 19. How to Run

### 1. Clone the repository

```bash
git clone <repository-url>
cd loan-approval-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/loan_approval_prediction.ipynb
```

and run the cells sequentially.

---

## 20. Limitations

Although the models achieved very high performance, the dataset is relatively small and highly structured.

The results should therefore not be interpreted as representative of real-world banking performance.

Additional real-world data would be required before deployment, including:

- Historical repayment behavior
- Debt-to-income information
- Employment stability
- Existing liabilities
- Loan purpose
- Credit history duration
- Macroeconomic factors
- Fairness and bias analysis

The dataset's extremely strong relationship between CIBIL score and loan status also means that performance may not generalize to other lending datasets.

---

## 21. Conclusion

This project demonstrates an end-to-end machine learning workflow for loan approval prediction.

The analysis showed that **CIBIL score is the strongest predictive factor**, while loan term, loan-to-income ratio, loan amount, and asset-related variables provide additional predictive information.

Among the three evaluated algorithms, **Random Forest achieved the strongest overall performance**, with 99.53% test accuracy and 99.91% ROC-AUC.

Five-fold cross-validation further demonstrated that the model's performance was highly stable, with 99.59% mean accuracy and only 0.28% standard deviation.

The final Random Forest pipeline was therefore selected as the project's best model.
