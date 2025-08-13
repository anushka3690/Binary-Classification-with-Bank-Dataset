# Binary-Classification-with-Bank-Dataset

From Kaggle Playground Series

## Project Overview

This project performs binary classification on a bank marketing dataset to predict whether a client will subscribe to a term deposit. The project explores various machine learning models and ensemble techniques to achieve the best ROC-AUC score.

## Dataset Information

- **Training Data Shape**: Contains multiple features including demographic information, previous contact details, and economic indicators
- **Target Variable**: Binary classification (subscription to term deposit: yes/no)
- **Data Quality**: No missing values detected in the dataset

## Workflow

### 1. Data Ingestion
- Load the dataset using pandas
- Initial data exploration and shape analysis

### 2. Exploratory Data Analysis (EDA)
- Comprehensive data inspection including shape, types, and missing values
- Analysis of categorical and numerical feature distributions
- Correlation analysis and visualization using seaborn and matplotlib
- Target variable distribution analysis to identify class imbalance

### 3. Data Preprocessing

#### Feature Engineering Steps:
- **Column Removal**: Dropped unnecessary temporal columns (`day`, `month`)
- **Skewness Treatment**: 
  - Applied **Yeo-Johnson Power Transformation** for highly skewed numerical features
  - Applied **Quantile Transformation** with normal distribution output for better feature scaling
- **Feature Type Conversion**: Converted specific numerical columns to categorical (e.g., `pdays`, `previous`)
- **Encoding**: 
  - **OneHotEncoder** for categorical variables (with `drop='first'` to avoid multicollinearity)
  - **StandardScaler** for numerical features
- **Pipeline Integration**: Created comprehensive preprocessing pipelines using ColumnTransformer

#### Class Imbalance Handling:
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance the target variable
- Achieved balanced dataset with equal representation of both classes

### 4. Model Development and Training

#### Models Implemented:

1. **Random Forest Classifier**
   - Accuracy: 91.25%
   - ROC-AUC: 97.17%

2. **Gradient Boosting Classifier**
   - Accuracy: 92.00%
   - Precision: 89.65%
   - Recall: 94.96%
   - F1-Score: 92.23%
   - ROC-AUC: **97.79%**

3. **Histogram-based Gradient Boosting**
   - Enhanced gradient boosting implementation
   - Cross-validated ROC-AUC: 97.57% ± 0.89%

4. **LightGBM Classifier**
   - Accuracy: 91.74%
   - ROC-AUC: 97.53%

5. **Stacking Ensemble**
   - **Base Learners**: Random Forest, Gradient Boosting, LightGBM
   - **Meta-learner**: Random Forest
   - **Final Results**:
     - Accuracy: **92.41%**
     - Precision: 91.03%
     - Recall: 94.10%
     - F1-Score: 92.54%
     - ROC-AUC: **97.93%** (Best Performance)

### 5. Best Model Performance

The **Stacking Ensemble** achieved the highest ROC-AUC score of **97.93%**, representing the optimal combination of:
- Multiple diverse base learners
- Advanced preprocessing pipeline
- SMOTE-balanced training data
- Comprehensive feature engineering

### 6. Model Pipeline Architecture

Each model follows a consistent pipeline structure:
```
Data → Preprocessing (ColumnTransformer) → SMOTE → Model → Predictions
```

Where preprocessing includes:
- Numerical features: StandardScaler
- Categorical features: OneHotEncoder
- Feature transformations: PowerTransformer & QuantileTransformer

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn (for SMOTE)
- lightgbm

## Usage

1. Place `train.csv` and `test.csv` in the project directory
2. Run the notebook `notebook.ipynb` step by step
3. Follow the complete workflow:
   - Data exploration and visualization
   - Feature engineering and preprocessing
   - Model training and evaluation
   - Ensemble method implementation
4. The final model generates predictions saved as `submission.csv`

## Key Results Summary

| Model | Accuracy | ROC-AUC | F1-Score | Notes |
|-------|----------|---------|-----------|-------|
| Random Forest | 91.25% | 97.17% | - | Single model baseline |
| Gradient Boosting | 92.00% | 97.79% | 92.23% | Strong individual performance |
| LightGBM | 91.74% | 97.53% | - | Fast and efficient |
| **Stacking Ensemble** | **92.41%** | **97.93%** | **92.54%** | **Best Overall Performance** |

## Model Architecture

The winning model uses a sophisticated stacking approach:
- **Level 1**: Random Forest, Gradient Boosting, LightGBM as base learners
- **Level 2**: Random Forest as meta-learner
- **Preprocessing**: Comprehensive pipeline with scaling, encoding, and transformations
- **Class Balance**: SMOTE oversampling technique

## Files Description

- `train.csv`: Training dataset with features and target variable
- `test.csv`: Test dataset for final predictions
- `notebook.ipynb`: Complete analysis and model development
- `submission.csv`: Final predictions from the best model
- `submission_ensemble.csv`: Ensemble model predictions
- `sample_submission.csv`: Sample submission format

## Performance Insights

The stacking ensemble achieved the best ROC-AUC score by:
1. Combining diverse algorithms with different strengths
2. Leveraging advanced preprocessing techniques
3. Handling class imbalance effectively with SMOTE
4. Using cross-validation for robust performance estimation