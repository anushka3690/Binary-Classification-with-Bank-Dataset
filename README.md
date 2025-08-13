# Binary-Classification-with-Bank-Dataset

From Kaggle Playground Series

## Project Overview

This project performs binary classification on a bank marketing dataset. The goal is to predict whether a client will subscribe to a term deposit.

## Workflow

### 1. Data Ingestion
- Load the dataset using pandas.

### 2. Exploratory Data Analysis (EDA)
- Inspect data shape, types, and missing values.
- Analyze categorical and numerical columns.
- Visualize distributions and relationships using seaborn and matplotlib.

### 3. Data Preprocessing
- Drop unnecessary columns (`day`, `month`, etc.).
- Handle categorical variables and encode them.
- Address skewness in numerical columns (e.g., `balance`) using transformations like Yeo-Johnson and QuantileTransformer.
- Convert certain numerical columns to categorical (e.g., `pdays`, `previous`).

### 4. Feature Engineering
- Analyze and transform features for better model performance.
- Visualize feature distributions before and after transformation.

### 5. Handling Imbalanced Data
- Use SMOTE to balance the target variable.

### 6. Model Preparation
- Split the data into training and test sets.
- Build a machine learning pipeline with preprocessing (scaling, encoding) and a classifier (RandomForest, LogisticRegression).

### 7. Model Training
- Fit the pipeline on the balanced dataset.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn

## Usage

1. Place `train.csv` in the project directory.
2. Run the notebook `notebook.ipynb` step by step.
3. Follow the workflow for data analysis, preprocessing, and model training.

##