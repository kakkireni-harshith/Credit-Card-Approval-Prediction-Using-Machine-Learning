# Credit Card Approval Prediction Using Machine Learning
[![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.x-150458?style=for-the-badge&logo=pandas)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?style=for-the-badge&logo=numpy)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24+-F7931E?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter)](https://jupyter.org/)


## Overview

Predicting whether a credit card application will be **approved** or **denied** is a critical task for financial institutions. Manual evaluation of these applications is often time-consuming, error-prone, and inefficient. This project automates the process using **machine learning** techniques, specifically **Logistic Regression** combined with **GridSearchCV** for hyperparameter tuning.

The dataset used is the **Credit Card Approval Dataset** from the **UCI Machine Learning Repository**, which contains anonymized information about applicants. The model we develop can streamline the credit card approval process by accurately predicting the approval outcome based on the provided features.


---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Getting Started](#getting-started)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Training and Evaluation](#model-training-and-evaluation)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Results](#results)
7. [Installation](#installation)
8. [Technologies Used](#technologies-used)
9. [Authors](#authors)
10. [License](#license)

---

## Project Structure

1. **Data Preprocessing**: 
    - Handle missing values using **mean imputation** for numerical data and **most frequent value imputation** for categorical data.
    - Convert categorical features to numeric using **Label Encoding**.
    - Scale numerical features using **MinMaxScaler**.

2. **Exploratory Data Analysis**: 
    - Analyze the features and labels to understand the relationships within the dataset.

3. **Model Training**:
    - Train a **Logistic Regression** model to classify credit card applications as approved or denied.

4. **Model Evaluation**:
    - Evaluate the model using accuracy and a confusion matrix to ensure balanced performance.

5. **Hyperparameter Tuning**:
    - Optimize model performance by tuning the hyperparameters with **GridSearchCV**.

---

## Getting Started

### Prerequisites

To run this project, you need to have the following libraries installed:

- **pandas**
- **numpy**
- **scikit-learn**

You can install these dependencies using the provided `requirements.txt` file.

### Dataset

The dataset used in this project is available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/credit+approval). It contains anonymized credit card applications with features such as income, debt, and prior defaults.

---

## Data Preprocessing

### 1. Handling Missing Values

To ensure that the machine learning model works optimally, it’s essential to address any missing values. In this project:
- Missing numeric values were replaced using **mean imputation**.
- Missing categorical values were replaced using the **most frequent value** in each column.

```python
# Replace missing values with NaN and fill numeric missing values with the column mean
cc_apps = cc_apps.replace("?", np.NaN)
cc_apps = cc_apps.fillna(cc_apps.mean())

# Impute missing categorical values with the most frequent value
for col in cc_apps.columns:
    if cc_apps[col].dtypes == 'object':
        cc_apps[col] = cc_apps[col].fillna(cc_apps[col].value_counts().index[0])
```

### 2. Encoding Categorical Data

We use **Label Encoding** to convert the categorical features into numeric values.

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in cc_apps.columns:
    if cc_apps[col].dtype == 'object':
        cc_apps[col] = le.fit_transform(cc_apps[col])
```

### 3. Feature Scaling

Scaling ensures that all features contribute equally to the model. We use **MinMaxScaler** to scale the features to a range between 0 and 1.

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
```

---

## Model Training and Evaluation

### 1. Splitting Data

We split the preprocessed dataset into **training** and **testing** sets, using 67% of the data for training and 33% for testing.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(rescaledX, y, test_size=0.33, random_state=42)
```

### 2. Training the Model

A **Logistic Regression** model is trained on the training data.

```python
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
```

### 3. Model Evaluation

We evaluate the model’s accuracy and analyze its performance using a **confusion matrix**.

```python
# Predictions
y_pred = logreg.predict(X_test)

# Accuracy and Confusion Matrix
from sklearn.metrics import confusion_matrix
print("Accuracy: ", logreg.score(X_test, y_test))
print(confusion_matrix(y_test, y_pred))
```

---

## Hyperparameter Tuning

To improve the model’s performance, we apply **GridSearchCV** for hyperparameter tuning. We focus on adjusting the `tol` and `max_iter` parameters of the Logistic Regression model.

```python
from sklearn.model_selection import GridSearchCV

# Define hyperparameters to tune
param_grid = {'tol': [0.01, 0.001, 0.0001], 'max_iter': [100, 150, 200]}
grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)

# Fit model and extract best parameters
grid_model_result = grid_model.fit(rescaledX, y)
print("Best score: %f using %s" % (grid_model_result.best_score_, grid_model_result.best_params_))
```

---

## Results

- **Initial Model Accuracy**: 83.33%
- **Confusion Matrix**:
    - True Positives: 98
    - True Negatives: 92
    - False Positives: 11
    - False Negatives: 27
- **Best Hyperparameters**:
    - `tol`: 0.01
    - `max_iter`: 100
- **Accuracy after Grid Search**: 85.51%

---

## Installation

To run this project on your local machine, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Hemanth1818/Credit-Card-Approval-Prediction-Using-Machine-Learning.git
   cd Credit-Card-Approval-Prediction-Using-Machine-Learning
   ```

2. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**:
   ```bash
   jupyter notebook
   ```

---

## Technologies Used

- **Python**: Main programming language for implementation.
- **pandas**: Data manipulation.
- **scikit-learn**: Machine learning library for model building.
- **numpy**: Numerical computations.
- **GridSearchCV**: Hyperparameter tuning.

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

This project provides an efficient solution for automating credit card approval predictions, leveraging machine learning techniques to optimize decision-making processes in financial institutions.
