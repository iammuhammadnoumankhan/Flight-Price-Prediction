# Flight Price Prediction

This project aims to predict the prices of flight tickets based on various features using machine learning techniques. The dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction), and the model is trained using a RandomForestRegressor with hyperparameter tuning via Grid Search.

## Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [License](#license)

## Dataset

The dataset used in this project is the "Flight Price Prediction" dataset from Kaggle. It includes various features such as:
- Date of Journey
- Departure Time
- Arrival Time
- Duration
- Total Stops
- Airline
- Source and Destination

[Kaggle Dataset Link](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)

## Installation

To run this project locally, you need to install the following dependencies:

```bash
pip install -r requirements.txt
```

---

## Exploratory Data Analysis

A thorough exploratory data analysis (EDA) was performed to understand the dataset and uncover hidden insights.

Key insights from EDA:
- Price varies significantly based on airlines.
- The number of stops has a notable impact on the flight price.
- Duration of flights also plays a crucial role in determining ticket prices.

## Data Preprocessing

The dataset was preprocessed to clean and prepare it for model training:
- Missing values were handled.
- Categorical features such as `Airline`, `Source`, and `Destination` were converted into numerical values using one-hot encoding.
- The `Duration` feature was converted into numerical format for model training.

## Model Training

The model used for flight price prediction is `RandomForestRegressor` from `sklearn`. The training process involved:
- Splitting the dataset into training and test sets.
- Fitting the `RandomForestRegressor` to the training data.

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train, y_train)
```

## Hyperparameter Tuning

Grid Search was used to perform hyperparameter tuning, optimizing parameters such as the number of estimators, maximum depth, and minimum samples split.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
     'n_estimators': [100,200,300,400,500],
     'max_depth': [None, 10,20,30],
     'min_samples_split': [2,5,10],
     'min_samples_leaf': [1,2,4],
     'max_features': ['auto', 'sqrt']
}

grid_search = GridSearchCV(estimator=reg, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
```

## Results

The final model's performance was evaluated on the test set using metrics such as R² score and Mean Absolute Error (MAE). The tuned RandomForestRegressor performed well, providing a satisfactory prediction accuracy for flight prices.

- **R² score**: _0.9804783568573654_
- **MAE**: _1424.608776910383_
- **MSE**: _10061668.962729437_
- **RMSE**: _3172.0133925835553_


---


**THANK YOU!!!**
```

