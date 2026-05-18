# 🚗 Car Price Prediction using Linear Regression

## 📌 Overview
This project is an end-to-end Machine Learning pipeline that predicts the selling price of used cars based on various vehicle features. Instead of relying on a standard out-of-the-box model, this project implements specific feature engineering and mathematical transformations to accurately model the real-world depreciation of vehicles.

## 📊 Dataset
The dataset used for this project is the **Car Details from Car Dekho** dataset. It includes historical data on used cars, including their selling price, kilometers driven, fuel type, transmission type, and ownership history.

## 🧠 Key Methodology & Feature Engineering
To achieve an accurate R² score and minimize the Mean Absolute Error (MAE), several advanced preprocessing steps were implemented:
* **Feature Engineering (Car Age):** The raw `year` column was converted into a `car_age` metric to provide the model with a continuous numerical variable that represents aging.
* **Handling Exponential Depreciation:** Car prices do not drop in a straight line; they lose value exponentially. To allow the Linear Regression model to fit this curve, a **Log-Transformation** (`np.log()`) was applied to the target variable (`selling_price`), and reversed (`np.exp()`) for final predictions.
* **Categorical Encoding:** Applied One-Hot Encoding (`pd.get_dummies`) to properly map nominal data (Fuel Type, Seller Type, Transmission, Owner) without assigning false mathematical weight.
* **Feature Scaling:** Utilized `StandardScaler` to normalize numerical features like kilometers driven and car age.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (LinearRegression, train_test_split, StandardScaler)
* **Data Visualization:** Matplotlib

## 📈 Results
The model successfully predicts car prices with high accuracy. The relationship between the actual prices and the model's predicted prices demonstrates a strong linear fit across various price ranges.


## 🚀 How to Run the Code
1. Clone this repository:
   ```bash
   git clone [https://github.com/ShehrozGhouri/Car-Price-Prediction-Model-Linear-Regression]
