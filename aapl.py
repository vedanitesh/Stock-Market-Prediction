import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
from datetime import datetime, timedelta
import streamlit as st

# Function to download stock data from Yahoo Finance
def download_stock_data(stock_symbol, start_date, end_date):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    return stock_data

# Function to preprocess stock data
def preprocess_data(stock_data):
    stock_data['Date'] = pd.to_datetime(stock_data.index)
    stock_data['Year'] = stock_data['Date'].dt.year
    stock_data['Month'] = stock_data['Date'].dt.month
    stock_data['Day'] = stock_data['Date'].dt.day
    
    # Add additional features such as moving averages, volume, etc.
    stock_data['50_MA'] = stock_data['Close'].rolling(window=50).mean()
    
    return stock_data.dropna()

# Function to create future dates for prediction
def create_future_dates(last_date, num_days):
    future_dates = [last_date + timedelta(days=i) for i in range(1, num_days + 1)]
    future_data = pd.DataFrame(future_dates, columns=['Date'])
    future_data['Year'] = future_data['Date'].dt.year
    future_data['Month'] = future_data['Date'].dt.month
    future_data['Day'] = future_data['Date'].dt.day
    return future_data

# Function to split data into features and target variable
def split_data(stock_data):
    X = stock_data[['Year', 'Month', 'Day', '50_MA']]
    y = stock_data['Close']
    return X, y

# Function to train regression models and evaluate performance
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # Initialize models
    models = {
        'Linear Regression': Ridge(),
        'Support Vector Machine': SVR(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor()
    }
    
    # Initialize pipelines for models requiring preprocessing
    pipelines = {
        'Linear Regression': Pipeline([('scaler', StandardScaler()), ('model', Ridge())]),
        'Support Vector Machine': Pipeline([('scaler', StandardScaler()), ('model', SVR())]),
        'Decision Tree': Pipeline([('model', DecisionTreeRegressor())]),
        'Random Forest': Pipeline([('model', RandomForestRegressor())])
    }

    # Hyperparameter grids for models
    param_grids = {
        'Linear Regression': {'model__alpha': [0.1, 1.0, 10.0]},
        'Support Vector Machine': {'model__C': [0.1, 1.0, 10.0], 'model__gamma': ['scale', 'auto']},
        'Decision Tree': {'model__max_depth': [None, 10, 20]},
        'Random Forest': {'model__n_estimators': [100, 200, 300], 'model__max_depth': [None, 10, 20]}
    }
    
    # Train models and evaluate performance
    model_results = {}
    for name, pipeline in pipelines.items():
        grid_search = GridSearchCV(pipeline, param_grid=param_grids[name], cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        y_pred = grid_search.best_estimator_.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        model_results[name] = {'RMSE': rmse, 'MAE': mae, 'R^2': r2, 'Best Params': grid_search.best_params_}
        joblib.dump(grid_search.best_estimator_, f"{name}_model.pkl")  # Save the best model
    
    return model_results

# Function to visualize predictions for each algorithm using subplots with shared axes
def visualize_predictions_subplots(stock_symbol, X_train, X_test, y_train, y_test, X_future, future_dates, best_model, model_results):
    plt.figure(figsize=(14, 10))  # Set the figure size
    plt.suptitle(f'Predictions vs Actual Prices for {stock_symbol}', fontsize=16)  # Set the main title

    for i, (name, model) in enumerate(best_model.items(), 1):
        plt.subplot(2, 2, i)  # Create subplots (2x2 grid)

        # Extract performance metrics
        rmse = model_results[name]['RMSE']
        mae = model_results[name]['MAE']
        r2 = model_results[name]['R^2']
        
        plt.title(f'{name} Predictions vs Actual Prices\nRMSE: {rmse:.2f}, MAE: {mae:.2f}, R^2: {r2:.2f}', fontsize=12)

        # Sort the data for consistent plotting
        X_train_sorted = X_train.sort_index()
        y_train_sorted = y_train.loc[X_train_sorted.index]
        X_test_sorted = X_test.sort_index()
        y_test_sorted = y_test.loc[X_test_sorted.index]

        # Predict using the trained model
        y_train_pred = model.predict(X_train_sorted)
        y_test_pred = model.predict(X_test_sorted)
        y_future_pred = model.predict(X_future.drop(columns=['Date']))

        # Plot actual prices for train and test datasets
        plt.plot(X_train_sorted.index, y_train_sorted, label='Actual Train Prices', color='green')
        plt.plot(X_test_sorted.index, y_test_sorted, label='Actual Test Prices', color='yellow')

        # Plot predicted prices for train, test, and future datasets
        plt.plot(X_train_sorted.index, y_train_pred, label='Predicted Train Prices', color='cyan')
        plt.plot(X_test_sorted.index, y_test_pred, label='Predicted Test Prices', color='blue')
        plt.plot(future_dates, y_future_pred, label='Predicted Future Prices', color='red')

        # Set axis labels
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to fit the main title
    st.pyplot(plt)

# Main function for Streamlit app
def main():
    st.title("Stock Price Prediction")

    # User inputs
    stock_symbol = st.text_input("Enter stock symbol (e.g., AAPL for Apple Inc.):")
    start_date = st.date_input("Enter start date:")
    end_date = st.date_input("Enter end date:")
    num_days = st.number_input("Enter the number of days for future prediction:", min_value=1, step=1)

    if st.button("Analyze Stock"):
        if not stock_symbol or not start_date or not end_date:
            st.error("Please fill out all fields.")
            return

        # Download stock data
        stock_data = download_stock_data(stock_symbol, str(start_date), str(end_date))

        if stock_data.empty:
            st.error("No data found for the given stock symbol and date range.")
            return

        # Preprocess data
        stock_data = preprocess_data(stock_data)

        # Split data into train and test sets
        X, y = split_data(stock_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)

        if X_train.empty or y_train.empty:
            st.error("Training set is empty after splitting. Please check the dataset.")
            return

        # Train and evaluate models
        model_results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        # Display model performance
        st.write("Model Performance:")
        for name, results in model_results.items():
            st.write(f"{name} - RMSE: {results['RMSE']:.2f}, MAE: {results['MAE']:.2f}, R^2: {results['R^2']:.2f}")

        # Create future dates for prediction
        last_date = stock_data.index[-1]
        future_data = create_future_dates(last_date, num_days)
        
        # Include the moving average for the future dates
        future_data['50_MA'] = stock_data['50_MA'].iloc[-1]  # Assuming the last available moving average value

        # Load best model
        best_model = {}
        for name in model_results.keys():
            best_model[name] = joblib.load(f"{name}_model.pkl")

        # Visualize predictions for each algorithm using subplots with shared axes
        visualize_predictions_subplots(stock_symbol, X_train, X_test, y_train, y_test, future_data, future_data['Date'], best_model, model_results)

if __name__ == "__main__":
    main()
