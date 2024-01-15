# LSTM split into functions
# Importing relevant libraries

from datetime import datetime, timedelta
import math
import pandas_datareader.data as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from  matplotlib import pyplot as plt
import yfinance as yf
import time

plt.style.use('fivethirtyeight')

# Collects last 7 years of data from Yahoo Finance API
# Input: string representing stock symbol
def gather_stock_data(input):
    yf.pdr_override()
    today = datetime.today()
    end_date = today.strftime('%Y-%m-%d')
    start_date = (today - timedelta(weeks=7*52)).strftime('%Y-%m-%d')
    df = web.get_data_yahoo(
        f'{input}', start=f'{start_date}', end=f'{end_date}')
    return df.to_json()

def gather_stock_data_df(input):
    yf.pdr_override()
    today = datetime.today()
    end_date = today.strftime('%Y-%m-%d')
    start_date = (today - timedelta(weeks=7*52)).strftime('%Y-%m-%d')
    df = web.get_data_yahoo(
        f'{input}', start=f'{start_date}', end=f'{end_date}')
    print(df.head(1)['Close'])
    print(df.shape)
    return df

# Currently causes an error when called!
def visualize_dataframe(yahoo_finance_response_df):
    plt.figure(figsize=(16, 8))
    plt.title = 'Close Price History'
    plt.plot(yahoo_finance_response_df['Close'])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price ($USD)', fontsize=18)
    plt.show()

# Create a dataframe with only closing prices and scale for preprocessing
def scale_closing_prices(yahoo_finance_response_df):
    data = yahoo_finance_response_df.filter(['Close'])
    # Convert dataframe to a numpy array
    dataset = data.values
    # Compute row count to train the model on => 80% of data for training
    training_data_len = math.ceil(len(dataset) * 0.8)
    print("Training data length is: ", training_data_len)
    # Computes min and max values for scaling, then transforms. Standardizes data from 0 to 1 inclusive
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)   
    scaled_data
    return data, dataset, scaled_data, training_data_len, scaler

# Create the training data set
def create_training_data_set(scaled_closing_prices, training_data_len):
    # Create the scaled training data set
    train_data = scaled_closing_prices[0:training_data_len, :]
    # Split data into x_train and y_train data sets
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])   # Positions 0 to 59
        y_train.append(train_data[i, 0])        # Position 60

    # Convert x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    # Reshape the x_train data set => Add third column

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_train.shape

    return x_train, y_train

# Build the LSTM Model
def build_lstm_model(x_train, y_train):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False)) 
    model.add(Dense(32))
    model.add(Dense(1))
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Create the testing data set
def test_data_set(dataset, scaled_closing_prices, training_data_len, model, scaler):
    # Array containing scaled values from index 1543 to 2003
    test_data = scaled_closing_prices[training_data_len - 60: , :]
    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    
    # Convert the data to a numpy array
    x_test = np.array(x_test)   
    # Reshape
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  # Num of rows, num of columns, num of features (just the close price)

    # Retrieve model's predicted price values
    predictions = model.predict(x_test)
    # Unscale predictions to compare to original closing prices (y_test)
    predictions = scaler.inverse_transform(predictions)  
 
    return predictions, x_test, y_test

def get_root_mean_square_error(predictions, y_test):
    rmse = np.sqrt( np.mean( predictions - y_test )**2 )
    print(rmse)
    return rmse

def visualize_predictions_with_actual(data, training_data_len, predictions):
    # Plot the data
    train = data[:training_data_len].copy()
    valid = data[training_data_len:].copy()
    valid['Predictions'] = predictions
    # Visualize the data
    plt.figure(figsize=(16,8))
    plt.title='Model'
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()
    print(valid)

    return valid

# Predict closing price for new day as 'YYYY-MM-DD'
def predict_closing_price(stock_symbol, model, date_to_predict, scaler):
    # Gather date to predict and 60 previous trading days (14 weeks just in case)
    yf.pdr_override()
    end_date = '2023-11-01'
    start_date = '2023-07-04'
    stock_quote = web.get_data_yahoo(f'{stock_symbol}', start=start_date, end=end_date)

    # Create a new dataframe
    new_df = stock_quote.filter(['Close'])

    # Retrieve only last 60 days of data and convert to array
    last_sixty_days = new_df[-60:].values

    # Scale the data to [0:1]
    last_sixty_days_scaled = scaler.transform(last_sixty_days)

    # Initialize list
    new_x_test = []
    new_x_test.append(last_sixty_days_scaled)

    # Convert to numpy array
    new_x_test = np.array(new_x_test)

    # Reshape
    new_x_test = np.reshape(new_x_test, (new_x_test.shape[0], new_x_test.shape[1], 1))
    # Calculate predictions
    pred_price = model.predict(new_x_test)

    # Rescale predictions
    pred_price = scaler.inverse_transform(pred_price)

    print('Predicted price on next day: ', pred_price)

    # Evaluate efficacy of model
    # Retrieve actual future quote data
    actual_start_date = '2023-11-01'
    actual_apple_quote=web.get_data_yahoo('MSFT', start='2023-11-01', end='2023-11-05')
    print(actual_apple_quote['Close'])

## LSTM Coordinator
def run_lstm(stock_symbol):
    df = gather_stock_data_df(stock_symbol)
    visualize_dataframe(df)
    data, dataset, scaled_closing_prices, training_data_len, scaler = scale_closing_prices(df)
    x_train, y_train = create_training_data_set(scaled_closing_prices, training_data_len)
    model = build_lstm_model(x_train, y_train)
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    predictions, x_test, y_test = test_data_set(dataset=dataset, scaled_closing_prices=scaled_closing_prices, training_data_len=training_data_len, model=model, scaler=scaler)
    rmse = get_root_mean_square_error(predictions, y_test)
    actual_and_predicted_closing_prices = visualize_predictions_with_actual(data, training_data_len, predictions)
    predict_closing_price(stock_symbol=stock_symbol, model=model, scaler=scaler, date_to_predict='2023-11-03')

def main():
    start = time.time()
    run_lstm('MSFT')
    end = time.time()

    print('Total time was ', end-start, ' seconds!')

if __name__=="__main__": 
    main() 