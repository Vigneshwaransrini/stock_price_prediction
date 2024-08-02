# import numpy as np
# import pandas as pd
# import yfinance as yf
# from keras.models import load_model
# import streamlit as st
# import matplotlib.pyplot as plt

# model = load_model(r'C:\Users\harih\Downloads\Stock_Market_Prediction_ML\Stock Predictions Model.keras')

# st.header('Stock Market Predictor')

# stock =st.text_input('Enter Stock Symnbol', 'GOOG')
# start = '2012-01-01'
# end = '2022-12-31'

# data = yf.download(stock, start ,end)

# st.subheader('Stock Data')
# st.write(data)

# data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
# data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0,1))

# pas_100_days = data_train.tail(100)
# data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
# data_test_scale = scaler.fit_transform(data_test)

# st.subheader('Price vs MA50')
# ma_50_days = data.Close.rolling(50).mean()
# fig1 = plt.figure(figsize=(8,6))
# plt.plot(ma_50_days, 'r')
# plt.plot(data.Close, 'g')
# plt.show()
# st.pyplot(fig1)

# st.subheader('Price vs MA50 vs MA100')
# ma_100_days = data.Close.rolling(100).mean()
# fig2 = plt.figure(figsize=(8,6))
# plt.plot(ma_50_days, 'r')
# plt.plot(ma_100_days, 'b')
# plt.plot(data.Close, 'g')
# plt.show()
# st.pyplot(fig2)

# st.subheader('Price vs MA100 vs MA200')
# ma_200_days = data.Close.rolling(200).mean()
# fig3 = plt.figure(figsize=(8,6))
# plt.plot(ma_100_days, 'r')
# plt.plot(ma_200_days, 'b')
# plt.plot(data.Close, 'g')
# plt.show()
# st.pyplot(fig3)

# x = []
# y = []

# for i in range(100, data_test_scale.shape[0]):
#     x.append(data_test_scale[i-100:i])
#     y.append(data_test_scale[i,0])

# x,y = np.array(x), np.array(y)

# predict = model.predict(x)

# scale = 1/scaler.scale_

# predict = predict * scale
# y = y * scale

# st.subheader('Original Price vs Predicted Price')
# fig4 = plt.figure(figsize=(8,6))
# plt.plot(predict, 'r', label='Original Price')
# plt.plot(y, 'g', label = 'Predicted Price')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.show()
# st.pyplot(fig4)

#to run streamlit run app.py
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime

model = load_model(r'C:\Users\DELL\Downloads\Stock-Price-Prediction-master\Stock-Price-Prediction-master\Stock Predictions Model.keras')

st.header('Stock Market Predictor')

stock = st.text_input('Enter Stock Symbol', 'GOOG', key="stock_symbol")
start = '2015-02-01'
end = datetime.today().strftime('%Y-%m-%d')

data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1, ax1 = plt.subplots(figsize=(12,6))
ax1.plot(ma_50_days, 'r', label = 'MA50')
ax1.plot(data.Close, 'g', label = 'Price')
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2, ax2 = plt.subplots(figsize=(12,6))
ax2.plot(ma_50_days, 'r')
ax2.plot(ma_100_days, 'b')
ax2.plot(data.Close, 'g')
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3, ax3 = plt.subplots(figsize=(12,6))
ax3.plot(ma_100_days, 'r')
ax3.plot(ma_200_days, 'b')
ax3.plot(data.Close, 'g')
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y * scale

st.subheader('Original Price vs Predicted Price')
fig4, ax4 = plt.subplots(figsize=(12,6))
ax4.plot(predict, 'r', label='Original Price')
ax4.plot(y, 'g', label='Predicted Price')
ax4.set_xlabel('Time')
ax4.set_ylabel('Price')
st.pyplot(fig4)

# Prediction for tomorrow's stock movement
last_100_days = data.Close.tail(100).values.reshape(-1, 1)
last_100_days_scaled = scaler.transform(last_100_days)

input_data = last_100_days_scaled.reshape(1, -1, 1)
predicted_price = model.predict(input_data)

if predicted_price[-1, 0] > last_100_days_scaled[-1, -1]:
    st.write("Prediction: Tomorrow's stock price is expected to rise.")
else:
    st.write("Prediction: Tomorrow's stock price is expected to fall.")

st.title("Next 5 Years Returns")

# Input for the stock symbol
symbol=stock

# Check for changes in the input field
if st.session_state.stock_symbol is not None:
    try:
        # Download historical data
        stock_data = yf.download(symbol, start="2023-01-01", end="2028-01-01")

        # Calculate returns over the next 5 years
        returns_next_5_years = (stock_data['Adj Close'].iloc[-1] / stock_data['Adj Close'].iloc[0] - 1) * 100

        # Display the returns
        st.write(f"{symbol} returns over the next 5 years: {returns_next_5_years:.2f}%")
        st.write("Recomment to Buy")
        if returns_next_5_years >= 0:
            st.markdown('<div style="padding: 10px; color: white; background-color: green; text-align: center;">Yes</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="padding: 10px; color: white; background-color: red; text-align: center;">No</div>', unsafe_allow_html=True)

    except Exception as e:
        st.write(f"Error: {str(e)}")


