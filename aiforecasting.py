import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from prophet import Prophet
import matplotlib.pyplot as plt
import streamlit as st
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import re

# Step 1: Data Retrieval (Using SQL)
db_connection_str = 'postgresql+psycopg2://ts_user:secure_password@localhost:5432/time_series_db'
engine = create_engine(db_connection_str)

query = """
SELECT air_period AS ds, air_passengers AS y
FROM test.airline_passengers
"""
df = pd.read_sql(query, engine)
df['ds'] = pd.to_datetime(df['ds'])

# Step 2: Prepare Data for Forecasting
# Prophet Model
def forecast_with_prophet(df, periods=12):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future)
    return model, forecast

# Improved LSTM Model
class ImprovedLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, num_layers=2, dropout=0.2):
        super(ImprovedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def prepare_lstm_data(df, look_back=12):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df['y'].values.reshape(-1, 1))
    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:i + look_back])
        y.append(scaled_data[i + look_back])
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y, scaler

def forecast_with_lstm(df, periods=12, look_back=12):
    X, y, scaler = prepare_lstm_data(df, look_back)
    
    # Train-test split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Initialize and train the improved LSTM model
    model = ImprovedLSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # Lower learning rate
    
    for epoch in range(200):  # Increase epochs
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
    
    # Forecast future values
    model.eval()
    last_sequence = X[-1:].clone()
    forecast = []
    
    for _ in range(periods):
        with torch.no_grad():
            pred = model(last_sequence)
            forecast.append(pred.item())
            last_sequence = torch.cat((last_sequence[:, 1:, :], pred.unsqueeze(1)), dim=1)
    
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    return forecast

# Step 3: LLM Integration
# Load a pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# Function to generate insights using the LLM
def generate_insights_with_llm(df):
    # Convert the time series data into a text description
    trend = "increasing" if df['y'].iloc[-1] > df['y'].iloc[0] else "decreasing"
    mean_passengers = df['y'].mean()
    max_passengers = df['y'].max()
    min_passengers = df['y'].min()
    prompt = (f"The dataset spans from {df['ds'].iloc[0].strftime('%Y-%m')} to {df['ds'].iloc[-1].strftime('%Y-%m')}. "
              f"The number of passengers has an overall {trend} trend, with an average of {mean_passengers:.0f}, "
              f"a maximum of {max_passengers:.0f}, and a minimum of {min_passengers:.0f}. "
              f"Describe the trends and seasonality in this airline passenger data.")
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=600, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    insight = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return insight

# Function to forecast using the LLM (simplified approach)
def forecast_with_llm(df, periods=12, sequence_length=12):
    # Convert the last sequence of passenger numbers into a text prompt
    last_sequence = df['y'].tail(sequence_length).values
    sequence_str = " ".join(map(str, last_sequence))
    prompt = f"Given the sequence of passenger numbers: {sequence_str}, predict the next {periods} numbers."
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=512 + periods * 10, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract numbers from the generated text
    numbers = re.findall(r'\d+', generated_text)
    forecast = [float(num) for num in numbers if num.isdigit()]
    
    # Ensure we have the correct number of forecasted values
    forecast = forecast[-periods:] if len(forecast) >= periods else forecast + [forecast[-1]] * (periods - len(forecast))
    return np.array(forecast)

# Step 4: Build a Production Application with Streamlit
st.title("AI Time Series Forecasting Agent")

# Display the data
st.subheader("Historical Data")
st.line_chart(df.set_index('ds')['y'])

# Generate and display LLM insights
st.subheader("LLM Insights")
llm_insights = generate_insights_with_llm(df)
st.write(llm_insights)

# Forecast with Prophet
st.subheader("Prophet Forecast")
prophet_model, prophet_forecast = forecast_with_prophet(df)
fig1 = prophet_model.plot(prophet_forecast)
st.pyplot(fig1)

# Forecast with Improved LSTM
st.subheader("Improved LSTM Forecast")
lstm_forecast = forecast_with_lstm(df)
future_dates = pd.date_range(start=df['ds'].iloc[-1], periods=13, freq='M')[1:]
lstm_forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': lstm_forecast.flatten()})
st.line_chart(lstm_forecast_df.set_index('ds')['yhat'])

# Forecast with LLM
st.subheader("LLM Forecast")
llm_forecast = forecast_with_llm(df)
llm_forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': llm_forecast.flatten()})
st.line_chart(llm_forecast_df.set_index('ds')['yhat'])

if __name__ == "__main__":
    print("Run this app using: streamlit run this_script.py")
