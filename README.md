# 🧠 AI Time Series Forecasting Agent

This project is an interactive forecasting agent built with **Streamlit**, supporting **Prophet**, **LSTM**, and **LLM-based** time series forecasting models. It connects to a PostgreSQL database and visualizes historical data and forecasts.

---

## 📁 Project Structure

```
.
├── aiforecasting.py                       # Main Streamlit application
├── convertinternational-airline-passengerscsv.py  # CSV pre-processing
├── international-airline-passengers.csv   # Raw dataset
├── international-airline-passengers-modified.csv  # Cleaned dataset
├── testsetup.py                           # Database table creation helper
```

---

## ⚙️ 1. Environment Setup

### 🔹 Python Version
Python 3.10+

### 🔹 Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 🔹 Install Dependencies
```bash
pip install -r requirements.txt
```

If `requirements.txt` does not exist, manually install:
```bash
pip install pandas numpy sqlalchemy psycopg2-binary streamlit matplotlib scikit-learn torch prophet transformers
```

---

## 🗃️ 2. PostgreSQL Setup

### 🔹 Create Database and User

Launch psql and run:
```sql
CREATE DATABASE time_series_db;
CREATE USER ts_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE time_series_db TO ts_user;
```

### 🔹 Create Table and Schema
Run `testsetup.py` to initialize the schema:
```bash
python testsetup.py
```

> This will create a schema `test` and the table `test.airline_passengers`.

---

## 📥 3. Prepare Data

Convert the original CSV (semicolon-delimited) to a standard format:
```bash
python convertinternational-airline-passengerscsv.py
```

---

## 🧠 4. Running the App

Launch the Streamlit interface:
```bash
streamlit run aiforecasting.py
```

Open the browser at:
- Local: [http://localhost:8501](http://localhost:8501)
- Network: Shown in terminal output

---

## 🔍 Features

- 📈 **Prophet Forecast**: Seasonality-aware model for time series.
- 🔁 **LSTM Forecast**: Deep learning-based forecasting with PyTorch.
- 🧠 **LLM Forecast**: Predict future values using a GPT-2-based language model.
- 💡 **LLM Insights**: Descriptive text generation to explain trends.

---

## 🔐 Credentials

Edit this in `aiforecasting.py` if needed:
```python
db_connection_str = 'postgresql+psycopg2://ts_user:secure_password@localhost:5432/time_series_db'
```

---

## 🧪 Testing Tips

- Ensure PostgreSQL is running: `sudo systemctl status postgresql`
- Validate DB schema: `psql -U ts_user -d time_series_db -c '\dt test.*'`

---

## 📦 Future Enhancements

- Add CSV upload via Streamlit UI
- Deploy with Docker
- Add anomaly detection or ARIMA
- Add explanations with GPT-4 or Gemini

---

## 📜 License

MIT License – feel free to use and modify.
