import pandas as pd
from sqlalchemy import create_engine

# Connect to the database
db_connection_str = "postgresql+psycopg2://ts_user:secure_password@localhost:5432/time_series_db"
engine = create_engine(db_connection_str)

# Query the data
query = """
SELECT air_period AS ds, air_passengers AS y
FROM test.airline_passengers
"""
df = pd.read_sql(query, engine)
#print(df.head())
print(df)
