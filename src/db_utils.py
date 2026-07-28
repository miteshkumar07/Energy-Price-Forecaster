import os
import urllib.parse
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load variables from the .env file automatically
load_dotenv() 

def get_db_engine():
    raw_pass = os.environ.get("DB_PASSWORD")
    if not raw_pass:
        raise ValueError("DB_PASSWORD environment variable is not set!")
    
    db_pass = urllib.parse.quote_plus(raw_pass)
    db_host = os.environ.get("DB_HOST")
    
    engine = create_engine(f"postgresql://postgres:{db_pass}@{db_host}:5432/postgres")
    return engine

def load_table(table_name):
    print(f" Fetching {table_name} from Google Cloud PostgreSQL...")
    engine = get_db_engine()
    return pd.read_sql(f"SELECT * FROM {table_name}", engine)