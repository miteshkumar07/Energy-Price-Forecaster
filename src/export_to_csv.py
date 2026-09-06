import os
import sys
import urllib.parse
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine


def export_database_tables_to_csv(output_dir="tableau_exports"):
    """
    Connects to PostgreSQL database, extracts the 2-year historical and 90-day rolling
    energy dataset tables, and saves them locally as CSV files for Tableau.
    """
    load_dotenv()  # Load environment variables from .env file if present
    # 1. Database Connection Configuration
    DB_USER = "postgres"
    DB_PASS = urllib.parse.quote_plus(os.environ.get("DB_PASSWORD", "fallback_local_pass"))
    DB_HOST = os.environ.get("DB_HOST", "localhost")
    DB_NAME = "postgres"
    DB_PORT = "5432"

    database_url = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    print(f"🔌 Connecting to PostgreSQL Database at {DB_HOST}...")

    try:
        engine = create_engine(database_url)
    except Exception as e:
        print(f" Failed to create database engine: {e}")
        sys.exit(1)

    # 2. Ensure Output Directory Exists
    os.makedirs(output_dir, exist_ok=True)

    # 3. Tables to Extract
    tables_to_export = [
        "historical_energy_data",
        "rolling_90d_energy_data"
    ]

    # 4. Extract and Save CSVs
    for table_name in tables_to_export:
        print(f"\n Fetching table '{table_name}' from PostgreSQL...")
        
        try:
            # Query the entire table using pandas
            df = pd.read_sql_table(table_name, engine)
            
            if df.empty:
                print(f" Warning: Table '{table_name}' is empty.")
                continue

            # Ensure datetime columns maintain clean string format for Tableau parser
            if 'datetime_berlin' in df.columns:
                df['datetime_berlin'] = pd.to_datetime(df['datetime_berlin']).dt.strftime('%Y-%m-%d %H:%M:%S')
            if 'datetime_utc' in df.columns:
                df['datetime_utc'] = pd.to_datetime(df['datetime_utc']).dt.strftime('%Y-%m-%d %H:%M:%S')

            # Define output path
            output_path = os.path.join(output_dir, f"{table_name}.csv")
            
            # Export to CSV
            df.to_csv(output_path, index=False)
            print(f" Successfully saved {len(df):,} rows to '{output_path}'")

        except Exception as e:
            print(f" Error exporting table '{table_name}': {e}")

    print(f"\n Export complete! All CSV files are available in the '{output_dir}/' folder.")


if __name__ == "__main__":
    export_database_tables_to_csv()