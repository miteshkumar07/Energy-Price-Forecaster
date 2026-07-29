import os
import sys
import urllib.parse
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import pandera.pandas as pa
from pandera.pandas import Column, Check, DataFrameSchema
from sqlalchemy import create_engine
import time
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

def get_retry_session():
    """Configures a requests session that automatically retries failed connections."""
    session = requests.Session()
    # Retries 5 times, waiting 1s, 2s, 4s, 8s, 16s between attempts
    retry = Retry(
        total=5,
        read=5,
        connect=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def fetch_dwd_spatial_weather(start_date, end_date):
    """
    Fetches official Deutscher Wetterdienst (DWD ICON) atmospheric variables 
    across 3 strategic German spatial hubs:
    1. North Wind Hub (Lower Saxony/Coast - Lat: 53.5, Lon: 8.5)
    2. South Solar Hub (Bavaria - Lat: 49.0, Lon: 11.5)
    3. Industrial Demand Hub (NRW - Lat: 51.2, Lon: 6.8)
    """
    print("Fetching DWD (Deutscher Wetterdienst ICON) multi-spatial weather data...")
    
    hubs = {
        "north_wind": {"lat": 53.5, "lon": 8.5},
        "south_solar": {"lat": 49.0, "lon": 11.5},
        "ind_demand": {"lat": 51.2, "lon": 6.8}
    }
    
    hourly_vars = [
        "temperature_2m", "apparent_temperature", "relative_humidity_2m", "surface_pressure",
        "wind_speed_10m", "wind_speed_100m", "wind_direction_10m", "wind_direction_100m",
        "shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation",
        "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "snow_depth"
    ]
    vars_str = ",".join(hourly_vars)
    now_berlin = pd.Timestamp.now(tz='Europe/Berlin')
    today_str = now_berlin.strftime('%Y-%m-%d')
    tomorrow_str = (now_berlin + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    combined_hubs_df = None
    session = get_retry_session()
    
    for hub_name, coords in hubs.items():
        # 1. Fetch Historical DWD Runs up to today
        hist_url = (
            f"https://historical-forecast-api.open-meteo.com/v1/forecast?"
            f"latitude={coords['lat']}&longitude={coords['lon']}&"
            f"start_date={start_date}&end_date={today_str}&"
            f"hourly={vars_str}&timezone=Europe/Berlin"
        )
        r_hist = session.get(hist_url, timeout=15)
        r_hist.raise_for_status()
        data_hist = r_hist.json()['hourly']
        
        df_hist = pd.DataFrame(data_hist)
        df_hist['datetime_berlin'] = pd.to_datetime(df_hist['time'])
        df_hist = df_hist.drop(columns=['time'])
        
        # 2. Fetch Live Morning DWD ICON Forecast for Tomorrow
        live_url = (
            f"https://api.open-meteo.com/v1/dwd-icon?"
            f"latitude={coords['lat']}&longitude={coords['lon']}&"
            f"start_date={tomorrow_str}&end_date={tomorrow_str}&"
            f"hourly={vars_str}&timezone=Europe/Berlin"
        )
        r_live = session.get(live_url, timeout=15)
        r_live.raise_for_status()
        data_live = r_live.json()['hourly']
        
        df_live = pd.DataFrame(data_live)
        df_live['datetime_berlin'] = pd.to_datetime(df_live['time'])
        df_live = df_live.drop(columns=['time'])
        
        # 3. Concatenate and Prefix Columns
        hub_df = pd.concat([df_hist, df_live], ignore_index=True)
        hub_df['datetime_berlin'] = hub_df['datetime_berlin'].dt.tz_localize(
            'Europe/Berlin', nonexistent='shift_forward', ambiguous='NaT'
        )
        hub_df = hub_df.dropna(subset=['datetime_berlin'])
        
        # Prefix feature columns with region name
        rename_dict = {col: f"{hub_name}_{col}" for col in hub_df.columns if col != 'datetime_berlin'}
        hub_df = hub_df.rename(columns=rename_dict)
        
        if combined_hubs_df is None:
            combined_hubs_df = hub_df
        else:
            combined_hubs_df = pd.merge(combined_hubs_df, hub_df, on='datetime_berlin', how='outer')

    combined_hubs_df['datetime_utc'] = combined_hubs_df['datetime_berlin'].dt.tz_convert('UTC')
    
    # 4. Feature Engineering: Degree Days & Macro Indices
    temp_col = 'ind_demand_temperature_2m'
    combined_hubs_df['HDD'] = (18.0 - combined_hubs_df[temp_col]).clip(lower=0.0)
    combined_hubs_df['CDD'] = (combined_hubs_df[temp_col] - 18.0).clip(lower=0.0)
    
    return combined_hubs_df


def fetch_financial_data(start_date, end_date):
    print("Fetching financial commodity data (Yahoo Finance)...")
    tickers = {
        "TTF=F": "Natural_Gas_Price",
        "KRBN": "Carbon_Proxy_Price",
        "BZ=F": "Oil_Proxy_Price"
    }
    
    fin_df = pd.DataFrame()
    for ticker, col_name in tickers.items():
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        if not hist.empty:
            hist = hist[['Close']].rename(columns={'Close': col_name})
            if hist.index.tz is None:
                hist.index = hist.index.tz_localize('UTC').tz_convert('Europe/Berlin').normalize()
            else:
                hist.index = hist.index.tz_convert('Europe/Berlin').normalize()
            
            if fin_df.empty:
                fin_df = hist
            else:
                fin_df = fin_df.join(hist, how='outer')
                
    fin_df = fin_df.reset_index().rename(columns={'Date': 'date_only'})
    fin_df['date_only'] = fin_df['date_only'] + pd.Timedelta(days=2) 
    return fin_df


def fetch_smard_actual_price(start_date, end_date):
    """
    Fetches strictly the actual Day-Ahead spot price from SMARD (Filter ID 4169).
    """
    print("Fetching SMARD Actual Day-Ahead Wholesale Prices...")
    cache_buster = int(time.time())
    INDEX_URL = "https://www.smard.de/app/chart_data/4169/DE/index_hour.json"
    
    session = get_retry_session()
    response = session.get(INDEX_URL, timeout=15)
    response.raise_for_status()
    timestamps = response.json()['timestamps'][-104:]
    series_data = []
    
    for week in timestamps:
        week_url = f"https://www.smard.de/app/chart_data/4169/DE/4169_DE_hour_{week}.json?n={cache_buster}"
        week_response = session.get(week_url, timeout=15)
        week_response.raise_for_status()
        series_data.extend(week_response.json()['series'])
        
    smard_df = pd.DataFrame(series_data, columns=['Timestamp_ms', 'Price (€/MWh)'])
    smard_df['datetime_utc'] = pd.to_datetime(smard_df['Timestamp_ms'], unit='ms').dt.tz_localize('UTC')
    smard_df['datetime_berlin'] = smard_df['datetime_utc'].dt.tz_convert('Europe/Berlin')
    
    return smard_df


def run_etl():
    print(" Starting ETL Pipeline...")
    
    # 1. Date Setup
    now_berlin = pd.Timestamp.now(tz='Europe/Berlin')
    end_dt = now_berlin + pd.Timedelta(days=1)
    start_dt = end_dt - pd.Timedelta(weeks=104)
    start_str = start_dt.strftime('%Y-%m-%d')
    end_str = end_dt.strftime('%Y-%m-%d')
    
    # 2. Extract Data from all sources
    smard_df = fetch_smard_actual_price(start_str, end_str)
    weather_df = fetch_dwd_spatial_weather(start_str, end_str)
    fin_df = fetch_financial_data(start_str, end_str)

    # 3. Merging data
    print("Merging DWD Atmospheric, Commodity, and Market Price streams...")
    master_df = pd.merge(smard_df, weather_df, on='datetime_utc', how='outer')
    
    # Fix datetime column after outer join
    master_df['datetime_berlin'] = master_df['datetime_utc'].dt.tz_convert('Europe/Berlin')
    master_df['date_only'] = master_df['datetime_berlin'].dt.normalize()
    
    master_df = pd.merge(master_df, fin_df, on='date_only', how='left')
    master_df = master_df.drop(columns=['datetime_berlin_x', 'datetime_berlin_y', 'date_only'], errors='ignore')
    
    # Clean duplicates & sort chronologically
    master_df = master_df.drop_duplicates(subset=['datetime_utc'], keep='last')
    master_df = master_df.sort_values('datetime_berlin').reset_index(drop=True)
    
    # 4. Forward-Fill Daily Financial Commodities into Tomorrow
    fin_cols = ['Natural_Gas_Price', 'Carbon_Proxy_Price', 'Oil_Proxy_Price']
    master_df[fin_cols] = master_df[fin_cols].ffill()
    
    # 5. Clean Missing Values (Ignore target price for tomorrow)
    cols_to_check = [col for col in master_df.columns if col not in ['Price (€/MWh)', 'Timestamp_ms']]
    master_df = master_df.dropna(subset=cols_to_check).reset_index(drop=True)

    # Force standard nanosecond datetimes
    master_df['datetime_utc'] = master_df['datetime_utc'].astype("datetime64[ns, UTC]")
    master_df['datetime_berlin'] = master_df['datetime_berlin'].astype("datetime64[ns, Europe/Berlin]")

    # 6. Pandera Schema Validation
    schema = DataFrameSchema({
        "Price (€/MWh)": Column(float, nullable=True, checks=[Check.in_range(-1000.0, 4000.0)]),
        "datetime_utc": Column("datetime64[ns, UTC]", nullable=False),
        "datetime_berlin": Column("datetime64[ns, Europe/Berlin]", nullable=False),
        
        # North Wind Hub
        "north_wind_wind_speed_100m": Column(float, nullable=False, checks=[Check.ge(0.0)]),
        "north_wind_wind_direction_100m": Column(float, nullable=False, checks=[Check.in_range(0.0, 360.0)]),
        
        # South Solar Hub
        "south_solar_shortwave_radiation": Column(float, nullable=False, checks=[Check.ge(0.0)]),
        "south_solar_direct_normal_irradiance": Column(float, nullable=False, checks=[Check.ge(0.0)]),
        "south_solar_cloud_cover": Column(float, nullable=False, checks=[Check.in_range(0.0, 100.0)]),
        
        # Industrial Demand & Macro
        "HDD": Column(float, nullable=False, checks=[Check.ge(0.0)]),
        "CDD": Column(float, nullable=False, checks=[Check.ge(0.0)]),
        "Natural_Gas_Price": Column(float, nullable=True),
        "Carbon_Proxy_Price": Column(float, nullable=True),
    })
    
    try:
        schema.validate(master_df)
        print(" Data validation successful!")
    except pa.errors.SchemaError as e:
        print(f" Data validation failed. Error in column: {e.schema.name}")
        sys.exit(1)

    # 7. Database Upload
    DB_USER = "postgres"
    DB_PASS = urllib.parse.quote_plus(os.environ.get("DB_PASSWORD", "fallback_local_pass"))
    DB_HOST = os.environ.get("DB_HOST", "localhost")
    DB_NAME = "postgres"

    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:5432/{DB_NAME}")

    rolling_90d = master_df.tail(90 * 24).reset_index(drop=True)

    print(" Pushing Data to Cloud Database...")
    master_df.to_sql('historical_energy_data', engine, if_exists='replace', index=False)
    rolling_90d.to_sql('rolling_90d_energy_data', engine, if_exists='replace', index=False)
    print(" ETL Execution Finished Successfully!")
    
    return master_df


if __name__ == "__main__":
    run_etl()