import holidays
import pandas as pd
import numpy as np

def extract_features(df):
    df = df.copy()
    df = df.sort_values('datetime_berlin').reset_index(drop=True)
    df['Day'] = df['datetime_berlin'].dt.day
    df['Hour'] = df['datetime_berlin'].dt.hour
    df['Weekday'] = df['datetime_berlin'].dt.weekday

    # Cyclical Time Transformations (maps 23:00 close to 00:00)
    df['hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['weekday_sin'] = np.sin(2 * np.pi * df['Weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['Weekday'] / 7)

    # German public holidays
    germany_holidays = holidays.Germany()
    df['Is_Weekend'] = (df['Weekday'] >= 5).astype(int)
    df['Is_Holiday'] = df['datetime_berlin'].dt.date.apply(lambda d: 1 if d in germany_holidays else 0)

    # Lag and rolling features
    target_col = 'Price (€/MWh)'

    # Lag features: Looking back at specific historical hours
    df['price_lag_24'] = df[target_col].shift(24)     # Same hour yesterday
    df['price_lag_48'] = df[target_col].shift(48)     # Same hour 2 days ago
    df['price_lag_168'] = df[target_col].shift(168)   # Same hour last week (7 days)
    
    # Rolling features: Trailing moving averages (using shift to prevent data leakage)
    df['price_rolling_mean_24'] = df[target_col].shift(24).rolling(window=24).mean()
    df['price_rolling_max_24'] = df[target_col].shift(24).rolling(window=24).max()
    df['price_rolling_min_24'] = df[target_col].shift(24).rolling(window=24).min()

    # Drop initial rows that contain NaNs created by shifts and rolling windows
    # We explicitly exclude the target column so tomorrow's forecast row isn't dropped
    feature_cols = [col for col in df.columns if col != 'Price (€/MWh)']
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    return df