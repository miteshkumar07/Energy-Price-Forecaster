import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from features import extract_features
import os
import urllib.parse
from sqlalchemy import create_engine
from db_utils import load_table

def run_walk_forward_backtest(days_to_test=90):
    print(" Loading and extracting features...")
    raw_df = load_table('historical_energy_data')
    df = extract_features(raw_df)
    
    df = df.sort_values('datetime_berlin').dropna().reset_index(drop=True)
    df['date_only'] = df['datetime_berlin'].dt.date
    
    feature_cols = [
       'north_wind_wind_speed_100m', 'north_wind_wind_direction_100m', 'north_wind_wind_speed_10m',
       'south_solar_shortwave_radiation', 'south_solar_direct_normal_irradiance', 'south_solar_diffuse_radiation',
       'south_solar_cloud_cover', 'south_solar_cloud_cover_low',
       'ind_demand_apparent_temperature', 'ind_demand_temperature_2m', 'ind_demand_relative_humidity_2m',
       'HDD', 'CDD',
       'Natural_Gas_Price', 'Carbon_Proxy_Price', 'Oil_Proxy_Price',
       'Day', 'Hour', 'Weekday', 'Is_Weekend', 'Is_Holiday', 
       'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 
       'price_lag_24', 'price_lag_48', 'price_lag_168',
       'price_rolling_mean_24', 'price_rolling_max_24', 'price_rolling_min_24'
    ]
    target_col = 'Price (€/MWh)'

    all_dates = sorted(df['date_only'].unique())
    test_dates = all_dates[-days_to_test:]
    
    print(f" Starting Walk-Forward Backtest for the last {days_to_test} days...")

    model_10 = lgb.LGBMRegressor(objective='quantile', alpha=0.10, n_estimators=1000, learning_rate=0.02, num_leaves=32, min_child_samples=100,
        max_depth=6, subsample=0.7, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
    model_50 = lgb.LGBMRegressor(objective='quantile', alpha=0.50, n_estimators=1000, learning_rate=0.02, num_leaves=32, min_child_samples=100,
        max_depth=6, subsample=0.7, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
    model_90 = lgb.LGBMRegressor(objective='quantile', alpha=0.90, n_estimators=1000, learning_rate=0.02, num_leaves=32, min_child_samples=100,
        max_depth=6, subsample=0.7, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)

    all_predictions_10, all_predictions_50, all_predictions_90, all_actuals, plot_timestamps = [], [], [], [], []

    for current_date in test_dates:
        train_mask = df['date_only'] < current_date
        test_mask = df['date_only'] == current_date
        
        X_train, y_train = df[train_mask][feature_cols], df[train_mask][target_col]
        X_test, y_test = df[test_mask][feature_cols], df[test_mask][target_col]
        
        if X_test.empty:
            continue
            
        model_10.fit(X_train, y_train, eval_X=X_test, eval_y=y_test, callbacks=[lgb.early_stopping(50, verbose=False)])
        model_50.fit(X_train, y_train, eval_X=X_test, eval_y=y_test, callbacks=[lgb.early_stopping(50, verbose=False)])
        model_90.fit(X_train, y_train, eval_X=X_test, eval_y=y_test, callbacks=[lgb.early_stopping(50, verbose=False)])

        all_predictions_10.extend(model_10.predict(X_test))
        all_predictions_50.extend(model_50.predict(X_test))
        all_predictions_90.extend(model_90.predict(X_test))
        all_actuals.extend(y_test)
        plot_timestamps.extend(df[test_mask]['datetime_berlin'])
        
        print(f"Evaluated {current_date} | {len(y_test)} hours predicted")

    results_df = pd.DataFrame({
        'Datetime': plot_timestamps, 'Actual_Price': all_actuals,
        'Predicted_Price_p10': all_predictions_10, 'Predicted_Price_p50': all_predictions_50, 'Predicted_Price_p90': all_predictions_90
    })
    
    results_df['Error (€)'] = results_df['Predicted_Price_p50'] - results_df['Actual_Price']
    results_df['Abs_Error (€)'] = results_df['Error (€)'].abs()
    results_df = results_df.round(2)
    
    print(f"\n FINAL 90-DAY WALK-FORWARD MAE: {results_df['Abs_Error (€)'].mean():.2f} €/MWh\n")

    print(" Saving backtest results to PostgreSQL database...")
    DB_USER = "postgres"
    DB_PASS = urllib.parse.quote_plus(os.environ.get("DB_PASSWORD", "fallback_local_pass"))
    DB_HOST = os.environ.get("DB_HOST", "localhost")
    DB_NAME = "postgres"
    try:
        engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:5432/{DB_NAME}")
        results_df.to_sql('backtest_results', engine, if_exists='replace', index=False)
        print(" Successfully saved to 'backtest_results' table in PostgreSQL!")
    except Exception as e:
        print(f" Failed to save to database: {e}")

    os.makedirs("visualizations", exist_ok=True)

    plt.figure(figsize=(16, 6))
    plot_days = 14 * 24 
    plt.fill_between(plot_timestamps[-plot_days:], all_predictions_10[-plot_days:], all_predictions_90[-plot_days:], color='darkorange', alpha=0.2, label='80% Confidence Interval')
    plt.plot(plot_timestamps[-plot_days:], all_actuals[-plot_days:], label='Actual Price (€/MWh)', color='black', linewidth=1.5)
    plt.plot(plot_timestamps[-plot_days:], all_predictions_50[-plot_days:], label='Median Prediction (p50)', color='darkorange', linewidth=2.0)
    
    plt.title('LightGBM Probabilistic Price Forecast vs Actuals (Last 14 Days)', fontsize=14, pad=15)
    plt.xlabel('Date / Time', fontsize=12)
    plt.ylabel('Price (€/MWh)', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.savefig("visualizations/backtest_results.png", bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    run_walk_forward_backtest(days_to_test=90)