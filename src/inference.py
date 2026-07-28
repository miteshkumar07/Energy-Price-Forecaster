import os
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from db_utils import get_db_engine, load_table
from features import extract_features

def run_inference():
    print(" Starting Daily Inference Pipeline...")

    print(" Fetching latest grid data...")
    raw_df = load_table('rolling_90d_energy_data')
    df = extract_features(raw_df).sort_values('datetime_berlin').reset_index(drop=True)
    
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
    
    recent_df = df.tail(48).copy()
    X_predict = recent_df[feature_cols]
    
    print("Loading Models...")
    model_10 = lgb.Booster(model_file="models/lightgbm_price_model_10.json")
    model_50 = lgb.Booster(model_file="models/lightgbm_price_model_50.json")
    model_90 = lgb.Booster(model_file="models/lightgbm_price_model_90.json")
    
    print("Generating Probabilistic Forecasts...")
    recent_df['Predicted_Price_p10'] = model_10.predict(X_predict)
    recent_df['Predicted_Price_p50'] = model_50.predict(X_predict)
    recent_df['Predicted_Price_p90'] = model_90.predict(X_predict)
    
    forecast_df = recent_df[['datetime_berlin', 'Price (€/MWh)', 'Predicted_Price_p10', 'Predicted_Price_p50', 'Predicted_Price_p90']]
    forecast_df = forecast_df.rename(columns={'Price (€/MWh)': 'Actual_Price'})
    
    print("Generating SHAP Explainability...")
    os.makedirs("visualizations", exist_ok=True)
    
    clean_feature_names = {
        'north_wind_wind_speed_100m': 'North Sea Wind Speed 100m (km/h)',
        'north_wind_wind_direction_100m': 'North Sea Wind Direction 100m',
        'north_wind_wind_speed_10m': 'North Sea Wind Speed 10m (km/h)',
        'south_solar_shortwave_radiation': 'Bavaria Solar Shortwave Radiation',
        'south_solar_direct_normal_irradiance': 'Bavaria Solar Direct Irradiance',
        'south_solar_diffuse_radiation': 'Bavaria Solar Diffuse Radiation',
        'south_solar_cloud_cover': 'Bavaria Total Cloud Cover (%)',
        'south_solar_cloud_cover_low': 'Bavaria Low Cloud Cover (%)',
        'ind_demand_apparent_temperature': 'NRW "Feels Like" Temp (°C)',
        'ind_demand_temperature_2m': 'NRW Actual Temp (°C)',
        'ind_demand_relative_humidity_2m': 'NRW Humidity (%)',
        'HDD': 'Heating Degree Days (Demand Proxy)',
        'CDD': 'Cooling Degree Days (Demand Proxy)',
        'Natural_Gas_Price': 'Natural Gas Price (€/MWh)',
        'Carbon_Proxy_Price': 'Carbon Emissions Price (€/t)',
        'Oil_Proxy_Price': 'Brent Crude Oil ($/bbl)',
        'Day': 'Day of Month', 'Hour': 'Hour of Day', 'Weekday': 'Day of Week',
        'Is_Weekend': 'Is Weekend', 'Is_Holiday': 'Is Public Holiday',
        'price_lag_24': 'Price Yesterday (€)', 'price_lag_48': 'Price 2 Days Ago (€)', 'price_lag_168': 'Price 1 Week Ago (€)',
        'price_rolling_mean_24': '24h Rolling Average Price (€)',
        'price_rolling_max_24': '24h Rolling Max Price (€)',
        'price_rolling_min_24': '24h Rolling Min Price (€)'
    }

    X_explain = X_predict.tail(24).copy()
    explainer = shap.TreeExplainer(model_50)
    shap_values = explainer(X_explain)
    
    shap_values.feature_names = [clean_feature_names.get(name, name) for name in shap_values.feature_names]
    X_explain_clean = X_explain.rename(columns=clean_feature_names)
    
    plt.figure(figsize=(12, 8))
    shap.plots.waterfall(shap_values[-1], show=False)
    plt.title("Real-Time Price Drivers (SHAP Waterfall)", fontsize=16, pad=20) 
    plt.savefig("visualizations/shap_waterfall.png", bbox_inches='tight', dpi=400)
    plt.close()
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_explain_clean, show=False) 
    plt.title("Macro Feature Impacts (Last 24 Hours)", fontsize=16, pad=20)
    plt.savefig("visualizations/shap_summary.png", bbox_inches='tight', dpi=400)
    plt.close()
    
    print("Uploading Forecast to Google Cloud...")
    engine = get_db_engine()
    forecast_df.to_sql('tomorrow_predictions', engine, if_exists='replace', index=False)
    print(" Inference Pipeline Complete! Dashboard is ready to serve.")

if __name__ == "__main__":
    run_inference()