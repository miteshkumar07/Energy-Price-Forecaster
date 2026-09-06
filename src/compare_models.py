import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from db_utils import load_table
from features import extract_features

def run_model_shootout():
    print(" Loading and preparing data from PostgreSQL...")
    raw_df = load_table('historical_energy_data')
    feat_extracted_df = extract_features(raw_df)
    
    # Ensure strict chronological sort
    cleaned_feat_df = feat_extracted_df.sort_values('datetime_berlin').reset_index(drop=True)
    
    # Feature columns
    feature_cols = [
       'north_wind_wind_speed_100m', 'north_wind_wind_direction_100m', 'north_wind_wind_speed_10m',
       'south_solar_shortwave_radiation', 'south_solar_direct_normal_irradiance', 'south_solar_diffuse_radiation',
       'south_solar_cloud_cover', 'south_solar_cloud_cover_low',
       'ind_demand_apparent_temperature', 'ind_demand_temperature_2m', 'ind_demand_relative_humidity_2m',
       'HDD', 'CDD',
       'Natural_Gas_Price', 'Carbon_Proxy_Price', 'Oil_Proxy_Price',
       'Hour', 'Weekday', 'Is_Weekend', 'Is_Holiday', 
       'price_lag_24', 'price_lag_48', 'price_lag_168',
       'price_rolling_mean_24', 'price_rolling_max_24', 'price_rolling_min_24'
    ]
    target_col = 'Price (€/MWh)'

    # 75/25 Chronological Split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        cleaned_feat_df[feature_cols], 
        cleaned_feat_df[target_col], 
        test_size=0.25, 
        shuffle=False
    )
    
    val_size = 14 * 24
    X_train = X_train_full.iloc[:-val_size]
    y_train = y_train_full.iloc[:-val_size]
    X_val = X_train_full.iloc[-val_size:]
    y_val = y_train_full.iloc[-val_size:]
    
    print(f" Training rows: {len(X_train)} | Val rows: {len(X_val)} | Testing rows: {len(X_test)}\n")
    print("-" * 60)

    # Define the Contenders
    models = {
        "XGBoost": xgb.XGBRegressor(
            objective='reg:squarederror',   # Back to the winning config!
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=50,
            random_state=42,
            n_jobs=-1
        ),
        "LightGBM": lgb.LGBMRegressor(
            objective='regression',
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,               
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
    }

    results = []

    for name, model in models.items():
        print(f" Training {name}...")
        start_time = time.time()
        
        # XGBoost and LightGBM support early stopping with eval_set
        if name in ["XGBoost", "LightGBM"]:
            if name == "LightGBM":
                # LightGBM handles early stopping via callbacks in newer versions
                model.fit(X_train, y_train, eval_X=X_val, eval_y=y_val, callbacks=[lgb.early_stopping(50, verbose=False)])
            else:
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)
            
        train_time = time.time() - start_time
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            "Model": name,
            "MAE (€/MWh)": round(mae, 2),
            "R2 Score": round(r2, 4),
            "Train Time (s)": round(train_time, 2)
        })
        
        print(f" {name} Finished | MAE: {mae:.2f} | R²: {r2:.4f} | Time: {train_time:.2f}s\n")

    print("-" * 60)
    print("🏆 FINAL LEADERBOARD 🏆")
    print("-" * 60)
    
    results_df = pd.DataFrame(results).sort_values(by="MAE (€/MWh)", ascending=True).reset_index(drop=True)
    print(results_df.to_string(index=False))
    print("-" * 60)

if __name__ == "__main__":
    run_model_shootout()