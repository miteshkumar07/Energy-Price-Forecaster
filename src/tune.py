import optuna
import optuna_integration.lightgbm as optuna_lgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from db_utils import load_table
from features import extract_features

# Keep terminal output clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    # 1. Fetch & Prepare Clean Data
    raw_df = load_table('historical_energy_data') 
    feat_df = extract_features(raw_df).sort_values('datetime_berlin').reset_index(drop=True)
    
    feature_cols = [
       'north_wind_wind_speed_100m', 'north_wind_wind_direction_100m', 'north_wind_wind_speed_10m',
       'south_solar_shortwave_radiation', 'south_solar_direct_normal_irradiance', 'south_solar_diffuse_radiation',
       'south_solar_cloud_cover', 'south_solar_cloud_cover_low',
       'ind_demand_apparent_temperature', 'ind_demand_temperature_2m', 'ind_demand_relative_humidity_2m',
       'HDD', 'CDD', 'Natural_Gas_Price', 'Carbon_Proxy_Price', 'Oil_Proxy_Price',
       'Day', 'Hour', 'Weekday', 'Is_Weekend', 'Is_Holiday', 
       'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos',
       'price_lag_24', 'price_lag_48', 'price_lag_168',
       'price_rolling_mean_24', 'price_rolling_max_24', 'price_rolling_min_24'
    ]
    target_col = 'Price (€/MWh)'
    
    cleaned_df = feat_df.dropna(subset=[target_col]).reset_index(drop=True)
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_df[feature_cols], cleaned_df[target_col], 
        test_size=0.25, shuffle=False
    )

    # 2. Hyperparameter Search Space
    params = {
        'objective': 'quantile',
        'alpha': 0.50, # Optimizing for the Median (p50) prediction
        'eval_metric': 'mae',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
        
        # 1. Slow down learning, increase trees
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.03, log=True),
        
        # 2. Leaf-wise complexity control
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'max_depth': -1,
        
        # 3. Outlier resistance
        'min_child_samples': trial.suggest_categorical('min_child_samples', [20, 30, 50, 100]),
        
        # 4. Feature and Row subsampling
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
    }

    # 3. Train Model with Early Stopping
    model = lgb.LGBMRegressor(**params, n_estimators=5000)
    
    model.fit(
        X_train, y_train,
        eval_X=X_test, 
        eval_y=y_test,
        callbacks=[
            lgb.early_stopping(100, verbose=False), 
            optuna_lgb.LightGBMPruningCallback(trial, "quantile") # Matches LightGBM's internal naming
        ]
    )
    # 4. Return Validation Metric
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)

if __name__ == "__main__":
    print("🎯 Starting Optuna Hyperparameter Optimization (50 Trials)...")
    
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    study.optimize(objective, n_trials=50, timeout=1800)

    print("\n" + "="*50)
    print("🏆 OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"Best Validation MAE: {study.best_value:.2f} €/MWh")
    print("\nReplace the parameters in src/train.py and src/backtest.py with these:\n")
    
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"    '{key}': {value:.4f},")
        else:
            print(f"    '{key}': {value},")