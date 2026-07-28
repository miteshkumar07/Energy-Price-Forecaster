import pandas as pd
from cost_calculator import calculate_schedule_cost
from db_utils import load_table

def optimize_continuous(df, hours, price_col='Predicted_Industrial'):
    df_copy = df.copy()
    df_copy['Consecutive_Cost'] = df_copy[price_col].rolling(window=hours).sum()
    df_copy = df_copy.dropna(subset=['Consecutive_Cost'])
    
    best_end_idx = df_copy['Consecutive_Cost'].idxmin()
    best_start_idx = best_end_idx - hours + 1
    
    schedule = df.loc[best_start_idx:best_end_idx]
    total_mwh_cost = schedule[price_col].sum()
    return schedule, total_mwh_cost

def optimize_interruptible(df, hours, price_col='Predicted_Industrial'):
    cheapest_hours = df.sort_values(by=price_col, ascending=True).head(hours)
    schedule = cheapest_hours.sort_index()
    total_mwh_cost = schedule[price_col].sum()
    return schedule, total_mwh_cost

def calculate_baseline_cost(df, hours, start_hour=9, price_col='Predicted_Industrial'):
    try:
        start_idx = df[df['Datetime'].dt.hour == start_hour].index[0]
        end_idx = start_idx + hours - 1
        baseline_schedule = df.loc[start_idx:end_idx]
        return baseline_schedule[price_col].sum()
    except IndexError:
        return df.head(hours)[price_col].sum()


def run_optimizer():
    print(" Loading Live Machine Learning Forecasts from Database...")
    # 1. Fetch live predictions from Cloud
    results_df = load_table('tomorrow_predictions')

    if results_df.empty:
        print(" Error: 'tomorrow_predictions' table is empty in database. Run inference.py first.")
        return

    results_df['Datetime'] = pd.to_datetime(results_df['datetime_berlin'])
    
    # Take the most recent 24-hour forecast block
    forecast_df = results_df.sort_values('Datetime').tail(24).reset_index(drop=True)
    print(f" Loaded live 24-hour forecast from PostgreSQL ({forecast_df['Datetime'].iloc[0].strftime('%Y-%m-%d %H:%M')} to {forecast_df['Datetime'].iloc[-1].strftime('%H:%M')})")

    RUN_HOURS = 4
    POWER_CONSUMPTION_KW = 500  
    
    # Dynamically select target pricing column from PostgreSQL table
    if 'Predicted_Industrial' in forecast_df.columns:
        TARGET_COLUMN = 'Predicted_Industrial'
    elif 'Predicted_Price_p50' in forecast_df.columns:
        TARGET_COLUMN = 'Predicted_Price_p50'
    else:
        TARGET_COLUMN = forecast_df.columns[1]
    
    print(f"\n SCHEDULING HEAVY MACHINERY ({RUN_HOURS} Hours Required | {POWER_CONSUMPTION_KW} kW)")
    print("-" * 60)
    
    # Calculate baseline schedule (08:00 start)
    baseline_mwh_sum = calculate_baseline_cost(forecast_df, RUN_HOURS, start_hour=8, price_col=TARGET_COLUMN)
    baseline_total_cost = (POWER_CONSUMPTION_KW / 1000.0) * baseline_mwh_sum
    
    # Calculate continuous operation
    cont_sched, cont_mwh_sum = optimize_continuous(forecast_df, RUN_HOURS, price_col=TARGET_COLUMN)
    cont_total_cost = calculate_schedule_cost(POWER_CONSUMPTION_KW, cont_sched[TARGET_COLUMN])
    cont_savings = baseline_total_cost - cont_total_cost
    
    # Calculate interruptible operation
    int_sched, int_mwh_sum = optimize_interruptible(forecast_df, RUN_HOURS, price_col=TARGET_COLUMN)
    int_total_cost = calculate_schedule_cost(POWER_CONSUMPTION_KW, int_sched[TARGET_COLUMN])
    int_savings = baseline_total_cost - int_total_cost

    print(" THE NAIVE SCHEDULE (08:00 start)")
    print(f"Cost to Run: €{baseline_total_cost:.2f}\n")
    
    print(" OPTION A: CONTINUOUS OPERATION")
    print(f"Schedule: {cont_sched['Datetime'].iloc[0].strftime('%H:%M')} -> {cont_sched['Datetime'].iloc[-1].strftime('%H:%M')}")
    print(f"Cost to Run: €{cont_total_cost:.2f}")
    if cont_total_cost < 0:
        print("    NEGATIVE PRICE ALERT: You are getting paid to consume this energy!")
    print(f" Savings vs Naive: €{cont_savings:.2f}\n")
    
    print("⚡ OPTION B: INTERRUPTIBLE OPERATION")
    print(f"Scheduled Hours: {', '.join(int_sched['Datetime'].dt.strftime('%H:%M').tolist())}")
    print(f"Cost to Run: €{int_total_cost:.2f}")
    if int_total_cost < 0:
        print("    NEGATIVE PRICE ALERT: You are getting paid to consume this energy!")
    print(f" Savings vs Naive: €{int_savings:.2f}\n")
    print("-" * 60)

if __name__ == "__main__":
    run_optimizer()