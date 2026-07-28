import pandas as pd
from cost_calculator import calculate_schedule_cost
from optimizer import optimize_continuous
from db_utils import load_table

def run_roi_calculator():
    print(" Starting 3-Month ROI Calculator...")
    results_df = load_table('backtest_results')

    if results_df.empty:
        print(" Error: 'backtest_results' table is empty in database. Run backtest.py first.")
        return

    # Normalize datetime column names (depending on how the backtest script saves it)
    if 'datetime_berlin' in results_df.columns:
        results_df['Datetime'] = pd.to_datetime(results_df['datetime_berlin'])
    else:
        results_df['Datetime'] = pd.to_datetime(results_df['Datetime'])
        
    results_df['Date_Only'] = results_df['Datetime'].dt.date

    RUN_HOURS = 4
    POWER_CONSUMPTION_KW = 500
    PREDICTION_COL = 'Predicted_Industrial' if 'Predicted_Industrial' in results_df.columns else 'Predicted_Price_p50'
    ACTUAL_COL = 'Actual_Industrial' if 'Actual_Industrial' in results_df.columns else 'Actual_Price'   
    
    total_baseline_cost = 0.0
    total_optimized_cost = 0.0
    total_savings = 0.0
    negative_hours_count = 0
    
    unique_days = results_df['Date_Only'].unique()
    
    for day in unique_days:
        day_df = results_df[results_df['Date_Only'] == day].reset_index(drop=True)
        
        if len(day_df) < RUN_HOURS:
            continue
            
        # --- BASELINE COST (09:00 to 13:00) ---
        try:
            start_idx = day_df[day_df['Datetime'].dt.hour == 9].index[0]
            end_idx = start_idx + RUN_HOURS - 1
            baseline_schedule = day_df.loc[start_idx:end_idx]
            # Bill calculated using the actual target stack
            realized_baseline = calculate_schedule_cost(POWER_CONSUMPTION_KW, baseline_schedule[ACTUAL_COL])
        except IndexError:
            realized_baseline = calculate_schedule_cost(POWER_CONSUMPTION_KW, day_df.head(RUN_HOURS)[ACTUAL_COL])
        
        # --- AI-OPTIMIZED COST ---
        # The AI blindly schedules the machine using ONLY the Predicted_Industrial price.
        cont_sched, _ = optimize_continuous(day_df, RUN_HOURS, price_col=PREDICTION_COL)
        
        # The factory's financial bill is calculated on the Actual_Industrial price of those chosen hours[cite: 4].
        actual_prices = cont_sched[ACTUAL_COL]
        realized_cost = calculate_schedule_cost(POWER_CONSUMPTION_KW, actual_prices)
        
        if (actual_prices < 0).any():
            negative_hours_count += (actual_prices < 0).sum()
        
        total_baseline_cost += realized_baseline
        total_optimized_cost += realized_cost
        total_savings += (realized_baseline - realized_cost)

    print("\n" + "=" * 65)
    print(" 90-DAY RETURN ON INVESTMENT (ROI) REPORT")
    print("=" * 65)
    print(f" Machine Specs: {POWER_CONSUMPTION_KW} kW | {RUN_HOURS} Hours/Day")
    print(f" Total Days Evaluated: {len(unique_days)}")
    print("-" * 65)
    
    print(f" NAIVE COST (Static 09:00 to {9+RUN_HOURS-1}:59):")
    print(f"   €{total_baseline_cost:,.2f}\n")
    
    print(f" AI-OPTIMIZED COST (Dynamic Schedule):")
    print(f"   €{total_optimized_cost:,.2f}")
    
    if negative_hours_count > 0:
        print(f"    The optimizer captured {negative_hours_count} hours of NEGATIVE actual prices!")
        
    print(f"\n TOTAL SAVINGS OVER 3 MONTHS:")
    print(f"   €{total_savings:,.2f} saved ({(total_savings/total_baseline_cost)*100:.1f}% reduction)")
    
    annual_savings = (total_savings / len(unique_days)) * 365
    print(f" PROJECTED ANNUAL SAVINGS: €{annual_savings:,.2f}")
    print("=" * 65)

if __name__ == "__main__":
    run_roi_calculator()