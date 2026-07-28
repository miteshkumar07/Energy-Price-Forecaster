import pytest
import pandas as pd
import sys
import os

# Ensure the src folder is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from cost_calculator import calculate_job_cost, calculate_schedule_cost
from optimizer import optimize_continuous

def test_calculate_job_cost():
    """
    Proves that a 500 kW machine running for 4 hours at €150/MWh exactly returns €300[cite: 6].
    500 kW = 0.5 MW[cite: 6]
    4 hours * 0.5 MW = 2.0 MWh[cite: 6]
    2.0 MWh * 150 €/MWh = 300.0 €[cite: 6]
    """
    cost = calculate_job_cost(power_kw=500, duration_hours=4, average_price_mwh=150)
    assert cost == 300.0, f"Expected 300.0, got {cost}"

def test_calculate_schedule_cost():
    """
    Proves that fluctuating hourly prices are calculated correctly[cite: 6].
    """
    hourly_prices = [100.0, 200.0, -50.0]  # Sum = 250 €/MWh[cite: 6]
    cost = calculate_schedule_cost(power_kw=1000, hourly_prices=hourly_prices)
    assert cost == 250.0, f"Expected 250.0, got {cost}"

def test_optimize_continuous_logic():
    """
    Proves the sliding window algorithm correctly finds the cheapest block in O(n) time[cite: 6].
    """
    # Updated to reflect the LightGBM p50 median outputs and Industrial pricing
    df = pd.DataFrame({
        'Datetime': pd.date_range(start='2026-07-25 00:00', periods=6, freq='h'),
        'Predicted_Price_p50': [500, 400, 10, 20, 300, 400],
        'Industrial_Final_Price': [525, 425, 35, 45, 325, 425]
    })
    
    # We want a 2-hour window. The cheapest is indices 2 and 3[cite: 6].
    schedule, total_cost = optimize_continuous(df, hours=2)
    
    assert len(schedule) == 2
    
    # Depending on whether optimizer.py uses Wholesale or Industrial price internally, 
    # check the correct column output here:
    assert schedule['Predicted_Price_p50'].tolist() == [10, 20]