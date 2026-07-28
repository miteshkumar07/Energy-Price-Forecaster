def calculate_job_cost(power_kw, duration_hours, average_price_mwh):
    """
    Calculates total cost using an average price.
    Formula: (kW / 1000) * hours * price_per_MWh
    """
    power_mw = power_kw / 1000.0
    total_energy_mwh = power_mw * duration_hours
    return total_energy_mwh * average_price_mwh

def calculate_schedule_cost(power_kw, hourly_prices):
    """
    Calculates the exact cost across varying hourly prices.
    Each item in hourly_prices represents 1 hour of operation.
    """
    power_mw = power_kw / 1000.0
    return power_mw * sum(hourly_prices)