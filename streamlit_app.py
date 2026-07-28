import streamlit as st
import pandas as pd
import sys
import os
import urllib.parse
from sqlalchemy import create_engine
import plotly.graph_objects as go

sys.path.append(os.path.abspath('src'))
from optimizer import optimize_continuous, optimize_interruptible

st.set_page_config(page_title="AI Energy Optimizer", page_icon="⚡", layout="wide")
st.title("⚡ Heavy Machinery Energy Optimizer")
st.markdown("Use machine learning to forecast Day-Ahead electricity prices and find the cheapest hours to run factory equipment.")

@st.cache_resource
def init_connection():
    db_pass = urllib.parse.quote_plus(os.environ.get("DB_PASSWORD", "fallback_local_pass"))
    db_host = os.environ.get("DB_HOST", "localhost")
    engine = create_engine(f"postgresql://postgres:{db_pass}@{db_host}:5432/postgres")
    return engine

@st.cache_data(ttl=3600)
def load_forecast_data():
    try:
        engine = init_connection()
        query = """SELECT * FROM tomorrow_predictions ORDER BY "datetime_berlin" DESC LIMIT 48"""
        df = pd.read_sql(query, engine)
        df['Datetime'] = pd.to_datetime(df['datetime_berlin'])
        return df.iloc[::-1].reset_index(drop=True)
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return pd.DataFrame()

df = load_forecast_data()

if not df.empty:
    # ==========================================
    #  EXACT GERMAN STATUTORY TAX & LEVY STACK
    # ==========================================
    STROMSTEUER_RES = 20.50      
    KWKG_UMLAGE = 4.46          
    OFFSHORE_UMLAGE = 9.41       
    STROMNEV_UMLAGE = 15.59      
    STATUTORY_LEVIES_TOTAL = STROMSTEUER_RES + KWKG_UMLAGE + OFFSHORE_UMLAGE + STROMNEV_UMLAGE 
    
    RETAIL_MARGIN = 20.00        
    STROMSTEUER_IND = 0.50       
    IND_REDUCED_LEVIES = 1.00    
    VAT = 1.19
    
    REGIONS = {
        "Nuremberg (N-ERGIE)": {"ind_grid": 23.50, "res_grid": 98.20, "concession": 19.90},
        "Munich (SWM)": {"ind_grid": 21.00, "res_grid": 85.50, "concession": 19.90},
        "Berlin (Stromnetz)": {"ind_grid": 26.80, "res_grid": 105.40, "concession": 23.90},
        "Hamburg (Stromnetz)": {"ind_grid": 24.10, "res_grid": 92.00, "concession": 23.90}
    }

    st.sidebar.header(" Regional Pricing Setup")
    selected_region = st.sidebar.selectbox("Select your grid region:", options=list(REGIONS.keys()), index=0)

    region_data = REGIONS[selected_region]
    ind_grid = region_data["ind_grid"]
    res_grid = region_data["res_grid"]
    concession = region_data["concession"]

    # ==========================================
    #  FINAL PRICING CALCULATIONS
    # ==========================================
    df['Wholesale_Price'] = df['Predicted_Price_p50']
    df['Industrial_Final_Price'] = df['Wholesale_Price'] + ind_grid + STROMSTEUER_IND + IND_REDUCED_LEVIES
    df['Residential_Final_Price'] = (df['Wholesale_Price'] + res_grid + concession + STATUTORY_LEVIES_TOTAL + RETAIL_MARGIN) * VAT

    if 'Actual_Price' in df.columns:
        df['Actual_Industrial_Price'] = df['Actual_Price'] + ind_grid + STROMSTEUER_IND + IND_REDUCED_LEVIES
        df['Actual_Residential_Price'] = (df['Actual_Price'] + res_grid + concession + STATUTORY_LEVIES_TOTAL + RETAIL_MARGIN) * VAT
    
    # --- UI TABS ---
    tab1, tab2, tab3 = st.tabs(["⚙️ Operations Optimizer", "📈 Market Risk Forecast", "🧠 AI Explainability"])
    
    with tab1:
        tomorrow_df = df.tail(24).reset_index(drop=True)
        target_date = tomorrow_df['Datetime'].iloc[-1].strftime('%A, %B %d, %Y')
        st.subheader(f"Schedule Your Machinery ({target_date})")        
        
        col1, col2 = st.columns(2)
        with col1:
            run_hours = st.number_input("Required Run Time (Hours):", min_value=1, max_value=24, value=8, step=1)
        with col2:
            power_kw = st.number_input("Machine Power (kW):", min_value=10, max_value=10000, value=500, step=50)

        st.markdown("#### 👷 Operational Constraints")
        shift_start, shift_end = st.slider("Select Legal Shift Hours (When can the machine run?)", min_value=0, max_value=23, value=(8, 18), format="%d:00")

        if st.button("Run Optimizer", type="primary"):
            shift_mask = (tomorrow_df['Datetime'].dt.hour >= shift_start) & (tomorrow_df['Datetime'].dt.hour <= shift_end)
            constrained_df = tomorrow_df[shift_mask].reset_index(drop=True)
            
            if run_hours > len(constrained_df):
                st.error(f"⚠️ Error: You requested {run_hours} hours of run time, but the shift window is only {len(constrained_df)} hours long!")
            else:
                cont_sched, _ = optimize_continuous(constrained_df, run_hours)
                int_sched, _ = optimize_interruptible(constrained_df, run_hours)
                
                power_mw = power_kw / 1000.0
                # Using the actual industrial rates for financial estimation
                cont_cost = power_mw * cont_sched['Industrial_Final_Price'].sum()
                int_cost = power_mw * int_sched['Industrial_Final_Price'].sum()

                st.success("Optimization Complete! Financial estimates based on AI predicted industrial rates.")
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.info("🔥 **OPTION A: Continuous Run**")
                    st.write(f"**Schedule:** {cont_sched['Datetime'].iloc[0].strftime('%H:%M')} to {cont_sched['Datetime'].iloc[-1].strftime('%H:%M')}")
                    st.write(f"**Estimated Cost:** €{cont_cost:.2f}")
                    
                with res_col2:
                    st.info("⚡ **OPTION B: Interruptible Run**")
                    st.write(f"**Scheduled Hours:** {', '.join(int_sched['Datetime'].dt.strftime('%H:%M').tolist())}")
                    st.write(f"**Estimated Cost:** €{int_cost:.2f}")

    with tab2:
        st.subheader("📊 48-Hour Final Cost Forecast vs Actuals")
        fig = go.Figure()
        
        df['datetime_berlin'] = pd.to_datetime(df['datetime_berlin'])
        if df['datetime_berlin'].dt.tz is None:
            df['datetime_berlin'] = df['datetime_berlin'].dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin')
        else:
            df['datetime_berlin'] = df['datetime_berlin'].dt.tz_convert('Europe/Berlin')
        df['datetime_berlin'] = df['datetime_berlin'].dt.tz_localize(None)

        # Predicted Lines
        fig.add_trace(go.Scatter(x=df['datetime_berlin'], y=df['Industrial_Final_Price'], mode='lines', line=dict(color='#00BFFF', width=2, dash='dot'), name=f'Predicted Industrial ({selected_region})', hovertemplate='%{y:.2f} €/MWh'))
        fig.add_trace(go.Scatter(x=df['datetime_berlin'], y=df['Residential_Final_Price'], mode='lines', line=dict(color='#32CD32', width=2, dash='dash'), name=f'Predicted Residential ({selected_region})', hovertemplate='%{y:.2f} €/MWh'))

        # Actual Lines
        if 'Actual_Industrial_Price' in df.columns:
            fig.add_trace(go.Scatter(x=df['datetime_berlin'], y=df['Actual_Industrial_Price'], mode='lines', line=dict(color='blue', width=2), name=f'Actual Industrial ({selected_region})', hovertemplate='%{y:.2f} €/MWh'))
            fig.add_trace(go.Scatter(x=df['datetime_berlin'], y=df['Actual_Residential_Price'], mode='lines', line=dict(color='green', width=2), name=f'Actual Residential ({selected_region})', hovertemplate='%{y:.2f} €/MWh'))

        fig.update_layout(hovermode="x unified", yaxis_title="Price (€/MWh)", xaxis_title="Time")
        st.plotly_chart(fig, width='stretch')

    with tab3:
        st.subheader("🧠 Why is the AI predicting this?")
        st.markdown("Understanding the market drivers behind the AI's most recent forecast.")
        try:
            st.image("visualizations/shap_waterfall.png", caption="Real-time Driver Analysis (SHAP Waterfall)", width='content')
            st.divider()
            st.image("visualizations/shap_summary.png", caption="Macro Feature Importance (Last 24 Hours)", width='content')
        except FileNotFoundError:
            st.warning("SHAP visualizations not found. The inference pipeline has not run yet.")