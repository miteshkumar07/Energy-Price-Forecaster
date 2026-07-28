import streamlit as st
import pandas as pd
import sys
import os
import urllib.parse
from sqlalchemy import create_engine
import plotly.graph_objects as go
from google.cloud import storage
from google.api_core.exceptions import NotFound

sys.path.append(os.path.abspath('src'))
from optimizer import optimize_continuous, optimize_interruptible

st.set_page_config(page_title="AI Energy Optimizer", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Gradient Title */
    .main-title {
        background: linear-gradient(90deg, #FDBB2D 0%, #22C1C3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.8rem !important;
        font-weight: 800;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    
    /* Custom Button */
    .stButton > button {
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%);
        color: white;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 15px rgba(255, 75, 43, 0.4);
        font-weight: 600;
        transition: all 0.3s ease;
        padding: 0.5rem 2rem;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 75, 43, 0.6);
        color: white;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 800;
        color: #22C1C3;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1e1e24;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">⚡ Energy Price Forecaster</h1>', unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.1rem; color: #a3a8b8; margin-bottom: 2rem;'>Empowering smart energy decisions with advanced machine learning. Forecast European Day-Ahead electricity prices, mitigate market risks, and optimize operational schedules with precise pricing models tailored for both industrial facilities and residential consumers.</p>", unsafe_allow_html=True)

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
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌐 Connect with Me")

    st.sidebar.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        .social-container {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-top: 10px;
        }
        .social-link {
            display: flex;
            align-items: center;
            gap: 10px;
            text-decoration: none !important;
            font-weight: 500;
            font-size: 15px;
        }
        .social-link:hover {
            opacity: 0.8;
        }
        .fa-linkedin { color: #0A66C2; }
        .fa-github { color: inherit; }
        .fa-globe { color: #00BFFF; }
    </style>
    <div class="social-container">
        <a class="social-link" href="https://www.linkedin.com/in/mitesh-kumar0707/" target="_blank">
            <i class="fab fa-linkedin fa-lg"></i> LinkedIn
        </a>
        <a class="social-link" href="https://github.com/miteshkumar07" target="_blank">
            <i class="fab fa-github fa-lg"></i> GitHub
        </a>
        <a class="social-link" href="https://miteshkumar.com/" target="_blank">
            <i class="fas fa-globe fa-lg"></i> Portfolio
        </a>
    </div>
    """, unsafe_allow_html=True)

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
            run_hours = st.number_input("Required Run Time (Hours):", min_value=1, max_value=24, value=10, step=1)
        with col2:
            power_kw = st.number_input("Machine Power (kW):", min_value=10, max_value=10000, value=500, step=50)

        st.markdown("#### 👷 Operational Constraints")
        shift_start, shift_end = st.slider("Select Legal Shift Hours (When can the machine run?)", min_value=0, max_value=23, value=(0, 23), format="%d:00")

        if st.button("Run Optimizer", type="primary"):
            shift_mask = (tomorrow_df['Datetime'].dt.hour >= shift_start) & (tomorrow_df['Datetime'].dt.hour <= shift_end)
            constrained_df = tomorrow_df[shift_mask].reset_index(drop=True)
            
            if run_hours > len(constrained_df):
                st.error(f"⚠️ Error: You requested {run_hours} hours of run time, but the shift window is only {len(constrained_df)} hours long!")
            else:
                cont_sched, _ = optimize_continuous(constrained_df, run_hours, price_col='Industrial_Final_Price')
                int_sched, _ = optimize_interruptible(constrained_df, run_hours, price_col='Industrial_Final_Price')
                
                power_mw = power_kw / 1000.0
                # Using the actual industrial rates for financial estimation
                cont_cost = power_mw * cont_sched['Industrial_Final_Price'].sum()
                int_cost = power_mw * int_sched['Industrial_Final_Price'].sum()

                st.success("✅ **Optimization Complete!** Financial estimates based on AI predicted industrial rates.")
                st.markdown("<br>", unsafe_allow_html=True)
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    with st.container(border=True):
                        st.markdown("### 🔥 OPTION A: Continuous Run")
                        st.markdown(f"**Optimal Window:** `{cont_sched['Datetime'].iloc[0].strftime('%H:%M')} - {cont_sched['Datetime'].iloc[-1].strftime('%H:%M')}`")
                        st.metric(label="Estimated Cost", value=f"€{cont_cost:.2f}")
                    
                with res_col2:
                    with st.container(border=True):
                        st.markdown("### ⚡ OPTION B: Interruptible Run")
                        st.markdown(f"**Scheduled Hours:** `{', '.join(int_sched['Datetime'].dt.strftime('%H:%M').tolist())}`")
                        st.metric(label="Estimated Cost", value=f"€{int_cost:.2f}")

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

        fig.update_layout(
            hovermode="x unified", 
            yaxis_title="Price (€/MWh)", 
            xaxis_title="Time", 
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=60)
        )
        st.plotly_chart(fig, width='stretch')

    with tab3:
        st.subheader("🧠 Why is the AI predicting this?")
        st.markdown("Understanding the market drivers behind the AI's most recent forecast.")
        bucket_name = "energy-visuals-mitesh"
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            
            col1, col2 = st.columns(2)
            
            # Image 1: SHAP Waterfall
            with col1:
                try:
                    blob_waterfall = bucket.blob("shap_waterfall.png")
                    waterfall_bytes = blob_waterfall.download_as_bytes()
                    st.image(waterfall_bytes, caption="Real-time Driver Analysis (SHAP Waterfall)", width='content')
                except NotFound:
                    st.info(" SHAP Waterfall analysis visual is generating...")
            
            # Image 2: SHAP Summary / Feature Importance
            with col2:
                try:
                    blob_summary = bucket.blob("shap_summary.png")
                    summary_bytes = blob_summary.download_as_bytes()
                    st.image(summary_bytes, caption="Macro Feature Importance (Last 24 Hours)", width='content')
                except NotFound:
                    st.info(" SHAP Summary analysis visual is generating...")
                    
        except Exception as e:
            st.error(f"Could not connect to Cloud Storage: {e}")