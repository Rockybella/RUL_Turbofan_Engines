import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# 1. Page Configuration
st.set_page_config(
    page_title="AeroNet RUL | Turbofan Analytics",
    page_icon="🚀",
    layout="wide"
)

# 2. Advanced Professional UI (No Margins, Glassmorphism elements)
st.markdown("""
<style>
    .block-container { padding: 0rem 2rem !important; }
    .main { background-color: #F1F5F9; }
    
    /* Elegant Header */
    .custom-header {
        background: linear-gradient(135deg, #0F172A 0%, #2563EB 100%);
        padding: 3rem 2rem;
        margin: 0 -2rem 2rem -2rem;
        color: white;
        border-bottom: 4px solid #3B82F6;
    }

    /* Metric "Cards" */
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #E2E8F0;
        padding: 1.5rem !important;
        border-radius: 16px !important;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
    }
    
    /* Clean Tab Navigation */
    .stTabs [data-baseweb="tab-list"] { gap: 12px; background: transparent; }
    .stTabs [data-baseweb="tab"] {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 10px 30px;
        font-weight: 700;
        transition: all 0.3s;
    }
    .stTabs [aria-selected="true"] {
        background: #2563EB !important;
        color: white !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
</style>

<div class="custom-header">
    <div style='display:flex; align-items:center; gap:20px;'>
        <div style='background:rgba(255,255,255,0.1); padding:15px; border-radius:12px; font-size:40px;'>🛡️</div>
        <div>
            <h1 style='margin:0; font-size:2.2rem; letter-spacing:-1px;'>AeroNet RUL</h1>
            <p style='opacity:0.8; margin:0; font-weight:300;'>PHM Deep Learning Ensemble • NASA C-MAPSS turbofan engines</p>
            <p style='opacity:0.8; margin:0; font-weight:300;'>"This framework utilizes a multi-stage deep learning pipeline to extract spatial features and model long-term temporal dependencies, delivering reliable RUL predictions for proactive maintenance scheduling."</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 3. Sidebar
with st.sidebar:
    st.image("https://img.icons8.com", width=60)
    st.title("System Control")
    engine_id = st.selectbox("Select Engine Unit", ["Unit #1001", "Unit #1002", "Unit #1003"])
    st.divider()
    st.markdown("### Model Config")
    st.code("CAE-BiLSTM-AM\nKeras 3.0 / TF 2.15", language="text")
    st.divider()
    if st.button("🔄 Force Sensor Sync"):
        st.toast("Synchronizing Telemetry...")

# 4. KPI Metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Estimated RUL", "142 Cycles", "-12", help="Estimated cycles until maintenance limit (Threshold: 0)")
with c2:
    st.metric("Confidence Index", "95.8%", "±4.2", help="Model uncertainty based on MC-Dropout")
with c3:
    st.metric("Sensor Integrity", "Normal", "Stable", delta_color="normal")
with c4:
    # Custom colored Status 
    st.markdown("""<div style='background:white; border:1px solid #E2E8F0; padding:1.2rem; border-radius:16px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);'>
    <p style='margin:0; font-size:14px; color:#64748B;'>Risk Assessment</p>
    <p style='margin:0; font-size:24px; font-weight:bold; color:#10B981;'>SAFE</p>
    </div>""", unsafe_allow_html=True)

st.write("")

# 5. Dashboard Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 LIVE TELEMETRY", "📉 MODEL ANALYTICS", "🔍 RESIDUALS", "🧠 INTERPRETABILITY"])

def update_plot(fig):
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter, sans-serif"))
    return fig

with tab1:
    col_main, col_side = st.columns([2.5, 1])
    with col_main:
        st.subheader("Critical Sensor Fusion")
        # Mock sensor data
        t = np.linspace(0, 50, 100)
        sensors = pd.DataFrame({
            "Cycle": t,
            "T24 (LPC Temp)": 600 + 5*t + np.random.normal(0, 2, 100),
            "P30 (HPC Exit)": 500 + 2*t + np.random.normal(0, 1, 100),
            "BPR (Bypass Ratio)": 20 + 0.1*t + np.random.normal(0, 0.2, 100)
        })
        fig = px.line(sensors, x="Cycle", y=sensors.columns[1:], color_discrete_sequence=['#2563EB', '#F59E0B', '#10B981'])
        st.plotly_chart(update_plot(fig), use_container_width=True)
    
    with col_side:
        st.subheader("RUL Probability")
        # Gaussian Distribution
        mu, sigma = 142, 4.2
        x = np.linspace(125, 160, 100)
        y = (1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu)/sigma)**2))
        fig_prob = go.Figure()
        fig_prob.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', fillcolor='rgba(37, 99, 235, 0.1)', line_color='#2563EB'))
        fig_prob.add_vline(x=mu, line_dash="dash", line_color="#1E293B", annotation_text="Mode")
        st.plotly_chart(update_plot(fig_prob), use_container_width=True)
        st.info("The engine is operating within safe historical degradation bounds for FD001 profile.")

with tab2:
    # Training curves
    e = np.arange(50)
    train_df = pd.DataFrame({"Epoch": e, "MSE": np.exp(-e/8) + 0.05, "Val_MSE": np.exp(-e/7.5) + 0.08})
    c_a, c_b = st.columns(2)
    with c_a:
        st.plotly_chart(update_plot(px.line(train_df, x="Epoch", y=["MSE", "Val_MSE"], title="Loss Convergence")), use_container_width=True)
    with c_b:
        st.markdown("### Model Architecture Specs")
        st.table(pd.DataFrame({
            "Layer": ["Conv1D", "Bi-LSTM", "Attention", "Dense"],
            "Output Shape": ["(None, 50, 64)", "(None, 50, 128)", "(None, 50, 128)", "(None, 1)"],
            "Activation": ["ReLU", "Tanh", "Softmax", "Linear"]
        }))

with tab3:
    st.subheader("Error Characterization")
    err_df = pd.DataFrame({"Actual": np.random.randint(50, 150, 50)})
    err_df["Predicted"] = err_df["Actual"] + np.random.normal(0, 5, 50)
    fig_err = px.scatter(err_df, x="Actual", y="Predicted", trendline="ols", color_discrete_sequence=['#2563EB'])
    fig_err.add_shape(type="line", x0=50, y0=50, x1=150, y1=150, line=dict(color="#94A3B8", dash="dot"))
    st.plotly_chart(update_plot(fig_err), use_container_width=True)

with tab4:
    st.subheader("Sensor Importance (SHAP Analysis)")
    s_names = ['T24', 'T50', 'P30', 'Ps30', 'phi', 'NRc', 'NRf', 'BPR']
    imp = np.sort(np.random.rand(len(s_names)))
    fig_imp = px.bar(x=imp, y=s_names, orientation='h', color=imp, color_continuous_scale="Blues")
    fig_imp.update_layout(coloraxis_showscale=False)
    st.plotly_chart(update_plot(fig_imp), use_container_width=True)

# Footer
st.markdown("<div style='text-align:center; padding:3rem; color:#94A3B8;'>© 2025 AeroNet RUL Predictive Maintenance Systems • Confidential Engineering Data</div>", unsafe_allow_html=True)
