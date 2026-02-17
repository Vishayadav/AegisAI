"""
AegisAI - Unified AI Observability Dashboard for Banking
Enterprise-grade monitoring platform for ML + LLM systems
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from collections import deque
import time

# Page configuration
st.set_page_config(
    page_title="AegisAI - AI Governance Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# Custom CSS Styling
# ============================

def inject_custom_css():
    """Inject premium dark theme CSS"""
    custom_css = """
    <style>
    :root {
        --primary: #0F62FE;
        --success: #24A148;
        --warning: #F1C21B;
        --danger: #DA1E28;
        --dark-bg: #0F1419;
        --dark-card: #161B22;
        --dark-border: #30363D;
        --text-primary: #E6EAEF;
        --text-secondary: #8B949E;
    }
    
    * {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    
    .main {
        background-color: var(--dark-bg);
        color: var(--text-primary);
    }
    
    [data-testid="stSidebar"] {
        background-color: var(--dark-card);
        border-right: 1px solid var(--dark-border);
    }
    
    .stMetric {
        background-color: var(--dark-card);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid var(--dark-border);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #0f1419 100%);
        padding: 24px;
        border-radius: 12px;
        border: 1px solid var(--dark-border);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(15, 98, 254, 0.15);
    }
    
    .status-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 14px;
        letter-spacing: 0.5px;
    }
    
    .status-stable {
        background-color: rgba(36, 161, 72, 0.2);
        color: #24A148;
        border: 1px solid #24A148;
    }
    
    .status-warning {
        background-color: rgba(241, 194, 27, 0.2);
        color: #F1C21B;
        border: 1px solid #F1C21B;
    }
    
    .status-danger {
        background-color: rgba(218, 30, 40, 0.2);
        color: #DA1E28;
        border: 1px solid #DA1E28;
    }
    
    .risk-indicator {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        padding: 24px;
        border-radius: 12px;
        background: rgba(15, 98, 254, 0.1);
        border: 2px solid var(--primary);
        font-size: 24px;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .risk-low {
        background: rgba(36, 161, 72, 0.1);
        border-color: #24A148;
        color: #24A148;
    }
    
    .risk-medium {
        background: rgba(241, 194, 27, 0.1);
        border-color: #F1C21B;
        color: #F1C21B;
    }
    
    .risk-high {
        background: rgba(218, 30, 40, 0.1);
        border-color: #DA1E28;
        color: #DA1E28;
    }
    
    .alert-item {
        background-color: var(--dark-card);
        padding: 16px;
        border-left: 4px solid var(--danger);
        border-radius: 8px;
        margin-bottom: 12px;
        border: 1px solid var(--dark-border);
    }
    
    .alert-item.warning {
        border-left-color: var(--warning);
    }
    
    .alert-item.info {
        border-left-color: var(--primary);
    }
    
    .header-title {
        color: var(--text-primary);
        font-size: 32px;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 8px;
    }
    
    .header-subtitle {
        color: var(--text-secondary);
        font-size: 14px;
        margin-bottom: 24px;
    }
    
    .divider {
        border: 0;
        height: 1px;
        background: linear-gradient(to right, transparent, var(--dark-border), transparent);
        margin: 24px 0;
    }
    
    .table-container {
        background-color: var(--dark-card);
        border: 1px solid var(--dark-border);
        border-radius: 12px;
        padding: 16px;
        overflow-x: auto;
    }
    
    [data-testid="stCheckbox"] label {
        color: var(--text-primary);
        font-weight: 500;
    }
    
    [data-testid="stSelectbox"] label {
        color: var(--text-primary);
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        border-bottom: 1px solid var(--dark-border);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: var(--text-secondary);
        border-radius: 8px 8px 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        color: var(--primary);
        border-bottom: 2px solid var(--primary);
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# ============================
# Session State Initialization
# ============================

def init_session_state():
    """Initialize session state variables"""
    if 'ml_history' not in st.session_state:
        st.session_state.ml_history = deque(maxlen=60)
    if 'llm_history' not in st.session_state:
        st.session_state.llm_history = deque(maxlen=60)
    if 'alerts' not in st.session_state:
        st.session_state.alerts = deque(maxlen=50)
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now()
    if 'page' not in st.session_state:
        st.session_state.page = "Overview"

init_session_state()


# ============================
# Data Generation Functions
# ============================

def generate_ml_metrics(drift_trigger=False, drift_base=0.15):
    """Generate ML monitoring metrics"""
    accuracy = np.random.normal(0.92, 0.02) if not drift_trigger else np.random.normal(0.78, 0.05)
    accuracy = np.clip(accuracy, 0.6, 0.99)
    
    precision = np.random.normal(0.90, 0.03) if not drift_trigger else np.random.normal(0.75, 0.05)
    precision = np.clip(precision, 0.6, 0.98)
    
    recall = np.random.normal(0.89, 0.03) if not drift_trigger else np.random.normal(0.74, 0.05)
    recall = np.clip(recall, 0.6, 0.98)
    
    drift_score = drift_base + (0.5 * np.random.random()) if drift_trigger else drift_base + (0.1 * np.random.random())
    drift_score = np.clip(drift_score, 0, 1)
    
    bias_score = np.random.uniform(0.05, 0.15)
    latency_ms = np.random.uniform(45, 150)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'drift_score': drift_score,
        'bias_score': bias_score,
        'latency_ms': latency_ms,
        'timestamp': datetime.now()
    }


def generate_llm_metrics(attack_trigger=False, cost_trigger=False, hallucination_base=0.05):
    """Generate LLM monitoring metrics"""
    latency_ms = np.random.uniform(450, 1500) if not attack_trigger else np.random.uniform(1200, 3000)
    
    token_usage = np.random.randint(150, 800) if not cost_trigger else np.random.randint(800, 2500)
    
    cost_usd = (token_usage / 1000) * 0.03 if not cost_trigger else (token_usage / 1000) * 0.05
    
    hallucination_rate = hallucination_base + (0.25 * np.random.random()) if attack_trigger else hallucination_base + (0.03 * np.random.random())
    hallucination_rate = np.clip(hallucination_rate, 0, 1)
    
    safety_flag = attack_trigger or np.random.random() < hallucination_rate
    
    throughput_rpm = np.random.uniform(45, 120)
    
    return {
        'latency_ms': latency_ms,
        'token_usage': token_usage,
        'cost_usd': cost_usd,
        'hallucination_rate': hallucination_rate,
        'safety_flag': safety_flag,
        'throughput_rpm': throughput_rpm,
        'timestamp': datetime.now()
    }


def get_ml_status(metrics):
    """Determine ML status from metrics"""
    if metrics['drift_score'] > 0.5:
        return "High Risk", "danger"
    elif metrics['drift_score'] > 0.3:
        return "Warning", "warning"
    elif metrics['accuracy'] < 0.8:
        return "At Risk", "warning"
    return "Stable", "success"


def get_llm_status(metrics):
    """Determine LLM status from metrics"""
    if metrics['safety_flag'] or metrics['hallucination_rate'] > 0.2:
        return "Unsafe", "danger"
    elif metrics['latency_ms'] > 2000 or metrics['hallucination_rate'] > 0.1:
        return "Warning", "warning"
    return "Safe", "success"


def calculate_global_risk(ml_metrics, llm_metrics):
    """Calculate overall risk level"""
    risk_score = 0
    
    if ml_metrics['drift_score'] > 0.5:
        risk_score += 40
    elif ml_metrics['drift_score'] > 0.3:
        risk_score += 20
    
    if ml_metrics['accuracy'] < 0.8:
        risk_score += 20
    
    if llm_metrics['safety_flag'] or llm_metrics['hallucination_rate'] > 0.2:
        risk_score += 40
    elif llm_metrics['hallucination_rate'] > 0.1:
        risk_score += 20
    
    risk_score = min(risk_score, 100)
    
    if risk_score >= 70:
        return "HIGH", "danger"
    elif risk_score >= 40:
        return "MEDIUM", "warning"
    return "LOW", "success"


def generate_alert(ml_metrics, llm_metrics, existing_alerts):
    """Generate relevant alerts based on metrics"""
    new_alerts = []
    
    if ml_metrics['drift_score'] > 0.4:
        new_alerts.append({
            'type': 'danger',
            'title': '⚠️ ML Model Drift Detected',
            'message': f"Drift score: {ml_metrics['drift_score']:.2%}. Action required.",
            'timestamp': datetime.now()
        })
    
    if ml_metrics['accuracy'] < 0.8:
        new_alerts.append({
            'type': 'warning',
            'title': '⚠️ Model Accuracy Degradation',
            'message': f"Current accuracy: {ml_metrics['accuracy']:.2%}. Monitor closely.",
            'timestamp': datetime.now()
        })
    
    if llm_metrics['safety_flag']:
        new_alerts.append({
            'type': 'danger',
            'title': '🚨 LLM Safety Incident',
            'message': "Unsafe response pattern detected. Immediate review required.",
            'timestamp': datetime.now()
        })
    
    if llm_metrics['hallucination_rate'] > 0.15:
        new_alerts.append({
            'type': 'warning',
            'title': '⚠️ Hallucination Rate Elevated',
            'message': f"Hallucination rate: {llm_metrics['hallucination_rate']:.2%}",
            'timestamp': datetime.now()
        })
    
    if llm_metrics['latency_ms'] > 2000:
        new_alerts.append({
            'type': 'warning',
            'title': '⏱️ High LLM Latency',
            'message': f"Latency: {llm_metrics['latency_ms']:.0f}ms. Check system load.",
            'timestamp': datetime.now()
        })
    
    for alert in new_alerts:
        st.session_state.alerts.appendleft(alert)


# ============================
# Header Components
# ============================

def render_top_navbar():
    """Render top navigation bar"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown("### 🛡️ AegisAI")
    
    with col2:
        pass
    
    with col3:
        env_badge = '<span class="status-badge" style="background-color: rgba(36, 161, 72, 0.2); color: #24A148; border: 1px solid #24A148;">🟢 Production</span>'
        st.markdown(env_badge, unsafe_allow_html=True)
        last_update = st.session_state.last_update.strftime("%Y-%m-%d %H:%M:%S UTC")
        st.caption(f"Last Updated: {last_update}")


def render_sidebar_navigation():
    """Render sidebar navigation"""
    st.sidebar.markdown("### Navigation")
    
    pages = ["Overview", "ML Observability", "LLM Observability", "Governance & Alerts"]
    selected_page = st.sidebar.radio("", pages, label_visibility="collapsed")
    st.session_state.page = selected_page
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Simulation Controls")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.session_state.trigger_drift = st.checkbox("ML Drift", value=False)
    with col2:
        st.session_state.trigger_hallucination = st.checkbox("LLM Hallucination", value=False)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.session_state.trigger_cost = st.checkbox("High Token Cost", value=False)
    with col2:
        st.session_state.trigger_safety = st.checkbox("Safety Incident", value=False)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    
    if st.session_state.ml_history and st.session_state.llm_history:
        latest_ml = st.session_state.ml_history[-1]
        latest_llm = st.session_state.llm_history[-1]
        
        ml_status, ml_color = get_ml_status(latest_ml)
        llm_status, llm_color = get_llm_status(latest_llm)
        
        st.sidebar.metric("ML Status", ml_status)
        st.sidebar.metric("LLM Status", llm_status)


# ============================
# Page Components
# ============================

def render_overview_page():
    """Render Overview/Executive Dashboard"""
    st.markdown('<div class="header-title">Executive Control Room</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Real-time AI system health and governance</div>', unsafe_allow_html=True)
    
    # Generate fresh metrics
    ml_metrics = generate_ml_metrics(
        drift_trigger=st.session_state.get('trigger_drift', False)
    )
    llm_metrics = generate_llm_metrics(
        attack_trigger=st.session_state.get('trigger_hallucination', False),
        cost_trigger=st.session_state.get('trigger_cost', False)
    )
    
    st.session_state.ml_history.append(ml_metrics)
    st.session_state.llm_history.append(llm_metrics)
    
    generate_alert(ml_metrics, llm_metrics, st.session_state.alerts)
    st.session_state.last_update = datetime.now()
    
    # Global Risk Indicator
    risk_level, risk_color = calculate_global_risk(ml_metrics, llm_metrics)
    risk_icons = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
    risk_class = f"risk-indicator risk-{risk_color}"
    
    st.markdown(f'<div class="{risk_class}">AI System Risk: {risk_icons[risk_level]} {risk_level}</div>', unsafe_allow_html=True)
    
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    
    # KPI Cards
    st.markdown("### Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 ML Accuracy", f"{ml_metrics['accuracy']:.2%}", 
                 delta=f"{(ml_metrics['accuracy'] - 0.92)*100:+.1f}%")
    
    with col2:
        st.metric("📈 Model Drift", f"{ml_metrics['drift_score']:.2%}", 
                 delta=f"{(ml_metrics['drift_score'] - 0.15)*100:+.1f}%")
    
    with col3:
        st.metric("⏱️ LLM Latency", f"{ml_metrics['latency_ms']:.0f}ms", 
                 delta=f"{llm_metrics['latency_ms'] - 800:.0f}ms")
    
    with col4:
        st.metric("💰 Daily LLM Cost", f"${llm_metrics['cost_usd']*100:.2f}", 
                 delta=f"+${llm_metrics['cost_usd']*10:.2f}")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Model Accuracy Trend")
        if st.session_state.ml_history:
            df_ml = pd.DataFrame(list(st.session_state.ml_history))
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=df_ml['accuracy'],
                mode='lines+markers',
                name='Accuracy',
                line=dict(color='#24A148', width=3),
                marker=dict(size=6),
                fill='tozeroy',
                fillcolor='rgba(36, 161, 72, 0.1)'
            ))
            fig.update_layout(
                template='plotly_dark',
                hovermode='x unified',
                showlegend=False,
                height=350,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(showgrid=False),
                yaxis=dict(range=[0.7, 1.0], showgrid=True, gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
    
    with col2:
        st.markdown("#### Token Usage Distribution")
        if st.session_state.llm_history:
            df_llm = pd.DataFrame(list(st.session_state.llm_history))
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=df_llm['token_usage'],
                marker=dict(color='#0F62FE', opacity=0.8),
                name='Tokens'
            ))
            fig.update_layout(
                template='plotly_dark',
                showlegend=False,
                height=350,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
    
    st.markdown("---")
    
    # Alert Feed
    st.markdown("#### 🚨 Active Alerts")
    if st.session_state.alerts:
        for alert in list(st.session_state.alerts)[:5]:
            alert_html = f'''
            <div class="alert-item {alert['type']}">
                <div style="font-weight: 600; margin-bottom: 4px;">{alert['title']}</div>
                <div style="font-size: 13px; color: #8B949E;">{alert['message']}</div>
                <div style="font-size: 11px; color: #6E7681; margin-top: 8px;">{alert['timestamp'].strftime("%H:%M:%S")}</div>
            </div>
            '''
            st.markdown(alert_html, unsafe_allow_html=True)
    else:
        st.info("✅ No active alerts. System operating normally.")


def render_ml_observability_page():
    """Render ML Observability Page"""
    st.markdown('<div class="header-title">ML Observability</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Deep dive into model performance and drift detection</div>', unsafe_allow_html=True)
    
    if not st.session_state.ml_history:
        st.warning("Collecting metrics...")
        return
    
    latest_ml = st.session_state.ml_history[-1]
    ml_status, ml_color = get_ml_status(latest_ml)
    
    # Status and Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_html = f'<div class="status-badge status-{ml_color}">{ml_status}</div>'
        st.markdown(status_html, unsafe_allow_html=True)
        st.caption("Model Status")
    
    with col2:
        st.metric("Accuracy", f"{latest_ml['accuracy']:.2%}")
    
    with col3:
        st.metric("Precision", f"{latest_ml['precision']:.2%}")
    
    with col4:
        st.metric("Recall", f"{latest_ml['recall']:.2%}")
    
    st.markdown("---")
    
    # Detailed Metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Drift Score Gauge")
        drift = latest_ml['drift_score']
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=drift * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            delta={'reference': 15},
            title={'text': "Data Drift (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#0F62FE'},
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(36, 161, 72, 0.3)'},
                    {'range': [30, 60], 'color': 'rgba(241, 194, 27, 0.3)'},
                    {'range': [60, 100], 'color': 'rgba(218, 30, 40, 0.3)'}
                ],
                'threshold': {
                    'line': {'color': 'red', 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=350, template='plotly_dark', margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
    
    with col2:
        st.markdown("#### Bias Score")
        bias = latest_ml['bias_score']
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=bias * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Bias Level (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#F1C21B'},
                'steps': [
                    {'range': [0, 10], 'color': 'rgba(36, 161, 72, 0.3)'},
                    {'range': [10, 20], 'color': 'rgba(241, 194, 27, 0.3)'},
                    {'range': [20, 100], 'color': 'rgba(218, 30, 40, 0.3)'}
                ]
            }
        ))
        fig.update_layout(height=350, template='plotly_dark', margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
    
    st.markdown("---")
    
    # Performance Trends
    st.markdown("#### Performance Trends")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.ml_history:
            df_ml = pd.DataFrame(list(st.session_state.ml_history))
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=df_ml['accuracy'], name='Accuracy', line=dict(color='#24A148')))
            fig.add_trace(go.Scatter(y=df_ml['precision'], name='Precision', line=dict(color='#0F62FE')))
            fig.add_trace(go.Scatter(y=df_ml['recall'], name='Recall', line=dict(color='#F1C21B')))
            fig.update_layout(template='plotly_dark', height=300, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
    
    with col2:
        if st.session_state.ml_history:
            df_ml = pd.DataFrame(list(st.session_state.ml_history))
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=df_ml['latency_ms'], fill='tozeroy', name='Latency', line=dict(color='#DA1E28')))
            fig.update_layout(template='plotly_dark', height=300, hovermode='x unified', yaxis_title='Latency (ms)')
            st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
    
    st.markdown("---")
    
    # Recent Predictions Table
    st.markdown("#### Recent Model Predictions")
    recent_predictions = []
    for i, metric in enumerate(list(st.session_state.ml_history)[-10:]):
        recent_predictions.append({
            'Timestamp': metric['timestamp'].strftime("%H:%M:%S"),
            'Accuracy': f"{metric['accuracy']:.2%}",
            'Drift': f"{metric['drift_score']:.2%}",
            'Bias': f"{metric['bias_score']:.2%}",
            'Latency': f"{metric['latency_ms']:.0f}ms",
            'Status': 'High Risk' if metric['drift_score'] > 0.5 else ('Warning' if metric['drift_score'] > 0.3 else 'Stable')
        })
    
    df_pred = pd.DataFrame(recent_predictions)
    st.dataframe(df_pred, use_container_width=True, hide_index=True)


def render_llm_observability_page():
    """Render LLM Observability Page"""
    st.markdown('<div class="header-title">LLM Observability</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Monitor language model performance, safety, and cost</div>', unsafe_allow_html=True)
    
    if not st.session_state.llm_history:
        st.warning("Collecting metrics...")
        return
    
    latest_llm = st.session_state.llm_history[-1]
    llm_status, llm_color = get_llm_status(latest_llm)
    
    # Status and Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_html = f'<div class="status-badge status-{llm_color}">{llm_status}</div>'
        st.markdown(status_html, unsafe_allow_html=True)
        st.caption("Safety Status")
    
    with col2:
        st.metric("Latency", f"{latest_llm['latency_ms']:.0f}ms")
    
    with col3:
        st.metric("Tokens/Call", f"{latest_llm['token_usage']:.0f}")
    
    with col4:
        st.metric("Cost/Call", f"${latest_llm['cost_usd']:.4f}")
    
    st.markdown("---")
    
    # Safety Indicators
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Hallucination Rate")
        hallucination = latest_llm['hallucination_rate']
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=hallucination * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            delta={'reference': 5},
            title={'text': "Hallucination Rate (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#0F62FE'},
                'steps': [
                    {'range': [0, 5], 'color': 'rgba(36, 161, 72, 0.3)'},
                    {'range': [5, 15], 'color': 'rgba(241, 194, 27, 0.3)'},
                    {'range': [15, 100], 'color': 'rgba(218, 30, 40, 0.3)'}
                ]
            }
        ))
        fig.update_layout(height=350, template='plotly_dark', margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
    
    with col2:
        st.markdown("#### Safety Compliance")
        compliance_score = 100 if not latest_llm['safety_flag'] else 30
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=compliance_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Compliance Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#24A148' if compliance_score > 50 else '#DA1E28'},
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(218, 30, 40, 0.3)'},
                    {'range': [50, 100], 'color': 'rgba(36, 161, 72, 0.3)'}
                ]
            }
        ))
        fig.update_layout(height=350, template='plotly_dark', margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
    
    st.markdown("---")
    
    # Cost and Performance Trends
    st.markdown("#### Cost & Performance Trends")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.llm_history:
            df_llm = pd.DataFrame(list(st.session_state.llm_history))
            df_llm['cumulative_cost'] = df_llm['cost_usd'].cumsum()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=df_llm['cumulative_cost'],
                fill='tozeroy',
                name='Cumulative Cost',
                line=dict(color='#F1C21B')
            ))
            fig.update_layout(
                template='plotly_dark',
                height=300,
                hovermode='x unified',
                yaxis_title='Cost (USD)'
            )
            st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
    
    with col2:
        if st.session_state.llm_history:
            df_llm = pd.DataFrame(list(st.session_state.llm_history))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=df_llm['hallucination_rate']*100, name='Hallucination %', line=dict(color='#DA1E28')))
            fig.add_trace(go.Scatter(y=df_llm['throughput_rpm'], name='Throughput (RPM)', line=dict(color='#0F62FE')))
            fig.update_layout(template='plotly_dark', height=300, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
    
    st.markdown("---")
    
    # LLM Interaction Log
    st.markdown("#### LLM Interaction Log")
    interaction_log = []
    for i, metric in enumerate(list(st.session_state.llm_history)[-10:]):
        interaction_log.append({
            'Time': metric['timestamp'].strftime("%H:%M:%S"),
            'Latency': f"{metric['latency_ms']:.0f}ms",
            'Tokens': f"{metric['token_usage']:.0f}",
            'Cost': f"${metric['cost_usd']:.4f}",
            'Hallucination': f"{metric['hallucination_rate']:.2%}",
            'Safety': '🚨 UNSAFE' if metric['safety_flag'] else '✅ SAFE'
        })
    
    df_log = pd.DataFrame(interaction_log)
    st.dataframe(df_log, use_container_width=True, hide_index=True)


def render_governance_page():
    """Render Governance & Responsible AI Page"""
    st.markdown('<div class="header-title">Governance & Responsible AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Compliance monitoring, alerts, and responsible AI controls</div>', unsafe_allow_html=True)
    
    # Active Alerts
    st.markdown("#### 🚨 Active Alerts & Incidents")
    if st.session_state.alerts:
        for alert in list(st.session_state.alerts)[:10]:
            alert_html = f'''
            <div class="alert-item {alert['type']}">
                <div style="font-weight: 600; margin-bottom: 4px;">{alert['title']}</div>
                <div style="font-size: 13px; color: #8B949E;">{alert['message']}</div>
                <div style="font-size: 11px; color: #6E7681; margin-top: 8px;">{alert['timestamp'].strftime("%H:%M:%S UTC")}</div>
            </div>
            '''
            st.markdown(alert_html, unsafe_allow_html=True)
    else:
        st.info("✅ No alerts. All systems compliant.")
    
    st.markdown("---")
    
    # Compliance Indicators
    st.markdown("#### Compliance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔒 Audit Logs", "12,547", "+256")
    
    with col2:
        st.metric("👥 Human Reviews", "98.2%", "+1.2%")
    
    with col3:
        st.metric("⚖️ Bias Score", "8.3%", "-0.5%")
    
    with col4:
        st.metric("📋 Policy Violations", "0", "=")
    
    st.markdown("---")
    
    # Risk Heatmap Data
    st.markdown("#### AI Risk by Component")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Heatmap visualization
        heatmap_data = np.random.uniform(0.2, 0.8, (5, 4))
        components = ['Model Drift', 'Accuracy', 'Hallucination', 'Safety', 'Cost']
        features = ['Feature A', 'Feature B', 'Feature C', 'Feature D']
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=features,
            y=components,
            colorscale='RdYlGn_r',
            colorbar=dict(title="Risk Score")
        ))
        fig.update_layout(
            template='plotly_dark',
            height=400,
            xaxis_title="",
            yaxis_title=""
        )
        st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
    
    with col2:
        st.markdown("#### Responsible AI Controls")
        st.markdown("""
        ##### 🤖 AI Governance Framework
        
        **1. Bias Monitoring**
        - Real-time model fairness tracking across demographics
        - Automated bias detection in predictions
        - Regular fairness audits (weekly)
        
        **2. Hallucination Tracking**
        - Content validation systems for LLM outputs
        - Fact-checking integration
        - User feedback loops for inaccuracies
        
        **3. Safety Enforcement**
        - Automated content filters
        - Compliance rule engines
        - Rate limiting on sensitive operations
        
        **4. Audit Logging**
        - Complete trace of all AI decisions
        - Decision explainability records
        - Regulatory compliance documentation
        
        **5. Human-in-the-Loop**
        - Critical decisions require human approval
        - Escalation procedures for anomalies
        - Manual override capabilities
        
        **6. Transparency & Explainability**
        - Model explanation generation
        - Feature importance tracking
        - Stakeholder reporting dashboards
        """)
    
    st.markdown("---")
    
    # Escalation & Approval Metrics
    st.markdown("#### Escalation & Approval Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("⬆️ Escalations (24h)", "12", "+3")
        st.caption("High-risk decisions escalated to humans")
    
    with col2:
        st.metric("✅ Approved Decisions", "1,247", "-5")
        st.caption("Decisions approved after human review")
    
    with col3:
        st.metric("⏸️ Rejected Decisions", "8", "+2")
        st.caption("Decisions rejected by human oversight")
    
    st.markdown("---")
    
    # Compliance Metrics Table
    st.markdown("#### Compliance Metrics Dashboard")
    
    compliance_metrics = pd.DataFrame({
        'Metric': [
            'Data Privacy Compliance',
            'Model Explainability',
            'Fairness Score',
            'Safety Coverage',
            'Audit Trail Completeness'
        ],
        'Status': ['✅ Pass', '✅ Pass', '⚠️ Warning', '✅ Pass', '✅ Pass'],
        'Score': ['100%', '94%', '78%', '96%', '100%'],
        'Last Audit': ['Today', 'Today', 'Today', 'Yesterday', 'Today']
    })
    
    st.dataframe(compliance_metrics, use_container_width=True, hide_index=True)


# ============================
# Main Application Loop
# ============================

def main():
    """Main application entry point"""
    inject_custom_css()
    
    # Render navbar
    render_top_navbar()
    st.markdown("---")
    
    # Render sidebar
    render_sidebar_navigation()
    
    # Route to page
    if st.session_state.page == "Overview":
        render_overview_page()
    elif st.session_state.page == "ML Observability":
        render_ml_observability_page()
    elif st.session_state.page == "LLM Observability":
        render_llm_observability_page()
    elif st.session_state.page == "Governance & Alerts":
        render_governance_page()
    
    # Auto-refresh simulation
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()


if __name__ == "__main__":
    main()
