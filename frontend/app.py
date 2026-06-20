import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
import torch

# Add root to sys.path so we can import from models and api
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.deps import get_ml_cache

st.set_page_config(page_title="Financial Addiction Detector", layout="wide")

# Initialize models
@st.cache_resource
def load_backend():
    cache = get_ml_cache()
    cache.initialize()
    return cache

cache = load_backend()

st.title("Financial Addiction Detection Dashboard")

tab1, tab2, tab3 = st.tabs(["User Risk Profile", "Intervention Panel", "Regulator Dashboard"])

# --- TAB 1: USER RISK PROFILE ---
with tab1:
    st.header("User Risk Profile")
    
    # Get user list
    user_ids = sorted(cache.labels.index.tolist())
    selected_user = st.selectbox("Select User ID to Analyze:", user_ids[:100]) # Limit to 100 for simplicity
    
    if selected_user:
        # Get proxy score
        proxy_score = int(cache.labels.loc[selected_user, 'total_score'])
        
        # Determine state
        if proxy_score <= 2: 
            state, color = "Casual", "green"
            risk_tier_id = 0
        elif proxy_score <= 5: 
            state, color = "Frequent", "orange"
            risk_tier_id = 1
        elif proxy_score <= 7: 
            state, color = "Compulsive", "red"
            risk_tier_id = 2
        else: 
            state, color = "Crisis", "darkred"
            risk_tier_id = 3
            
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"### Current State: <span style='color:{color}'>{state}</span>", unsafe_allow_html=True)
            
            # Risk Gauge
            gauge_fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = proxy_score * 10, # scale to 100
                title = {'text': "Addiction Risk Index"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 29], 'color': "lightgreen"},
                        {'range': [30, 59], 'color': "wheat"},
                        {'range': [60, 79], 'color': "salmon"},
                        {'range': [80, 100], 'color': "darkred"}
                    ],
                }
            ))
            st.plotly_chart(gauge_fig, use_container_width=True)
            
        with col2:
            st.subheader("Transaction Behavior Heatmap")
            # Get user transactions
            user_txns = cache.transactions[cache.transactions['user_id'] == selected_user].copy()
            if not user_txns.empty:
                user_txns['day'] = user_txns['timestamp'].dt.day_name()
                user_txns['hour'] = user_txns['timestamp'].dt.hour
                
                # Order days
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                heatmap_data = user_txns.groupby(['day', 'hour']).size().reset_index(name='count')
                heatmap_pivot = heatmap_data.pivot(index='day', columns='hour', values='count').reindex(days).fillna(0)
                
                fig = px.imshow(heatmap_pivot,
                                labels=dict(x="Hour of Day", y="Day of Week", color="Txn Count"),
                                x=heatmap_pivot.columns,
                                y=heatmap_pivot.index,
                                color_continuous_scale='Purples')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No transactions available for heatmap.")
                
        # SHAP EXPLANATION
        st.subheader("AI Explainability (SHAP)")
        features = cache.tx_agg.loc[selected_user][cache.feature_names].values.reshape(1, -1)
        scaled_features = cache.scaler.transform(features)
        tensor_features = torch.tensor(scaled_features, dtype=torch.float32)
        
        shap_values = cache.explainer.shap_values(tensor_features)
        
        with torch.no_grad():
            predicted_class = torch.argmax(cache.model(tensor_features), dim=1).item()
            
        # Extract SV correctly (shape is usually samples, features, classes in DP-SGD)
        if isinstance(shap_values, list):
            sv = shap_values[predicted_class][0]
        else:
            if len(shap_values.shape) == 3:
                sv = shap_values[0, :, predicted_class]
            else:
                sv = shap_values[0]
                
        top_idx = np.argsort(np.abs(sv))[::-1][:3]
        
        st.write("Based on the underlying DP-SGD neural network, here are the top 3 behavioral factors influencing this user's risk score:")
        for idx in top_idx:
            feat = cache.feature_names[idx]
            impact = sv[idx]
            direction = "increasing" if impact > 0 else "decreasing"
            arrow = "🔴" if impact > 0 else "🟢"
            st.info(f"{arrow} The feature **{feat}** is significantly **{direction}** the user's risk profile (impact: {impact:+.4f}).")

# --- TAB 2: INTERVENTION PANEL ---
with tab2:
    st.header("Intervention Management")
    if selected_user:
        st.write(f"Currently managing: **User {selected_user}** (State: {state})")
        
        if risk_tier_id >= 2:
            st.error("🚨 **CRITICAL ALERT:** User has entered a high-risk addiction state (Compulsive/Crisis). Mandatory intervention review required.")
        else:
            st.success("User is currently in a safe or monitored state. No immediate intervention required.")
            
        st.markdown("---")
        st.subheader("DPDP Act Compliance & Consent Notice")
        st.markdown("""
        > Under the **Digital Personal Data Protection (DPDP) Act**, all behavioral interventions and risk categorizations must be transparent and strictly necessary for the purpose of protecting user financial well-being.
        > The user has explicitly consented to behavioral risk monitoring under Section 6 of the Terms of Service. If an intervention limits account functionality, the user must be notified immediately with an option to appeal.
        """)
        
        st.subheader("Active Interventions")
        cooling_off = st.toggle("Enable 24-Hour Cooling-Off Period (Lock Account)")
        if cooling_off:
            st.warning("Cooling-off period active. User will be restricted from making new deposits or bets for 24 hours.")
    else:
        st.write("Please select a user in Tab 1.")
        
# --- TAB 3: REGULATOR DASHBOARD ---
with tab3:
    st.header("Regulator Oversight Leaderboard")
    st.write("Aggregated platform telemetry assessing potential exploitative mechanics (Dark Patterns) across the fintech ecosystem.")
    st.markdown("> *Note: Correlation computed via Pearson's r on simulated clickstream data (synthetic demonstration, not real platform data).*")
    
    from models.dark_pattern_scorer import generate_report
    df_leaderboard = generate_report(data_dir='data')
    
    st.subheader("Platform Risk Leaderboard")
    st.dataframe(df_leaderboard, use_container_width=True)
    
    st.subheader("Correlation: Nudges vs. Escalation")
    # Plotly combo chart
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=df_leaderboard["Platform"],
        y=df_leaderboard["Avg Post-Loss Notif Rate (%)"],
        name="Post-Loss Notif Rate",
        marker_color="royalblue"
    ))
    fig2.add_trace(go.Scatter(
        x=df_leaderboard["Platform"],
        y=df_leaderboard["Avg Escalation Count"],
        name="Escalation Count",
        yaxis="y2",
        mode="lines+markers",
        marker=dict(size=10, color="red")
    ))
    fig2.update_layout(
        yaxis=dict(title="Avg Post-Loss Notif Rate (%)"),
        yaxis2=dict(title="Avg Escalation Count", overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig2, use_container_width=True)
