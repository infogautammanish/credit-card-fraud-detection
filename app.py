# ============================================
# app.py - Credit Card Fraud Detection Dashboard
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Load Data and Models
# ============================================

@st.cache_resource
def load_data():
    df = pd.read_csv('credit_card_fraud_dataset.csv')
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    
    # Extract time features
    df['Year'] = df['TransactionDate'].dt.year
    df['Month'] = df['TransactionDate'].dt.month
    df['Day'] = df['TransactionDate'].dt.day
    df['Hour'] = df['TransactionDate'].dt.hour
    df['DayOfWeek'] = df['TransactionDate'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    return df

@st.cache_resource
def load_models():
    try:
        model = joblib.load('models/best_fraud_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        le_merchant = joblib.load('models/le_merchant.pkl')
        le_location = joblib.load('models/le_location.pkl')
        le_type = joblib.load('models/le_type.pkl')
        le_category = joblib.load('models/le_category.pkl')
        return model, scaler, le_merchant, le_location, le_type, le_category
    except:
        return None, None, None, None, None, None

# Load data and models
df = load_data()
model, scaler, le_merchant, le_location, le_type, le_category = load_models()

# ============================================
# Sidebar Navigation
# ============================================

st.sidebar.title("💳 Fraud Detection System")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📋 Navigation",
    ["📊 Fraud Overview", "🔮 Fraud Prediction", "🚨 Anomaly Report", "📈 Risk Intelligence"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **ℹ️ About**
    This system uses Machine Learning to detect credit card fraud.
    
    **Model:** XGBoost
    **Threshold:** 0.50
    """
)

# ============================================
# PAGE 1: FRAUD OVERVIEW DASHBOARD
# ============================================

if page == "📊 Fraud Overview":
    st.title("📊 Fraud Overview Dashboard")
    st.markdown("---")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        locations = ['All'] + sorted(df['Location'].unique().tolist())
        selected_location = st.selectbox("📍 Location", locations)
    with col2:
        types = ['All'] + sorted(df['TransactionType'].unique().tolist())
        selected_type = st.selectbox("🏷️ Transaction Type", types)
    with col3:
        fraud_status = st.selectbox("🎯 Fraud Status", ['All', 'Legitimate Only', 'Fraud Only'])
    with col4:
        min_amount = st.slider("💰 Min Amount", 0, 5000, 0, step=100)
    
    # Filter data
    filtered_df = df.copy()
    if selected_location != 'All':
        filtered_df = filtered_df[filtered_df['Location'] == selected_location]
    if selected_type != 'All':
        filtered_df = filtered_df[filtered_df['TransactionType'] == selected_type]
    if fraud_status == 'Legitimate Only':
        filtered_df = filtered_df[filtered_df['IsFraud'] == 0]
    elif fraud_status == 'Fraud Only':
        filtered_df = filtered_df[filtered_df['IsFraud'] == 1]
    if min_amount > 0:
        filtered_df = filtered_df[filtered_df['Amount'] >= min_amount]
    
    # KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_trans = len(filtered_df)
    total_amount = filtered_df['Amount'].sum()
    fraud_count = filtered_df['IsFraud'].sum()
    fraud_rate = (fraud_count / total_trans * 100) if total_trans > 0 else 0
    avg_amount = filtered_df['Amount'].mean() if total_trans > 0 else 0
    
    with col1:
        st.metric("📊 Total Transactions", f"{total_trans:,}")
    with col2:
        st.metric("💰 Total Amount", f"${total_amount:,.2f}")
    with col3:
        st.metric("🚨 Fraud Count", f"{fraud_count:,}")
    with col4:
        st.metric("📈 Fraud Rate", f"{fraud_rate:.2f}%")
    with col5:
        st.metric("💵 Avg Amount", f"${avg_amount:,.2f}")
    
    st.markdown("---")
    
    # Row 1: Two Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Fraud Distribution
        fraud_dist = filtered_df['IsFraud'].value_counts().reset_index()
        fraud_dist.columns = ['Status', 'Count']
        fraud_dist['Status'] = fraud_dist['Status'].map({0: 'Legitimate', 1: 'Fraudulent'})
        
        fig1 = px.pie(fraud_dist, values='Count', names='Status', 
                      title='Transaction Distribution',
                      color='Status', color_discrete_map={'Legitimate': '#2ecc71', 'Fraudulent': '#e74c3c'})
        fig1.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Fraud by Location
        loc_fraud = filtered_df.groupby('Location').agg(
            total=('TransactionID', 'count'),
            fraud=('IsFraud', 'sum')
        ).reset_index()
        loc_fraud['fraud_rate'] = (loc_fraud['fraud'] / loc_fraud['total']) * 100
        loc_fraud = loc_fraud.sort_values('fraud_rate', ascending=False).head(10)
        
        fig2 = px.bar(loc_fraud, x='Location', y='fraud_rate',
                      title='Fraud Rate by Location',
                      color='fraud_rate', color_continuous_scale='Reds',
                      labels={'fraud_rate': 'Fraud Rate (%)'})
        st.plotly_chart(fig2, use_container_width=True)
    
    # Row 2: Two Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Fraud by Transaction Type
        type_fraud = filtered_df.groupby('TransactionType').agg(
            total=('TransactionID', 'count'),
            fraud=('IsFraud', 'sum')
        ).reset_index()
        type_fraud['fraud_rate'] = (type_fraud['fraud'] / type_fraud['total']) * 100
        
        fig3 = px.bar(type_fraud, x='TransactionType', y='fraud_rate',
                      title='Fraud Rate by Transaction Type',
                      color='fraud_rate', color_continuous_scale='Oranges',
                      labels={'fraud_rate': 'Fraud Rate (%)'})
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Fraud by Hour
        hourly_fraud = filtered_df.groupby('Hour').agg(
            total=('TransactionID', 'count'),
            fraud=('IsFraud', 'sum')
        ).reset_index()
        
        fig4 = px.line(hourly_fraud, x='Hour', y='fraud',
                       title='Fraudulent Transactions by Hour',
                       markers=True)
        fig4.update_traces(line_color='#2c3e50', marker_color='#e74c3c', marker_size=8)
        st.plotly_chart(fig4, use_container_width=True)
    
    # Row 3: Monthly Trend
    monthly_fraud = filtered_df.groupby('Month').agg(
        total=('TransactionID', 'count'),
        fraud=('IsFraud', 'sum')
    ).reset_index()
    monthly_fraud['fraud_rate'] = (monthly_fraud['fraud'] / monthly_fraud['total']) * 100
    
    fig5 = px.bar(monthly_fraud, x='Month', y='fraud_rate',
                  title='Monthly Fraud Rate Trend',
                  color='fraud_rate', color_continuous_scale='Purples',
                  labels={'fraud_rate': 'Fraud Rate (%)'})
    st.plotly_chart(fig5, use_container_width=True)
    
    # Recent Fraudulent Transactions
    st.markdown("---")
    st.subheader("🔍 Recent Suspicious Transactions")
    
    recent_fraud = filtered_df[filtered_df['IsFraud'] == 1].sort_values('TransactionDate', ascending=False).head(20)
    if len(recent_fraud) > 0:
        display_cols = ['TransactionID', 'TransactionDate', 'Amount', 'MerchantID', 'TransactionType', 'Location']
        st.dataframe(recent_fraud[display_cols].style.background_gradient(subset=['Amount'], cmap='Reds'), use_container_width=True)
    else:
        st.info("No fraudulent transactions found with current filters.")

# ============================================
# PAGE 2: FRAUD PREDICTION
# ============================================

elif page == "🔮 Fraud Prediction":
    st.title("🔮 Fraud Prediction Explorer")
    st.markdown("---")
    
    if model is None:
        st.error("❌ Model not found! Please train the model first.")
    else:
        st.info("Enter transaction details below to get a fraud risk assessment.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input("💰 Transaction Amount ($)", min_value=0.0, max_value=10000.0, value=100.0, step=10.0)
            merchant = st.selectbox("🏪 Merchant ID", sorted(df['MerchantID'].unique()))
            transaction_type = st.selectbox("📝 Transaction Type", sorted(df['TransactionType'].unique()))
            
        with col2:
            location = st.selectbox("📍 Location", sorted(df['Location'].unique()))
            hour = st.slider("🕐 Hour of Day", 0, 23, 14)
            day = st.selectbox("📅 Day of Month", list(range(1, 32)))
            month = st.selectbox("📆 Month", list(range(1, 13)))
        
        # Calculate derived features
        merchant_data = df[df['MerchantID'] == merchant]
        merchant_transactions = len(merchant_data)
        merchant_avg_amount = merchant_data['Amount'].mean() if merchant_transactions > 0 else 0
        merchant_fraud_rate = merchant_data['IsFraud'].mean() if merchant_transactions > 0 else 0
        
        location_data = df[df['Location'] == location]
        location_transactions = len(location_data)
        location_avg_amount = location_data['Amount'].mean() if location_transactions > 0 else 0
        location_fraud_rate = location_data['IsFraud'].mean() if location_transactions > 0 else 0
        
        type_data = df[df['TransactionType'] == transaction_type]
        type_transactions = len(type_data)
        type_fraud_rate = type_data['IsFraud'].mean() if type_transactions > 0 else 0
        
        # Prepare features for prediction
        amount_log = np.log1p(amount)
        amount_zscore = (amount - df['Amount'].mean()) / df['Amount'].std() if df['Amount'].std() > 0 else 0
        
        # Encode categoricals
        try:
            merchant_enc = le_merchant.transform([merchant])[0]
        except:
            merchant_enc = 0
        try:
            location_enc = le_location.transform([location])[0]
        except:
            location_enc = 0
        try:
            type_enc = le_type.transform([transaction_type])[0]
        except:
            type_enc = 0
        
        # Category encoding
        if amount <= 100:
            category_enc = 0
        elif amount <= 500:
            category_enc = 1
        elif amount <= 1000:
            category_enc = 2
        else:
            category_enc = 3
        
        # Feature vector
        features = np.array([[
            amount, hour, day, month, 0, 0, 0,  # IsWeekend, IsNight will be derived
            merchant_transactions, merchant_avg_amount, merchant_fraud_rate,
            location_transactions, location_avg_amount, location_fraud_rate,
            type_transactions, type_fraud_rate, amount_log, amount_zscore,
            merchant_enc, location_enc, type_enc, category_enc
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict
        if st.button("🔍 Assess Fraud Risk", type="primary"):
            prob = model.predict_proba(features_scaled)[0][1]
            prediction = 1 if prob >= 0.5 else 0
            
            st.markdown("---")
            st.subheader("📊 Risk Assessment Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.error(f"🚨 FRAUD ALERT")
                else:
                    st.success("✅ Legitimate Transaction")
            
            with col2:
                st.metric("Fraud Probability", f"{prob*100:.1f}%")
            
            with col3:
                if prob >= 0.7:
                    risk_level = "🔴 HIGH RISK"
                    color = "#e74c3c"
                elif prob >= 0.4:
                    risk_level = "🟡 MEDIUM RISK"
                    color = "#f39c12"
                else:
                    risk_level = "🟢 LOW RISK"
                    color = "#2ecc71"
                st.markdown(f"### {risk_level}")
            
            # Risk meter
            st.markdown("---")
            st.subheader("Risk Meter")
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                title = {'text': "Fraud Risk Score"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#e74c3c"},
                    'steps': [
                        {'range': [0, 30], 'color': "#2ecc71"},
                        {'range': [30, 60], 'color': "#f1c40f"},
                        {'range': [60, 100], 'color': "#e74c3c"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': prob * 100
                    }
                }
            ))
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommended Action
            st.markdown("---")
            st.subheader("💡 Recommended Action")
            
            if prob >= 0.7:
                st.error("""
                🚨 **Immediate Action Required**
                
                - Block this transaction
                - Send for manual review
                - Notify fraud investigation team
                - Contact cardholder for verification
                """)
            elif prob >= 0.4:
                st.warning("""
                ⚠️ **Additional Verification Recommended**
                
                - Request OTP verification
                - Flag for review
                - Monitor for unusual patterns
                - Consider temporary hold
                """)
            else:
                st.success("""
                ✅ **Normal Processing**
                
                - Process transaction normally
                - No additional verification needed
                - Continue standard monitoring
                """)

# ============================================
# PAGE 3: ANOMALY REPORT
# ============================================

elif page == "🚨 Anomaly Report":
    st.title("🚨 Anomaly Detection Report")
    st.markdown("---")
    
    # Apply Isolation Forest if not already done
    if 'Anomaly_IF' not in df.columns:
        from sklearn.ensemble import IsolationForest
        
        anomaly_features = ['Amount', 'Hour']
        X_anomaly = df[anomaly_features].fillna(0)
        
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        df['Anomaly_IF'] = iso_forest.fit_predict(X_anomaly)
        df['Anomaly_IF'] = df['Anomaly_IF'].map({1: 0, -1: 1})
        
        # Z-Score anomaly
        df['AmountZScore'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
        df['Anomaly_ZScore'] = (np.abs(df['AmountZScore']) > 2).astype(int)
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        anomaly_filter = st.selectbox("🔍 Filter", ['All', 'Isolation Forest Anomalies', 'Z-Score Anomalies', 'Fraud Only'])
    
    with col2:
        min_amount = st.number_input("💰 Minimum Amount", min_value=0, value=0, step=50)
    
    with col3:
        location_filter = st.selectbox("📍 Location", ['All'] + sorted(df['Location'].unique()))
    
    # Apply filters
    filtered_df = df.copy()
    
    if anomaly_filter == 'Isolation Forest Anomalies':
        filtered_df = filtered_df[filtered_df['Anomaly_IF'] == 1]
    elif anomaly_filter == 'Z-Score Anomalies':
        filtered_df = filtered_df[filtered_df['Anomaly_ZScore'] == 1]
    elif anomaly_filter == 'Fraud Only':
        filtered_df = filtered_df[filtered_df['IsFraud'] == 1]
    
    if min_amount > 0:
        filtered_df = filtered_df[filtered_df['Amount'] >= min_amount]
    
    if location_filter != 'All':
        filtered_df = filtered_df[filtered_df['Location'] == location_filter]
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Anomalies", len(filtered_df))
    with col2:
        fraud_in_anomalies = filtered_df['IsFraud'].sum()
        st.metric("Fraud in Anomalies", fraud_in_anomalies)
    with col3:
        anomaly_fraud_rate = (fraud_in_anomalies / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        st.metric("Anomaly Fraud Rate", f"{anomaly_fraud_rate:.1f}%")
    with col4:
        avg_amount_anomaly = filtered_df['Amount'].mean() if len(filtered_df) > 0 else 0
        st.metric("Avg Amount (Anomalies)", f"${avg_amount_anomaly:,.2f}")
    
    st.markdown("---")
    
    # Anomaly Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        # Anomaly method comparison
        comparison = df.groupby(['Anomaly_IF', 'Anomaly_ZScore']).size().reset_index(name='Count')
        comparison['Method'] = comparison.apply(
            lambda x: 'Both' if (x['Anomaly_IF'] == 1 and x['Anomaly_ZScore'] == 1) else
                     ('Only IF' if x['Anomaly_IF'] == 1 else
                      ('Only Z-Score' if x['Anomaly_ZScore'] == 1 else 'None')),
            axis=1
        )
        comparison = comparison[comparison['Method'] != 'None']
        
        fig1 = px.pie(comparison, values='Count', names='Method',
                      title='Anomaly Detection Method Comparison')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Amount distribution of anomalies
        fig2 = px.histogram(filtered_df, x='Amount', color='IsFraud',
                            title='Amount Distribution in Anomalies',
                            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                            labels={'IsFraud': 'Fraud Status'})
        fig2.update_layout(barmode='overlay')
        fig2.update_traces(opacity=0.7)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Anomaly Table
    st.markdown("---")
    st.subheader("📋 Anomalous Transactions")
    
    display_cols = ['TransactionID', 'TransactionDate', 'Amount', 'MerchantID', 'TransactionType', 'Location', 'IsFraud', 'Anomaly_IF', 'Anomaly_ZScore']
    
    if len(filtered_df) > 0:
        table_df = filtered_df[display_cols].sort_values('Amount', ascending=False).head(100)
        
        # Color formatting
        styled_df = table_df.style.background_gradient(subset=['Amount'], cmap='Reds')
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.info("No anomalies found with current filters.")

# ============================================
# PAGE 4: RISK INTELLIGENCE
# ============================================

else:
    st.title("📈 Risk Intelligence Dashboard")
    st.markdown("---")
    
    st.info("This page provides comprehensive risk analysis at merchant, location, and segment levels.")
    
    # Risk Tabs
    tab1, tab2, tab3 = st.tabs(["🏪 Merchant Risk", "🌎 Location Risk", "👥 Segment Analysis"])
    
    with tab1:
        st.subheader("🏪 Merchant Risk Analysis")
        
        # Merchant risk table
        merchant_risk = df.groupby('MerchantID').agg(
            transactions=('TransactionID', 'count'),
            fraud_count=('IsFraud', 'sum'),
            avg_amount=('Amount', 'mean'),
            total_amount=('Amount', 'sum')
        ).reset_index()
        merchant_risk['fraud_rate'] = (merchant_risk['fraud_count'] / merchant_risk['transactions']) * 100
        
        def assign_risk(row):
            if row['transactions'] < 10:
                return 'Insufficient Data'
            elif row['fraud_rate'] > 15:
                return '🔴 Critical Risk'
            elif row['fraud_rate'] > 10:
                return '🟠 High Risk'
            elif row['fraud_rate'] > 5:
                return '🟡 Medium Risk'
            else:
                return '🟢 Low Risk'
        
        merchant_risk['risk_level'] = merchant_risk.apply(assign_risk, axis=1)
        
        # Risk distribution
        col1, col2 = st.columns([2, 1])
        
        with col1:
            risk_dist = merchant_risk[merchant_risk['transactions'] >= 10]['risk_level'].value_counts().reset_index()
            risk_dist.columns = ['Risk Level', 'Count']
            
            fig1 = px.bar(risk_dist, x='Risk Level', y='Count',
                          title='Merchant Risk Distribution',
                          color='Risk Level')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.metric("Total Merchants", len(merchant_risk))
            st.metric("High/Critical Risk Merchants", 
                     len(merchant_risk[merchant_risk['risk_level'].isin(['🟠 High Risk', '🔴 Critical Risk'])]))
        
        # Top risk merchants
        st.markdown("---")
        st.subheader("🔴 Top 20 High-Risk Merchants")
        
        high_risk = merchant_risk[merchant_risk['transactions'] >= 10].sort_values('fraud_rate', ascending=False).head(20)
        st.dataframe(high_risk[['MerchantID', 'transactions', 'fraud_count', 'fraud_rate', 'risk_level']].style.background_gradient(subset=['fraud_rate'], cmap='Reds'), use_container_width=True)
    
    with tab2:
        st.subheader("🌎 Location Risk Analysis")
        
        # Location risk
        location_risk = df.groupby('Location').agg(
            transactions=('TransactionID', 'count'),
            fraud_count=('IsFraud', 'sum'),
            avg_amount=('Amount', 'mean'),
            total_amount=('Amount', 'sum')
        ).reset_index()
        location_risk['fraud_rate'] = (location_risk['fraud_count'] / location_risk['transactions']) * 100
        location_risk = location_risk.sort_values('fraud_rate', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig2 = px.bar(location_risk, x='Location', y='fraud_rate',
                          title='Fraud Rate by Location',
                          color='fraud_rate', color_continuous_scale='Reds',
                          labels={'fraud_rate': 'Fraud Rate (%)'})
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            fig3 = px.scatter(location_risk, x='transactions', y='fraud_rate',
                              title='Location Risk: Volume vs Fraud Rate',
                              hover_data=['Location'],
                              color='fraud_rate', color_continuous_scale='Reds',
                              labels={'transactions': 'Transaction Volume', 'fraud_rate': 'Fraud Rate (%)'})
            st.plotly_chart(fig3, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📋 Location Risk Summary")
        st.dataframe(location_risk[['Location', 'transactions', 'fraud_count', 'fraud_rate', 'avg_amount']].style.background_gradient(subset=['fraud_rate'], cmap='Reds'), use_container_width=True)
    
    with tab3:
        st.subheader("👥 Transaction Segment Analysis")
        
        # Apply clustering if not already done
        if 'Cluster' not in df.columns:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            cluster_features = ['Amount', 'Hour']
            X_cluster = df[cluster_features].fillna(0)
            scaler_cluster = StandardScaler()
            X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)
            
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            df['Cluster'] = kmeans.fit_predict(X_cluster_scaled)
        
        # Cluster analysis
        cluster_analysis = df.groupby('Cluster').agg(
            count=('TransactionID', 'count'),
            avg_amount=('Amount', 'mean'),
            avg_hour=('Hour', 'mean'),
            fraud_rate=('IsFraud', 'mean')
        ).reset_index()
        cluster_analysis['fraud_rate'] = cluster_analysis['fraud_rate'] * 100
        
        # Cluster visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig4 = px.scatter(df, x='Amount', y='Hour', color='Cluster',
                              title='Transaction Clusters',
                              color_continuous_scale='viridis')
            fig4.update_traces(marker=dict(size=5, opacity=0.6))
            st.plotly_chart(fig4, use_container_width=True)
        
        with col2:
            cluster_summary = cluster_analysis.copy()
            cluster_summary['avg_amount'] = cluster_summary['avg_amount'].round(2)
            cluster_summary['fraud_rate'] = cluster_summary['fraud_rate'].round(2)
            st.dataframe(cluster_summary, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📊 Cluster Characteristics")
        
        # Define cluster labels
        def label_cluster(row):
            if row['avg_amount'] > 500 and row['fraud_rate'] > 2:
                return 'High-Value, High-Risk'
            elif row['avg_amount'] > 500:
                return 'High-Value, Low-Risk'
            elif row['fraud_rate'] > 2:
                return 'Low-Value, High-Risk'
            else:
                return 'Low-Value, Low-Risk'
        
        cluster_analysis['business_label'] = cluster_analysis.apply(label_cluster, axis=1)
        
        for _, row in cluster_analysis.iterrows():
            with st.expander(f"Cluster {row['Cluster']}: {row['business_label']}"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Transactions", f"{row['count']:,}")
                with col2:
                    st.metric("Avg Amount", f"${row['avg_amount']:.2f}")
                with col3:
                    st.metric("Avg Hour", f"{row['avg_hour']:.1f}:00")
                with col4:
                    st.metric("Fraud Rate", f"{row['fraud_rate']:.1f}%")
                
                if row['fraud_rate'] > 5:
                    st.warning("⚠️ This cluster has elevated fraud risk. Recommended: Enhanced monitoring.")
                elif row['avg_amount'] > 500:
                    st.info("💡 This cluster has high transaction values. Recommended: Standard monitoring with higher scrutiny.")
                else:
                    st.success("✅ This cluster shows normal behavior. Recommended: Standard monitoring.")

# ============================================
# Footer
# ============================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>💳 Credit Card Fraud Detection System | Built with Streamlit & Machine Learning</p>
        <p style='font-size: 12px;'>Data Science Internship Project</p>
    </div>
    """,
    unsafe_allow_html=True
)