# =====================================================
# NexGen Predictive Delivery Dashboard — Final Integrated Version
# =====================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import os
import ml_predictor as ml_predictor  # ✅ ML Module Integration

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(
    page_title="NexGen Predictive Delivery Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------
# CUSTOM THEME (Dark, Elegant, Classy)
# ------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

.stApp {
    background: radial-gradient(circle at top left, #0b0c10, #1f2833);
    color: #C5C6C7;
    font-family: 'Poppins', sans-serif;
}
h1, h2, h3, h4 {
    color: #66FCF1 !important;
    font-weight: 600;
}
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #1f2833, #0b0c10);
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0px 0px 15px rgba(102, 252, 241, 0.15);
    transition: all 0.3s ease;
}
div[data-testid="stMetric"]:hover {
    transform: scale(1.03);
    box-shadow: 0px 0px 20px rgba(102, 252, 241, 0.4);
}
.stTabs [data-baseweb="tab"] {
    background-color: #1f2833;
    color: #C5C6C7;
    border-radius: 10px;
    padding: 10px 25px;
    transition: all 0.3s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #45A29E, #66FCF1);
    color: black !important;
    font-weight: bold;
    box-shadow: 0 0 10px rgba(102,252,241,0.5);
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# TITLE
# ------------------------------------------------------
st.title("🚚 NexGen Predictive Delivery Dashboard")
st.markdown("#### Intelligent Supply Chain Insights & Customer Feedback Analysis")

# ------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------
@st.cache_data
def load_data():
    base = os.path.join(os.getcwd(), "data")
    if not os.path.exists(base):
        st.error(f"❌ Data folder not found: {base}")
        st.stop()
    files = {
        "cost": "cost_breakdown.csv",
        "feedback": "customer_feedback.csv",
        "delivery": "delivery_performance.csv",
        "orders": "orders.csv",
        "routes": "routes_distance.csv",
        "vehicles": "vehicle_fleet.csv",
        "warehouse": "warehouse_inventory.csv"
    }
    missing = [f for f in files.values() if not os.path.exists(os.path.join(base, f))]
    if missing:
        st.error(f"❌ Missing files in data folder: {missing}")
        st.stop()
    return {name: pd.read_csv(os.path.join(base, fname)) for name, fname in files.items()}

# ------------------------------------------------------
# LOAD DATASETS
# ------------------------------------------------------
data = load_data()
cost, feedback, delivery, orders, routes, vehicles, warehouse = (
    data["cost"], data["feedback"], data["delivery"], data["orders"],
    data["routes"], data["vehicles"], data["warehouse"]
)

# ------------------------------------------------------
# SIDEBAR OVERVIEW
# ------------------------------------------------------
st.sidebar.header("📊 Dataset Overview")
for name, df in data.items():
    st.sidebar.write(f"**{name.capitalize()}** — {df.shape[0]} rows, {df.shape[1]} cols")

# ------------------------------------------------------
# KPI SECTION
# ------------------------------------------------------
st.subheader("📈 Key Performance Indicators")
col1, col2, col3 = st.columns(3)
col1.metric("Total Orders", f"{orders['Order_ID'].nunique()}")
col2.metric("Average Rating", f"{feedback['Rating'].mean():.2f} ⭐")
if "Delivery_Time" in delivery.columns:
    col3.metric("Avg Delivery Time", f"{delivery['Delivery_Time'].mean():.1f} hrs")
else:
    col3.metric("Avg Delivery Time", "N/A")

st.markdown("---")

# ------------------------------------------------------
# VISUALIZATION TABS
# ------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📦 Orders", "🚚 Delivery", "💬 Feedback", "💰 Cost", "🧠 ML Predictor"
])

# =====================================================
# ORDERS TAB
# =====================================================
with tab1:
    st.subheader("📦 Orders Overview")
    st.dataframe(orders.head(), use_container_width=True)
    if "Order_Date" in orders.columns:
        orders_over_time = orders.groupby('Order_Date').size().reset_index(name='Order_Count')
        orders_over_time = orders_over_time.sort_values('Order_Date')
        fig_orders = px.line(
            orders_over_time, x='Order_Date', y='Order_Count',
            title='📅 Orders Over Time', markers=True,
            color_discrete_sequence=['#45A29E']
        )
        fig_orders.update_traces(line=dict(width=3))
        fig_orders.update_layout(
            title_font=dict(size=22, color='#66FCF1'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_orders, use_container_width=True)

# =====================================================
# DELIVERY TAB
# =====================================================
with tab2:
    st.subheader("🚚 Delivery Performance")
    st.dataframe(delivery.head(), use_container_width=True)
    if "Delivery_Status" in delivery.columns:
        fig_delivery = px.pie(
            delivery, names="Delivery_Status",
            title="📊 Delivery Status Distribution", hole=0.45,
            color_discrete_sequence=px.colors.sequential.Teal
        )
        fig_delivery.update_layout(
            title_font=dict(size=22, color='#66FCF1'),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_delivery, use_container_width=True)
    if "Delivery_Time" in delivery.columns:
        if "Route_ID" in delivery.columns:
            avg_time = delivery.groupby("Route_ID")["Delivery_Time"].mean().reset_index()
            x_col = "Route_ID"
        elif "Vehicle_ID" in delivery.columns:
            avg_time = delivery.groupby("Vehicle_ID")["Delivery_Time"].mean().reset_index()
            x_col = "Vehicle_ID"
        else:
            avg_time = None
        if avg_time is not None:
            fig_time = px.bar(
                avg_time, x=x_col, y="Delivery_Time",
                title=f"⏱️ Average Delivery Time by {x_col.replace('_', ' ')}",
                color_discrete_sequence=['#45A29E']
            )
            fig_time.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font=dict(size=22, color='#66FCF1'),
                font=dict(color='white')
            )
            st.plotly_chart(fig_time, use_container_width=True)

# =====================================================
# FEEDBACK TAB
# =====================================================
with tab3:
    st.subheader("💬 Customer Feedback Insights")
    st.dataframe(feedback.head(), use_container_width=True)
    if "Rating" in feedback.columns:
        fig_ratings = px.bar(
            feedback, x="Rating", title="⭐ Customer Rating Distribution",
            color_discrete_sequence=['#45A29E']
        )
        fig_ratings.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font=dict(size=22, color='#66FCF1'),
            font=dict(color='white')
        )
        st.plotly_chart(fig_ratings, use_container_width=True)
    if "Issue_Category" in feedback.columns:
        feedback_clean = feedback.copy()
        feedback_clean["Issue_Category"] = feedback_clean["Issue_Category"].fillna("Unknown")
        feedback_clean = feedback_clean[feedback_clean["Issue_Category"].str.strip() != ""]
        if not feedback_clean.empty:
            fig_issue = px.pie(
                feedback_clean, names="Issue_Category",
                title="🧩 Feedback Issues Breakdown", hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_issue.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                title_font=dict(size=22, color='#66FCF1')
            )
            st.plotly_chart(fig_issue, use_container_width=True)

# =====================================================
# COST TAB
# =====================================================
with tab4:
    st.subheader("💰 Cost Breakdown")
    st.dataframe(cost.head(), use_container_width=True)
    if "Cost_Type" in cost.columns and "Amount" in cost.columns:
        fig_cost = px.pie(
            cost, names="Cost_Type", values="Amount",
            title="💸 Operational Cost Distribution",
            hole=0.45, color_discrete_sequence=px.colors.sequential.Purples
        )
        fig_cost.update_layout(
            title_font=dict(size=22, color='#66FCF1'),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_cost, use_container_width=True)

# =====================================================
# ML PREDICTOR TAB — Integrated from ml_predictor_final.py
# =====================================================
with tab5:
    ml_predictor.main()

st.success("✅ Dashboard loaded successfully")
