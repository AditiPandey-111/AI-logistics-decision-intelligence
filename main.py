import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ----------------------------------
# Load saved assets
# ----------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed_orders_full.csv")
    return df

@st.cache_resource
def load_model():
    model = joblib.load("models/delay_predictor.pkl")
    features = joblib.load("models/features.pkl")
    return model, features


# ----------------------------------
# APP LAYOUT
# ----------------------------------

st.title("🚚 NexGen Logistics – Delay Root Cause & Cost Impact Analyzer")

df = load_data()
model, feature_cols = load_model()


# ======================================================================================
#  SIDEBAR NAVIGATION
# ======================================================================================

page = st.sidebar.radio(
    "📌 Select Dashboard Module",
    [
        "📈 Delay Analytics Dashboard",
        "🔮 Predict Delay Risk",
        "🛠 Root Cause Analyzer",
        "⚡ Cost Impact Simulator",
    ]
)


# ======================================================================================
#  PAGE 1 – DELAY INSIGHTS DASHBOARD
# ======================================================================================

if page == "📈 Delay Analytics Dashboard":

    st.header("📊 Operational Performance Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Orders", len(df))

    with col2:
        st.metric("Avg. Transport Cost (INR)", round(df["Transport_Cost"].mean(),2))

    # delay distribution
    fig = px.histogram(df, x="Delay_Days", title="Distribution of Delivery Delay")
    st.plotly_chart(fig)

    # cost vs delay relationship
    fig2 = px.scatter(
        df,
        x="Transport_Cost",
        y="Delay_Days",
        color="Cost_Anomaly_Flag",
        title="Cost–Delay Relationship"
    )
    st.plotly_chart(fig2)

    # route efficiency
    fig3 = px.scatter(
        df,
        x="Distance_KM",
        y="Route_Efficiency_Index",
        color="Vehicle_Age",
        title="Route Efficiency vs Distance"
    )
    st.plotly_chart(fig3)

    st.write("---")


# ======================================================================================
#  PAGE 2 – PREDICT DELAY RISK
# ======================================================================================

elif page == "🔮 Predict Delay Risk":

    st.header("Predict Delay Probability for New Shipment")

    inp = {}

    for f in feature_cols:
        inp[f] = st.number_input(f, min_value=0.0)

    if st.button("Predict Delay"):

        values = np.array([list(inp.values())])

        pred = model.predict(values)[0]
        prob = model.predict_proba(values)[0][1]

        st.success(f"Predicted Delay Risk: {prob*100:.2f}%")

        if pred == 1:
            st.error("⚠ Shipment likely to be delayed.")
        else:
            st.success("✔ Shipment likely on time.")


# ======================================================================================
#  PAGE 3 – ROOT CAUSE ANALYZER
# ======================================================================================

elif page == "🛠 Root Cause Analyzer":

    st.header("Identify Drivers Behind Delay + Costs")

    feature_importance = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    fig = px.bar(
        feature_importance,
        title="Feature Importance – Delay Contributors"
    )
    st.plotly_chart(fig)

    st.write("""
    🔍 **This explains which operational variables drive delays most.**  
    Helps decision-makers prioritize improvements.
    """)


# ======================================================================================
#  PAGE 4 – COST IMPACT SIMULATOR
# ======================================================================================

elif page == "⚡ Cost Impact Simulator":

    st.header("Estimate Delay Cost Impact")

    dist = st.slider("Distance (KM)", 0, 500, 200)
    cost = st.slider("Transport Cost (INR)", 1000, 50000, 10000)
    delay = st.slider("Delay Days", 0, 10, 3)

    delay_cost = delay * (cost / 30)

    st.metric("Estimated Delay Cost Impact (INR)", round(delay_cost,2))

    fig_sim = px.line(
        x=[0, delay],
        y=[0, delay_cost],
        title="Cost Growth with Delay Days"
    )
    st.plotly_chart(fig_sim)
