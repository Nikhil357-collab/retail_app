# app_replenish_csv.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm

st.title("Retail Forecast & Replenishment (CSV Upload)")

# ------------------ Upload CSV ------------------
uploaded_file = st.file_uploader("Upload Sales CSV", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file, parse_dates=["date"])
    st.write("Preview of Uploaded Data")
    st.dataframe(data.head())

    # ------------------ Feature Engineering ------------------
    def create_features(df):
        df = df.sort_values("date")
        df["lag_1"] = df["qty_sold"].shift(1)
        df["lag_7"] = df["qty_sold"].shift(7)
        df["avg_7"] = df["qty_sold"].shift(1).rolling(7).mean()
        df["std_7"] = df["qty_sold"].shift(1).rolling(7).std()
        df["day_of_week"] = df["date"].dt.dayofweek
        return df

    data = data.groupby(["store_id","item_id"], group_keys=False).apply(create_features).dropna()
    features = ["lag_1","lag_7","avg_7","std_7","day_of_week"]
    X = data[features]
    y = data["qty_sold"]

    # ------------------ Model (RandomForest) ------------------
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    st.success("Model Trained Successfully!")

    # ------------------ Inventory Inputs ------------------
    store = st.text_input("Store ID", value=str(data['store_id'].iloc[0]))
    item = st.text_input("Item ID", value=str(data['item_id'].iloc[0]))
    on_hand = st.number_input("Current Stock", min_value=0, value=50)
    lead_time = st.number_input("Lead Time (days)", min_value=1, value=7)
    service_level = st.slider("Service Level (%)", min_value=50, max_value=99, value=95)

    if st.button("Recommend Order Quantity"):
        pred = rf.predict(X)
        avg_demand = pred.mean()
        resid_std = np.std(y - pred)

        z = norm.ppf(service_level/100)
        demand_lead = avg_demand * lead_time
        safety_stock = z * resid_std * np.sqrt(lead_time)
        ROP = demand_lead + safety_stock
        order_qty = max(0, ROP - on_hand)

        st.subheader("Recommendation")
        st.write(f"Average Daily Demand: {avg_demand:.2f}")
        st.write(f"Safety Stock: {safety_stock:.2f}")
        st.write(f"Reorder Point (ROP): {ROP:.2f}")
        st.write(f"Order Quantity: {int(np.ceil(order_qty))}")
