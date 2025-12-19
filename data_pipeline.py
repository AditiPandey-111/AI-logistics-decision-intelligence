import pandas as pd
import numpy as np

def load_and_prepare_data():

    # -------- load raw CSVs -------- #
    orders = pd.read_csv("data/orders.csv")
    delivery = pd.read_csv("data/delivery_performance.csv")
    routes = pd.read_csv("data/routes_distance.csv")
    fleet = pd.read_csv("data/vehicle_fleet.csv")
    cost = pd.read_csv("data/cost_breakdown.csv")

    # -------- merge datasets -------- #
    df = orders.merge(delivery, on="Order_ID", how="left")
    df = df.merge(routes, on="Order_ID", how="left")
    df = df.merge(cost, on="Order_ID", how="left")

    # no vehicle ID exists → mapping based on logic
    df = assign_vehicle(df, fleet)

    # fill missing values
    df = df.fillna(0)

    # -------- derived delay metrics -------- #
    df["Delay_Days"] = df["Actual_Delivery_Days"] - df["Promised_Delivery_Days"]
    df["Delay_Flag"] = np.where(df["Delay_Days"] > 0, 1, 0)

    # -------- cost metrics -------- #
    df["Transport_Cost"] = (
    df["Fuel_Cost"] +
    df["Labor_Cost"] +
    df["Vehicle_Maintenance"] +
    df["Insurance"] +
    df["Packaging_Cost"] +
    df["Technology_Platform_Fee"] +
    df["Other_Overhead"]
)

    df["Delay_Cost_Impact"] = df["Delay_Days"] * (df["Transport_Cost"] / 30)

    return df


def assign_vehicle(df, fleet_df):

    # helper: pick nearest vehicle based on warehouse location + vehicle availability
    mapping = {}

    for idx, row in df.iterrows():

        warehouse = row["Origin"]

        candidates = fleet_df[fleet_df["Current_Location"] == warehouse]

        if len(candidates) == 0:
            assigned = fleet_df.sample(1).iloc[0]
        else:
            assigned = candidates.sort_values("Age_Years").iloc[0]

        mapping[row["Order_ID"]] = {
            "Vehicle_ID": assigned["Vehicle_ID"],
            "Vehicle_Type": assigned["Vehicle_Type"],
            "Vehicle_Age": assigned["Age_Years"],
            "CO2_Rate": assigned["CO2_Emissions_Kg_per_KM"],
        }

    mapped_df = pd.DataFrame(mapping).T.reset_index().rename(columns={"index":"Order_ID"})
    df = df.merge(mapped_df, on="Order_ID", how="left")

    return df







# ---------------- FEATURE ENGINEERING LAYER ---------------- #

def add_feature_engineering(df): 

        # ensure numeric types for delay + severity fields
    numeric_cols = [
        "Delay_Days",
        "Traffic_Delay_Minutes",
        "Transport_Cost",
        "Customer_Rating",
        "Weather_Impact"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)


    # --- Route efficiency score --- #
    # lower delay + lower cost + lower distance → better
    df["Route_Efficiency_Index"] = (
        (1 / (df["Delay_Days"] + 1)) *        # delivery speed
        (1 / (df["Transport_Cost"] + 1)) *    # cost efficiency
        (1 / (df["Distance_KM"] + 1))         # shorter distance
    )

    # normalize for readability (0–100 score)
    df["Route_Efficiency_Index"] = (
        df["Route_Efficiency_Index"] /
        df["Route_Efficiency_Index"].max()
    ) * 100


    # --- Cost anomaly baseline --- #
    mean_cost = df["Transport_Cost"].mean()
    std_cost = df["Transport_Cost"].std()

    # anomaly if cost > μ + 2σ
    df["Cost_Anomaly_Flag"] = np.where(
        df["Transport_Cost"] > mean_cost + 2*std_cost,
        1,0
    )

    # anomaly score for ranking severity
    df["Cost_Anomaly_Score"] = (
        (df["Transport_Cost"] - mean_cost) /
        (std_cost + 1e-6)
    ).clip(lower=0)


    # --- Delay severity score --- #
    df["Delay_Severity_Score"] = (
        df["Delay_Days"] * 0.5 + 
        df["Traffic_Delay_Minutes"] * 0.3 + 
        (5 - df["Customer_Rating"]) * 1.2 +
        df["Weather_Impact"] * 0.8
    )

    return df




# ---------------- ML PREP ---------------- #

def prepare_training_data(df):

    feature_columns = [
        "Distance_KM",
        "Vehicle_Age",
        "Traffic_Delay_Minutes",
        "Transport_Cost",
        "Delay_Severity_Score"
    ]

    df_train = df[feature_columns + ["Delay_Flag"]].dropna()

    X = df_train[feature_columns]
    y = df_train["Delay_Flag"]

    return X, y
