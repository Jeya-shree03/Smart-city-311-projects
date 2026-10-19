import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Smart City 311 Analytics", page_icon="🏙️", layout="wide")
st.title("🏙️ Smart City 311 Complaint Analytics & Service Prediction")
st.write("Beginner-friendly mini project using NYC 311 service requests.")

@st.cache_data(ttl=3600)
def load_data(limit=5000):
    url = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
    params = {
        "$limit": limit,
        "$order": "created_date DESC",
        "$where": "created_date >= '2025-01-01T00:00:00'"
    }
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    df = pd.DataFrame(response.json())
    return df

with st.sidebar:
    st.header("Data settings")
    limit = st.selectbox("Number of recent records", [1000, 3000, 5000], index=1)
    st.caption("Data source: NYC Open Data — 311 Service Requests")

try:
    with st.spinner("Loading recent NYC 311 data..."):
        df = load_data(limit)
except Exception as e:
    st.error("Could not load live data. Please check your internet connection and try again.")
    st.caption(f"Technical detail: {e}")
    st.stop()

if df.empty:
    st.warning("No records returned. Try again later.")
    st.stop()

for col in ["created_date", "closed_date"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

df["resolution_days"] = (df["closed_date"] - df["created_date"]).dt.total_seconds() / 86400
df.loc[(df["resolution_days"] < 0) | (df["resolution_days"] > 3650), "resolution_days"] = None
df["status_group"] = df["closed_date"].notna().map({True: "Closed", False: "Not marked closed"})

c1, c2, c3 = st.columns(3)
c1.metric("Records loaded", f"{len(df):,}")
c2.metric("Marked closed", f"{int(df['closed_date'].notna().sum()):,}")
median_days = df["resolution_days"].median()
c3.metric("Median resolution time", f"{median_days:.1f} days" if pd.notna(median_days) else "Not available")

left, right = st.columns(2)
with left:
    st.subheader("Top complaint types")
    if "complaint_type" in df:
        counts = df["complaint_type"].fillna("Unknown").value_counts().head(10).reset_index()
        counts.columns = ["Complaint type", "Count"]
        st.plotly_chart(px.bar(counts, x="Count", y="Complaint type", orientation="h"), use_container_width=True)
with right:
    st.subheader("Requests by borough")
    if "borough" in df:
        boroughs = df["borough"].fillna("Unknown").value_counts().reset_index()
        boroughs.columns = ["Borough", "Count"]
        st.plotly_chart(px.bar(boroughs, x="Borough", y="Count"), use_container_width=True)

st.subheader("Monthly request trend")
if "created_date" in df:
    monthly = df.dropna(subset=["created_date"]).copy()
    monthly["Month"] = monthly["created_date"].dt.strftime("%Y-%m")
    monthly = monthly.groupby("Month").size().reset_index(name="Requests")
    if not monthly.empty:
        st.plotly_chart(px.line(monthly, x="Month", y="Requests", markers=True), use_container_width=True)

st.subheader("Service resolution prediction (demo model)")
st.write("The model predicts whether a request takes more than 7 days to close, using records that already have a valid closure time. This is a learning demo, not an official NYC service forecast.")
model_df = df.dropna(subset=["created_date", "closed_date", "resolution_days"]).copy()
model_df = model_df[model_df["resolution_days"] >= 0]
if len(model_df) >= 30:
    model_df["slow_resolution"] = (model_df["resolution_days"] > 7).astype(int)
    feature_cols = [c for c in ["complaint_type", "borough", "agency"] if c in model_df.columns]
    if feature_cols and model_df["slow_resolution"].nunique() == 2:
        X = pd.get_dummies(model_df[feature_cols].fillna("Unknown"), dummy_na=False)
        y = model_df["slow_resolution"]
        if len(X.columns) and y.value_counts().min() >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )
            model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
            model.fit(X_train, y_train)
            accuracy = accuracy_score(y_test, model.predict(X_test))
            st.metric("Demo model accuracy", f"{accuracy:.1%}")
            st.caption(f"Trained on {len(model_df):,} closed records; test set contains {len(X_test):,} records.")
        else:
            st.info("Not enough examples in each class to train the demo model with this sample.")
    else:
        st.info("The sample does not have enough variation for the demo model.")
else:
    st.info("Not enough closed records in this sample to train the demo model yet.")

with st.expander("Preview data"):
    st.dataframe(df.head(100), use_container_width=True)

st.caption("Educational project. Data is fetched live from NYC Open Data; fields and availability can change.")
