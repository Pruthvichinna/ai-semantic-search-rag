"""
Streamlit dashboard to visualize RAG query logs.
Run:
    streamlit run code/monitoring_dashboard.py
Logs format (JSON list):
[
  {"query":"...","latency":1.42,"relevance":0.92,"tokens":534,"timestamp":"2025-08-15T12:00:00Z"}
]
"""

import json
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="RAG Monitoring", layout="wide")
st.title("RAG System Monitoring Dashboard")

uploaded = st.file_uploader("Upload query logs (JSON)", type=["json"])
if uploaded:
    try:
        logs = json.load(uploaded)
        df = pd.DataFrame(logs)
        st.success(f"Loaded {len(df)} log rows.")
        if not df.empty:
            cols = st.columns(4)
            cols[0].metric("Avg Relevance", f"{df['relevance'].mean():.2f}")
            cols[1].metric("Avg Latency (s)", f"{df['latency'].mean():.2f}")
            cols[2].metric("Median Tokens", f"{df['tokens'].median():.0f}")
            cols[3].metric("Total Queries", f"{len(df)}")

            st.subheader("Logs Table")
            st.dataframe(df.sort_values(by=df.columns[0], ascending=False), use_container_width=True)

            st.subheader("Relevance over time")
            if "timestamp" in df.columns and "relevance" in df.columns:
                df_ts = df.dropna(subset=["timestamp", "relevance"]).copy()
                if not df_ts.empty:
                    df_ts["timestamp"] = pd.to_datetime(df_ts["timestamp"])
                    st.line_chart(df_ts.set_index("timestamp")["relevance"])
    except Exception as e:
        st.error(f"Failed to parse JSON: {e}")
else:
    st.info("Upload a JSON file of logs to view analytics.")
