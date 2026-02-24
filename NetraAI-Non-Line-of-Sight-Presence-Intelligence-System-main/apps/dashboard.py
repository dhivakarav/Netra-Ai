import json, glob
import streamlit as st

st.title("NetraAI Alerts Dashboard")

files = sorted(glob.glob("outputs/alerts/*.json"))[-200:][::-1]
st.write(f"Alerts: {len(files)}")

for fp in files[:50]:
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    st.subheader(fp.split("\\")[-1])
    st.json(data)
