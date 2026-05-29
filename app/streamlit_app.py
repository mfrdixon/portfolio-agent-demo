from __future__ import annotations

import pandas as pd
import streamlit as st
from pathlib import Path
from agent_backend import ask_agent
from dotenv import load_dotenv
load_dotenv()

BASE = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="Portfolio Agent Demo", layout="wide")
st.title("Portfolio Risk Agent Demo")
st.caption("Read-only agentic AI demo using dummy portfolio data and deterministic Python risk tools.")

positions = pd.read_csv(BASE / "data" / "positions.csv")

with st.sidebar:
    st.header("Demo queries")
    examples = [
        "Summarize the portfolio and identify the biggest concentration risks.",
        "Calculate 1-day 95% VaR and expected shortfall. Explain limitations.",
        "Run a stress test with equities down 20%, credit down 8%, rates up 75 bps, USD up 4%, and vol up 5 vols.",
        "Which positions contribute most to VaR?",
        "Check risk limits and explain any breaches.",
        "Prepare a CIO-style morning risk summary.",
    ]
    selected = st.selectbox("Choose an example", examples)

st.subheader("Dummy portfolio")
st.dataframe(positions, use_container_width=True)

question = st.text_area("Ask the risk agent", value=selected, height=120)

if st.button("Run agent", type="primary"):
    if not question.strip():
        st.warning("Enter a question.")
    else:
        with st.spinner("Agent is calling risk tools..."):
            answer = ask_agent(question)
        st.subheader("Agent answer")
        st.markdown(answer)
