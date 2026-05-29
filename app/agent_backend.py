from __future__ import annotations

from agents import Agent, Runner
from risk_tools import (
    calculate_component_var,
    calculate_historical_var,
    calculate_sensitivities,
    check_risk_limits,
    get_portfolio_summary,
    get_top_positions,
    run_stress_test,
)
from dotenv import load_dotenv
load_dotenv()

SYSTEM_INSTRUCTIONS = """
You are a read-only portfolio risk analytics agent for an institutional client.

Rules:
- Use tools for all numbers. Never invent portfolio data or risk values.
- Explain assumptions and limitations.
- No trade execution. You may discuss hedging ideas conceptually but must not recommend trades as personalized investment advice.
- Prefer concise executive summaries with bullet points and clear drivers.
- If a query asks for risk change, use available dummy data and state that this is a demo dataset.
"""

portfolio_agent = Agent(
    name="Portfolio Risk Agent",
    instructions=SYSTEM_INSTRUCTIONS,
    tools=[
        get_portfolio_summary,
        get_top_positions,
        calculate_historical_var,
        calculate_component_var,
        run_stress_test,
        calculate_sensitivities,
        check_risk_limits,
    ],
)


def ask_agent(question: str) -> str:
    result = Runner.run_sync(portfolio_agent, question)
    return result.final_output
