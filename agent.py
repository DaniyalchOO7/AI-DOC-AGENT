"""
agent.py
--------
Core agent loop using the google-genai SDK.
Feeds extracted document text to Gemini, lets the model decide which
tools to call and in what order, and returns both the reasoning trace
and the final plain-English summary.
"""

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

from tools import (
    check_against_budget,
    log_entry,
    send_alert,
    get_spending_summary,
)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in your .env file")

client = genai.Client(api_key=GEMINI_API_KEY)

# Map tool names to actual Python functions — add new tools here
TOOL_MAP = {
    "check_against_budget": check_against_budget,
    "log_entry": log_entry,
    "send_alert": send_alert,
    "get_spending_summary": get_spending_summary,
}

AGENT_INSTRUCTIONS = """
You are a financial document processing agent. You have been given the raw
text of a receipt or invoice. Your job is to process it step by step:

1. Identify the vendor name, expense category (one of: food, travel,
   office_supplies, software, other), total amount, and date.

2. Call get_spending_summary to understand the user's overall spending
   history before making any judgement about this expense.

3. Call check_against_budget with the identified category and amount.

4. Call log_entry to permanently record this expense. Set status to
   'over_budget' if it exceeds the limit, otherwise 'ok'.

5. If the expense is over budget, call send_alert with a clear message
   that includes the vendor name, amount, budget limit, and overage.

6. After all tool calls, give a plain-English summary that includes:
   - What you found in the document (vendor, amount, date, category)
   - Whether it is within budget or over budget, and by how much
   - Context from the full spending history (e.g. "You have now spent
     $340 of your $500 travel budget this month")
   - Any advice or flags worth noting

Be concise but specific. Always ground your summary in the actual numbers.

Document text:
---
{document_text}
---
"""


def run_agent_on_text(document_text: str) -> tuple[str, list[str]]:
    """
    Runs the full agent loop on extracted document text.

    Returns:
        summary  — the model's final plain-English response
        trace    — list of strings showing each tool call and result,
                   displayed in the Streamlit UI as the reasoning trace
    """
    prompt = AGENT_INSTRUCTIONS.format(document_text=document_text)
    trace = []
    messages = [types.Content(role="user", parts=[types.Part(text=prompt)])]

    # Agentic loop: keep going until the model stops calling tools
    while True:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=messages,
            config=types.GenerateContentConfig(
                tools=list(TOOL_MAP.values()),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True  # manual so we can capture the trace
                ),
            ),
        )

        candidate = response.candidates[0].content
        messages.append(candidate)

        # Check for tool calls in this response
        tool_calls = [p for p in candidate.parts if p.function_call]

        if not tool_calls:
            # No tool calls — model is done, return the final text
            final_text = "".join(
                p.text for p in candidate.parts
                if hasattr(p, "text") and p.text
            )
            return final_text, trace

        # Execute each tool call and collect results
        tool_result_parts = []
        for part in candidate.parts:
            if not part.function_call:
                continue

            fn_name = part.function_call.name
            fn_args = dict(part.function_call.args)

            trace.append(f"→ Called: **{fn_name}**")
            trace.append(f"  Args: {fn_args}")

            if fn_name in TOOL_MAP:
                result = TOOL_MAP[fn_name](**fn_args)
            else:
                result = {"error": f"Unknown tool: {fn_name}"}

            trace.append(f"  Result: {result}")

            tool_result_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fn_name,
                        response=result,
                    )
                )
            )

        # Feed tool results back into the conversation
        messages.append(
            types.Content(role="user", parts=tool_result_parts)
        )