"""
tools.py
--------
Defines the real-world actions the agent can take. Each function here
becomes a "tool" that Gemini can choose to call. Keeping these in their
own file means adding a new tool later is just adding a new function
and registering it in agent.py — nothing else needs to change.
"""

import os
import csv
import json
from datetime import datetime
from pathlib import Path

import requests

LOG_FILE = Path("expense_log.csv")
BUDGET_FILE = Path("budgets.json")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")


def get_budgets() -> dict:
    """Load budget limits per category. Defaults are created on first run."""
    if not BUDGET_FILE.exists():
        defaults = {
            "food": 300,
            "travel": 500,
            "office_supplies": 150,
            "software": 200,
            "other": 100,
        }
        BUDGET_FILE.write_text(json.dumps(defaults, indent=2))
        return defaults
    return json.loads(BUDGET_FILE.read_text())


def check_against_budget(category: str, amount: float) -> dict:
    """
    Tool: Check whether an expense amount exceeds the budget for its category.
    Returns the category, amount, budget limit, whether it is over budget,
    and by how much.

    Args:
        category: The expense category (food, travel, office_supplies, software, other)
        amount: The total expense amount in dollars
    """
    budgets = get_budgets()
    category_key = category.lower().strip().replace(" ", "_")
    limit = budgets.get(category_key, budgets.get("other", 100))
    over_budget = amount > limit
    return {
        "category": category,
        "amount": amount,
        "budget_limit": limit,
        "over_budget": over_budget,
        "overage_amount": round(amount - limit, 2) if over_budget else 0,
    }


def log_entry(vendor: str, category: str, amount: float, date: str, status: str) -> dict:
    """
    Tool: Append a structured record of this expense to the CSV log.
    This acts as the persistent memory of everything the agent has processed.

    Args:
        vendor: The name of the vendor or merchant
        category: The expense category (food, travel, office_supplies, software, other)
        amount: The total expense amount in dollars
        date: The date of the expense (e.g. 2024-06-01)
        status: Either 'ok' or 'over_budget'
    """
    file_exists = LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "vendor", "category", "amount", "date", "status"])
        writer.writerow([datetime.now().isoformat(), vendor, category, amount, date, status])
    return {"logged": True, "file": str(LOG_FILE)}


def send_alert(message: str) -> dict:
    """
    Tool: Send a notification about a budget issue. Uses a Slack webhook
    if configured in the environment, otherwise falls back to printing
    to the console so the project still runs without any external setup.

    Args:
        message: The alert message to send, describing the vendor, amount, and overage
    """
    if SLACK_WEBHOOK_URL:
        try:
            resp = requests.post(SLACK_WEBHOOK_URL, json={"text": message}, timeout=5)
            resp.raise_for_status()
            return {"sent": True, "channel": "slack"}
        except requests.RequestException as e:
            print(f"[Slack send failed: {e}] Falling back to console.")
    print(f"\n🔔 ALERT: {message}\n")
    return {"sent": True, "channel": "console"}


def get_spending_summary() -> dict:
    """
    Tool: Return total spending per category from the full expense log.
    Use this to give context-aware advice — for example, telling the user
    they are close to their monthly limit across all office supply purchases,
    not just on this one receipt.

    Returns a breakdown of total spent per category and the overall total.
    """
    if not LOG_FILE.exists():
        return {"summary": "No expenses logged yet.", "spending_by_category": {}, "total_spent": 0}

    import pandas as pd
    df = pd.read_csv(LOG_FILE)

    if df.empty:
        return {"summary": "No expenses logged yet.", "spending_by_category": {}, "total_spent": 0}

    by_category = df.groupby("category")["amount"].sum().round(2).to_dict()
    total = round(df["amount"].sum(), 2)
    budgets = get_budgets()

    # Add how much of the budget has been used per category
    usage = {}
    for cat, spent in by_category.items():
        limit = budgets.get(cat.lower().replace(" ", "_"), budgets.get("other", 100))
        usage[cat] = {
            "spent": spent,
            "limit": limit,
            "remaining": round(limit - spent, 2),
            "percent_used": round((spent / limit) * 100, 1) if limit > 0 else 0,
        }

    return {
        "spending_by_category": usage,
        "total_spent": total,
    }