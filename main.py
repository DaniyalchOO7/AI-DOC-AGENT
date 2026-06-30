"""
main.py
-------
Command-line entry point. Accepts a PDF path as an argument,
extracts the text, runs the agent, and prints the trace + summary.

Usage:
    python main.py path/to/receipt.pdf
"""

import sys
from extraction import extract_text
from agent import run_agent_on_text


def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py path/to/document.pdf")
        sys.exit(1)

    file_path = sys.argv[1]
    print(f"\n📄 Processing: {file_path}")

    try:
        raw_text = extract_text(file_path)
    except ValueError as e:
        print(f"\n❌ Extraction failed: {e}")
        sys.exit(1)

    print("✅ Text extracted successfully.\n")

    summary, trace = run_agent_on_text(raw_text)

    print("🤖 Agent reasoning trace:")
    for step in trace:
        print(f"  {step}")

    print("\n📝 Final summary:")
    print(summary)


if __name__ == "__main__":
    main()