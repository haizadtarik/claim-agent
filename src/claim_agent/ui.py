# ruff: noqa: E402
import json
import sys
from pathlib import Path

# Add src to sys.path
src_path = str(Path(__file__).resolve().parents[1])
if src_path not in sys.path:
    sys.path.append(src_path)

import streamlit as st

from claim_agent.agent import DEFAULT_MODEL, build_agent


def _text(content) -> str:
    """Flatten LangChain message content (string or content blocks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def visible_messages(messages) -> list[tuple[str, str]]:
    """Map raw agent state to (role, text) pairs worth rendering.

    Keeps human/user messages and AI messages with text; hides system
    messages, tool calls, and tool results.
    """
    visible = []
    for message in messages:
        if isinstance(message, dict):
            role, content = message.get("role"), message.get("content")
        else:
            role, content = message.type, message.content
        text = _text(content)
        if not text:
            continue
        if role in ("user", "human"):
            visible.append(("user", text))
        elif role in ("assistant", "ai"):
            visible.append(("assistant", text))
    return visible


def latest_decision(messages) -> dict | None:
    """Return the newest assess_claim tool result carrying a decision.

    The official decision must come from the fraud model's tool output, never
    from the LLM's prose — a hallucinated "approved" has no tool result.
    """
    for message in reversed(list(messages)):
        if isinstance(message, dict) or message.type != "tool":
            continue
        if getattr(message, "name", None) != "assess_claim":
            continue
        try:
            payload = json.loads(_text(message.content))
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict) and "decision" in payload:
            return {
                "decision": payload["decision"],
                "fraud_probability": payload.get("fraud_probability"),
            }
    return None


@st.cache_resource
def get_agent(model_name: str = DEFAULT_MODEL):
    return build_agent(model_name)


def main():
    st.set_page_config(page_title="Claim Agent", page_icon="🚗")
    st.title("Claim Agent")
    st.caption(
        "File a vehicle insurance claim in chat. The agent gathers the "
        "details, runs the fraud check, and gives you the final decision."
    )

    with st.sidebar:
        st.markdown(f"**Model:** `{DEFAULT_MODEL}` (via Ollama)")
        if st.button("New conversation"):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    decision = latest_decision(st.session_state.messages)
    if decision is None:
        st.caption(
            "No fraud assessment has been run yet — any decision mentioned "
            "in the chat is not official."
        )
    else:
        prob = decision["fraud_probability"]
        detail = f" (fraud probability {prob:.1%})" if prob is not None else ""
        if decision["decision"] == "approved":
            st.success(f"Official decision from the fraud model: approved{detail}")
        else:
            st.error(f"Official decision from the fraud model: disapproved{detail}")

    for role, text in visible_messages(st.session_state.messages):
        with st.chat_message(role):
            st.markdown(text)

    if prompt := st.chat_input("Describe your claim..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    state = get_agent().invoke({"messages": st.session_state.messages})
                except Exception as exc:
                    st.error(f"Sorry, something went wrong ({exc}). Please try again.")
                else:
                    st.session_state.messages = state["messages"]
                    # Redraw so the decision banner above reflects this turn.
                    st.rerun()


if __name__ == "__main__":
    main()
