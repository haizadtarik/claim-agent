from pathlib import Path
from unittest.mock import patch

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from streamlit.testing.v1 import AppTest

UI_SCRIPT = str(Path(__file__).resolve().parents[1] / "src" / "claim_agent" / "ui.py")


class FakeAgent:
    """Stub agent that appends a canned reply, for CI-safe UI tests."""

    def __init__(self, reply="Your claim has been approved."):
        self.reply = reply

    def invoke(self, state):
        messages = list(state["messages"])
        messages.append(AIMessage(content=self.reply))
        return {"messages": messages}


def test_text_flattens_string_and_block_content():
    from claim_agent.ui import _text

    assert _text("hello") == "hello"
    assert _text(["hel", {"type": "text", "text": "lo"}]) == "hello"
    assert _text([{"type": "tool_use", "name": "assess_claim"}]) == ""
    assert _text(None) == ""


def test_visible_messages_hides_tool_traffic():
    from claim_agent.ui import visible_messages

    messages = [
        SystemMessage(content="system prompt"),
        {"role": "user", "content": "Hi, I crashed my car."},
        AIMessage(
            content="", tool_calls=[{"name": "assess_claim", "args": {}, "id": "1"}]
        ),
        ToolMessage(content='{"decision": "approved"}', tool_call_id="1"),
        AIMessage(content="Good news, your claim is approved."),
        HumanMessage(content="Thanks!"),
    ]

    assert visible_messages(messages) == [
        ("user", "Hi, I crashed my car."),
        ("assistant", "Good news, your claim is approved."),
        ("user", "Thanks!"),
    ]


def test_app_renders_and_chats_with_stub_agent():
    from claim_agent import agent

    st.cache_resource.clear()
    with patch.object(agent, "build_agent", return_value=FakeAgent()):
        at = AppTest.from_file(UI_SCRIPT, default_timeout=10).run()
        assert not at.exception
        assert at.title[0].value == "Claim Agent"

        at.chat_input[0].set_value("Hi, I'd like to file a claim.").run()

    assert not at.exception
    rendered = [message.markdown[0].value for message in at.chat_message]
    assert "Hi, I'd like to file a claim." in rendered
    assert "Your claim has been approved." in rendered


def test_app_reports_agent_errors_without_crashing():
    from claim_agent import agent

    st.cache_resource.clear()
    broken = FakeAgent()
    broken.invoke = lambda state: (_ for _ in ()).throw(RuntimeError("Ollama down"))
    with patch.object(agent, "build_agent", return_value=broken):
        at = AppTest.from_file(UI_SCRIPT, default_timeout=10).run()
        at.chat_input[0].set_value("Hello?").run()

    assert not at.exception
    assert any("Ollama down" in err.value for err in at.error)
