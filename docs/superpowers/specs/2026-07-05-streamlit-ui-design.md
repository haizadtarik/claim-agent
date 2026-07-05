# Streamlit UI Design — Claim Agent Web Chat

**Date:** 2026-07-05
**Goal:** A browser-based chat UI for the existing claim approval agent so
claimants can converse with it without a terminal.

## Context

`src/claim_agent/agent.py` already exposes everything a UI needs:
`build_agent(model_name)` returns a LangGraph agent whose `invoke({"messages":
...})` carries the conversation, and the terminal `main()` loop is just a thin
shell around it. The `insurance` conda env already has `streamlit` 1.58
installed.

## Approaches considered

1. **Streamlit chat page reusing `build_agent()` in-process (chosen).** One new
   module `src/claim_agent/ui.py` built on `st.chat_message`/`st.chat_input`,
   holding the LangChain message history in `st.session_state` and caching the
   agent with `@st.cache_resource`. Same code path as the terminal chat, no new
   services, CI-testable with `streamlit.testing.v1.AppTest`.
2. **Streamlit front end calling a FastAPI wrapper around the agent.** Adds a
   second process and chat-state serialization for no functional gain in a
   local deployment; the existing FastAPI app serves the raw model, not the
   agent, and nothing else needs an agent HTTP API today.
3. **Form-based Streamlit page hitting `predict_fraud` directly.** Not a UI for
   the agent — it would bypass the conversational interview entirely and
   duplicate what the FastAPI docs page already offers.

## Design

### Components

- `src/claim_agent/ui.py` — the whole UI, mirroring the repo's
  one-module-per-concern layout:
  - `_text(content)` — flatten LangChain message content (string or list of
    content blocks) to displayable text.
  - `visible_messages(messages)` — map raw agent state (`HumanMessage`,
    `AIMessage`, `ToolMessage`, plus plain `{"role", "content"}` dicts) to
    `(role, text)` pairs for rendering: human/user messages and non-empty AI
    messages only; tool calls, tool results, and system messages stay hidden.
  - `get_agent()` — `@st.cache_resource`-cached `build_agent(DEFAULT_MODEL)`
    so the agent is built once per server process.
  - `main()` — page config, title, sidebar (model name + "New conversation"
    reset button), render history, then on each `st.chat_input` submission:
    append the user message, `invoke` the agent with the full history inside a
    spinner, replace the session history with the returned state, and render
    the reply. Guarded by `if __name__ == "__main__"` (true under
    `streamlit run` and `AppTest`) so the module stays importable in tests.

### Data flow

Browser input → `st.chat_input` → session history + new user message →
`agent.invoke` (gemma4 + `assess_claim` tool → shared `predict_fraud`) →
returned state stored in `st.session_state` → `visible_messages` renders the
conversation, hiding tool traffic.

### Error handling

- Agent invocation fails (Ollama down, registry missing) → `st.error` with the
  exception message, mirroring the terminal loop's apology; the user's message
  stays in history so they can retry.
- Model-level failures are already handled inside `assess_claim`, which
  returns an error payload the agent relays conversationally.

### Testing

- `tests/test_ui.py`, CI-safe (no Ollama, no MLflow):
  - unit tests for `_text` and `visible_messages` filtering/mapping;
  - an `AppTest` end-to-end run with `build_agent` patched to a stub agent:
    page renders without exception, a chat submission renders the user
    message and the stub's reply.
- Manual/E2E: `make ui` against real gemma4 + registered model.

### Packaging

- New Makefile target `ui` (`PYTHONPATH=src streamlit run src/claim_agent/ui.py`).
- `requirements.txt` gains `streamlit>=1.33.0`.
- README section 7 gains the web UI alternative to the terminal chat.
