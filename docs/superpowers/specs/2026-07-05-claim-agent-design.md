# Claim Agent Design — LangChain + Ollama gemma4

**Date:** 2026-07-05
**Goal:** An AI agent that chats with a claimant, scores their vehicle insurance
claim with the existing fraud detection model, and approves the claim when the
model says it is not fraud or disapproves it when the model says it is fraud.

## Context

The repo already ships the fraud detector three ways: a training pipeline that
registers `FraudDetectionModel` in a local MLflow registry, a FastAPI app, and
an MCP server (`src/fraud_detection/mcp/server.py`) exposing `predict_fraud`.
The `insurance` conda env already has `langchain` 1.3, `langchain-ollama`,
`langgraph`, and Ollama serves `gemma4:latest`, which natively supports tool
calling.

## Approaches considered

1. **LangChain `create_agent` with a local fraud tool (chosen).** Build a
   tool-calling agent (`langchain.agents.create_agent`) on
   `ChatOllama(model="gemma4")`. The tool wraps the same scoring function the
   MCP server uses (imported from `fraud_detection.mcp.server`), so there is
   one scoring code path and no duplicated MLflow logic.
2. **Custom LangGraph state machine** (collect → extract → predict → decide →
   respond). More control over the flow, but more code, and unnecessary since
   gemma4 supports native tool calling.
3. **Agent as an MCP client of the existing stdio server.** Architecturally
   pure, but requires `langchain-mcp-adapters` (not installed) plus async
   subprocess lifecycle management, for no functional gain when the scoring
   function is importable in-process. The MCP server remains available for
   external assistants.

## Design

### Components

- `src/claim_agent/agent.py` — the whole agent, mirroring the repo's
  one-module-per-concern layout:
  - `decide(fraud_prediction)` — deterministic policy: `1 → "disapproved"`,
    `0 → "approved"`. The LLM never makes the approval decision; it only
    relays it.
  - `assess_claim(...)` — a `@tool` with flat, optional, documented parameters
    matching the model's claim features (age, incident_type,
    incident_severity, total_claim_amount, …). Calls the shared
    `predict_fraud`, returns `{decision, fraud_prediction, fraud_probability}`.
  - `build_agent(model_name="gemma4")` — `create_agent` over `ChatOllama` with
    the tool and a system prompt: greet the claimant, gather claim details
    conversationally, call `assess_claim` once enough details are collected,
    then deliver the tool's decision (approved/disapproved) with the fraud
    probability, never overriding it.
  - `main()` — terminal chat loop (`input()`), carrying the message history
    across turns; `quit`/`exit` ends the session.

### Data flow

User message → agent (gemma4) asks follow-ups until it has claim details →
tool call `assess_claim` → shared `predict_fraud` (MLflow model) → tool returns
deterministic decision → agent tells the user approved or disapproved.

### Error handling

- Model registry unavailable → the tool returns an error string; the agent
  apologizes and asks the user to try later (no crash mid-chat).
- Ollama not running → `main()` fails fast with a clear message.

### Testing

- `tests/test_agent.py`: unit tests for `decide`, for `assess_claim`'s output
  contract with `predict_fraud` mocked (no MLflow needed), and for agent
  construction with a fake chat model (no Ollama needed) — CI-safe.
- Manual/E2E: scripted chat against real gemma4 + real model covering one
  legitimate-looking claim (approve) and one fraud-looking claim (disapprove).

### Packaging

- New Makefile target `agent` (`PYTHONPATH=src python -m claim_agent.agent`).
- `requirements.txt` gains `langchain>=1.0`, `langchain-ollama>=1.0`.
- README gets a "Chat with the Claim Agent" section.
