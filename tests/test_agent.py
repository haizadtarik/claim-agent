from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


class FakeToolChat(GenericFakeChatModel):
    """Fake chat model that accepts tool binding, for CI-safe agent tests."""

    def bind_tools(self, tools, **kwargs):
        return self


def _not_fraud_response():
    return {"predictions": [{"fraud_prediction": 0, "fraud_probability": 0.12}]}


def _fraud_response():
    return {"predictions": [{"fraud_prediction": 1, "fraud_probability": 0.91}]}


def test_decide_maps_predictions_to_decisions():
    from claim_agent.agent import decide

    assert decide(0) == "approved"
    assert decide(1) == "disapproved"


def test_assess_claim_approves_legitimate_claim():
    from claim_agent import agent

    with patch.object(agent, "predict_fraud", return_value=_not_fraud_response()):
        result = agent.assess_claim.invoke(
            {"age": 35, "incident_type": "Collision", "total_claim_amount": 5000}
        )

    assert result["decision"] == "approved"
    assert result["fraud_prediction"] == 0
    assert result["fraud_probability"] == 0.12


def test_assess_claim_disapproves_fraudulent_claim():
    from claim_agent import agent

    with patch.object(agent, "predict_fraud", return_value=_fraud_response()):
        result = agent.assess_claim.invoke(
            {"age": 22, "incident_type": "Vehicle Theft", "total_claim_amount": 90000}
        )

    assert result["decision"] == "disapproved"
    assert result["fraud_prediction"] == 1


def test_assess_claim_only_sends_provided_fields():
    from claim_agent import agent

    with patch.object(
        agent, "predict_fraud", return_value=_not_fraud_response()
    ) as mock_predict:
        agent.assess_claim.invoke({"age": 35, "witnesses": 2})

    claims = mock_predict.call_args[0][0]
    assert claims == [{"age": 35, "witnesses": 2}]


def test_assess_claim_requires_some_details():
    from claim_agent import agent

    with patch.object(agent, "predict_fraud") as mock_predict:
        result = agent.assess_claim.invoke({})

    assert "error" in result
    mock_predict.assert_not_called()


def test_assess_claim_coerces_loose_llm_typed_arguments():
    from claim_agent import agent

    with patch.object(
        agent, "predict_fraud", return_value=_fraud_response()
    ) as mock_predict:
        agent.assess_claim.invoke(
            {
                "incident_hour_of_the_day": "3 AM",
                "total_claim_amount": "$72,000",
                "witnesses": False,
                "authorities_contacted": False,
                "incident_severity": "Major Damage",
            }
        )

    claims = mock_predict.call_args[0][0]
    assert claims == [
        {
            "incident_hour_of_the_day": 3,
            "total_claim_amount": 72000.0,
            "witnesses": 0,
            "incident_severity": "Major Damage",
        }
    ]


def test_assess_claim_drops_unparseable_values():
    from claim_agent import agent

    with patch.object(
        agent, "predict_fraud", return_value=_not_fraud_response()
    ) as mock_predict:
        agent.assess_claim.invoke(
            {"age": 35, "witnesses": "none that I know of", "injury_claim": "unknown"}
        )

    claims = mock_predict.call_args[0][0]
    assert claims == [{"age": 35}]


def test_assess_claim_reports_model_errors_without_raising():
    from claim_agent import agent

    with patch.object(
        agent, "predict_fraud", side_effect=RuntimeError("registry down")
    ):
        result = agent.assess_claim.invoke({"age": 35})

    assert "error" in result
    assert "registry down" in result["error"]


def test_assess_claim_is_a_langchain_tool():
    from claim_agent.agent import assess_claim

    assert assess_claim.name == "assess_claim"
    schema = assess_claim.args
    for field in ("incident_type", "incident_severity", "total_claim_amount"):
        assert field in schema


def test_build_agent_binds_tool_and_responds():
    from claim_agent import agent

    fake_llm = FakeToolChat(messages=iter([AIMessage(content="Hello, claimant!")]))
    with patch.object(agent, "ChatOllama", return_value=fake_llm) as mock_ollama:
        graph = agent.build_agent(model_name="gemma4")
        result = graph.invoke({"messages": [{"role": "user", "content": "hi"}]})

    mock_ollama.assert_called_once()
    assert mock_ollama.call_args.kwargs["model"] == "gemma4"
    assert result["messages"][-1].content == "Hello, claimant!"
