# ruff: noqa: E402
import re
import sys
from pathlib import Path

# Add src to sys.path
src_path = str(Path(__file__).resolve().parents[1])
if src_path not in sys.path:
    sys.path.append(src_path)

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from fraud_detection.mcp.server import predict_fraud

DEFAULT_MODEL = "gemma4"

SYSTEM_PROMPT = """\
You are a vehicle insurance claim assistant for the claims department.

Your job, in order:
1. Greet the claimant and gather the details of their claim in conversation:
   incident type and severity, whether authorities were contacted, hour of the
   incident, vehicles involved, bodily injuries, witnesses, claim amounts
   (injury, property, vehicle, total), and about the customer themselves
   (age, months as a customer, annual premium, hobbies). Ask at most two
   short clarifying questions per turn.
2. Once you have at least the incident description and the claim amount, call
   the assess_claim tool with every detail the user provided. Never fill in
   details the user did not give you. Call it again only if the user corrects
   or adds details afterwards. If the claimant asks you to assess the claim
   with the details already given, call the tool immediately instead of
   asking more questions. Pass hours as an integer 0-23, amounts as plain
   numbers, and authorities_contacted as one of "Police", "Fire",
   "Ambulance", "Other", "None".
3. The tool's decision is final. If it returns "approved", tell the user
   their claim is approved. If it returns "disapproved", tell the user their
   claim is disapproved because it was flagged as likely fraudulent. Never
   override or second-guess the decision. You may mention the fraud
   probability.

Be concise, professional, and empathetic.
"""


def decide(fraud_prediction: int) -> str:
    """Map the model's fraud prediction to a claim decision."""
    return "disapproved" if fraud_prediction == 1 else "approved"


_INT_FIELDS = frozenset(
    {
        "months_as_customer",
        "age",
        "incident_hour_of_the_day",
        "number_of_vehicles_involved",
        "bodily_injuries",
        "witnesses",
    }
)
_FLOAT_FIELDS = frozenset(
    {
        "policy_annual_premium",
        "injury_claim",
        "property_claim",
        "vehicle_claim",
        "total_claim_amount",
    }
)


def _coerce(field: str, value):
    """Coerce loosely typed LLM tool arguments; return None when unusable."""
    if field in _INT_FIELDS:
        if isinstance(value, (bool, int, float)):
            return int(value)
        match = re.search(r"-?\d+", str(value))
        return int(match.group()) if match else None
    if field in _FLOAT_FIELDS:
        if isinstance(value, (bool, int, float)):
            return float(value)
        match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
        return float(match.group()) if match else None
    return value if isinstance(value, str) else None


@tool
def assess_claim(
    months_as_customer: int | str | None = None,
    age: int | str | None = None,
    policy_annual_premium: float | str | None = None,
    incident_type: str | None = None,
    incident_severity: str | None = None,
    authorities_contacted: str | bool | None = None,
    incident_hour_of_the_day: int | str | None = None,
    number_of_vehicles_involved: int | str | None = None,
    bodily_injuries: int | str | None = None,
    witnesses: int | str | None = None,
    injury_claim: float | str | None = None,
    property_claim: float | str | None = None,
    vehicle_claim: float | str | None = None,
    total_claim_amount: float | str | None = None,
    insured_hobbies: str | None = None,
) -> dict:
    """Run the fraud model on a claim and return the final approval decision.

    Pass every claim detail the claimant provided and leave unknown fields
    unset. Hours are integers 0-23, claim amounts are numbers in dollars, and
    authorities_contacted is one of "Police", "Fire", "Ambulance", "Other",
    "None". Returns the decision ("approved" or "disapproved"), the fraud
    prediction (1 = fraud, 0 = not fraud) and the fraud probability. The
    decision is final and must be relayed to the claimant as-is.
    """
    provided = {key: value for key, value in locals().items() if value is not None}
    claim = {
        key: coerced
        for key, value in provided.items()
        if (coerced := _coerce(key, value)) is not None
    }
    if not claim:
        return {"error": "No claim details provided; ask the claimant for them."}
    try:
        result = predict_fraud([claim])
    except Exception as exc:
        return {"error": f"Fraud model unavailable: {exc}"}
    prediction = result["predictions"][0]
    return {
        "decision": decide(prediction["fraud_prediction"]),
        "fraud_prediction": prediction["fraud_prediction"],
        "fraud_probability": prediction["fraud_probability"],
    }


def build_agent(model_name: str = DEFAULT_MODEL):
    """Build the claim approval agent on an Ollama-served chat model."""
    llm = ChatOllama(model=model_name, temperature=0.1)
    return create_agent(llm, tools=[assess_claim], system_prompt=SYSTEM_PROMPT)


def main():
    agent = build_agent()
    print("Claim Agent ready (model: %s). Type 'quit' to exit." % DEFAULT_MODEL)
    messages = []
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break
        if not user_input:
            continue
        messages.append({"role": "user", "content": user_input})
        try:
            state = agent.invoke({"messages": messages})
        except Exception as exc:
            print(f"Agent: Sorry, something went wrong ({exc}). Please try again.\n")
            continue
        messages = state["messages"]
        print(f"Agent: {messages[-1].content}\n")


if __name__ == "__main__":
    main()
