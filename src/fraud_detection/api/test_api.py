import requests
import json

API_URL = "http://127.0.0.1:8000/predict"


def test_prediction():
    # A sample claim record containing some random features
    sample_data = {
        "claims": [
            {
                "months_as_customer": 328,
                "age": 48,
                "policy_annual_premium": 1406.91,
                "incident_type": "Single Vehicle Collision",
                "incident_severity": "Major Damage",
                "authorities_contacted": "Police",
                "incident_hour_of_the_day": 5,
                "number_of_vehicles_involved": 1,
                "bodily_injuries": 1,
                "witnesses": 2,
                "injury_claim": 6510,
                "property_claim": 13020,
                "vehicle_claim": 52080,
                "total_claim_amount": 71610,
                "insured_hobbies": "sleeping",
            },
            {
                "months_as_customer": 228,
                "age": 42,
                "policy_annual_premium": 1197.22,
                "incident_type": "Vehicle Theft",
                "incident_severity": "Trivial Damage",
                "authorities_contacted": "None",
                "incident_hour_of_the_day": 8,
                "number_of_vehicles_involved": 1,
                "bodily_injuries": 0,
                "witnesses": 1,
                "injury_claim": 0,
                "property_claim": 0,
                "vehicle_claim": 5000,
                "total_claim_amount": 5000,
                "insured_hobbies": "reading",
            },
        ]
    }

    try:
        print("Sending POST request to /predict...")
        response = requests.post(API_URL, json=sample_data)

        if response.status_code == 200:
            print("Success! Response:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error {response.status_code}: {response.text}")

    except requests.exceptions.ConnectionError:
        print(
            "Connection Error: Make sure the FastAPI server is running on http://127.0.0.1:8000"
        )


if __name__ == "__main__":
    test_prediction()
