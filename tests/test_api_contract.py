from fastapi.testclient import TestClient

from backend_services.api import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_uses_hld_schema():
    response = client.post(
        "/predict",
        json={
            "age": 32,
            "income": 52000,
            "credit_score": 690,
            "dti": 0.31,
            "employment_length": 6,
            "existing_loans": 1,
            "loan_amount": 300000,
            "tenure_months": 36,
            "ask_explain": True,
            "user_text": "Please explain the decision",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["decision"] in {"Approved", "Rejected"}
    assert "risk_score" in payload
    assert payload["explain_available"] is True


def test_chat_endpoint_returns_reply():
    response = client.post(
        "/chat",
        json={
            "message": "Why was my loan rejected?",
            "history": [],
            "applicant_context": {"decision": "Rejected", "risk_score": 0.74},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["reply"]
    assert payload["source"] in {"gemini", "fallback"}
