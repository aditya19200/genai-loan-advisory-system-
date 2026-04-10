from app.services.compliance_llm import _rule_based_compliance
from app.services.extractor import extract_credit_report_data, validate_document_data


def test_credit_report_parser_handles_summary_layout():
    text = """
    Credit Report
    Client Overview Credit Score Analysis
    Name: Sam Sample Client Current Score 12 Month Post- Net Credit Score
    Address: Sample Street Sam Sample 560 630 +60
    680 725 +45
    Summary of Accounts with Balances
    Installment 10 $2,831 $17,673 2 20.00% $557
    Revolving 24 $2,171 $59,292 9 37.50% $13,724
    """

    data = extract_credit_report_data(text)

    assert data["credit_score"] == 560
    assert data["active_loans"] == 10


def test_validation_requires_manual_review_for_missing_credit_fields():
    validation = validate_document_data(
        "credit_report",
        {"credit_score": None, "active_loans": None},
        {
            "char_count": 1500,
            "ocr_used": False,
            "warnings": [],
        },
    )

    assert validation["review_required"] is True
    assert validation["status"] == "insufficient"
    assert set(validation["missing_critical_fields"]) == {"credit_score", "active_loans"}


def test_rule_based_compliance_uses_manual_review_when_critical_fields_missing():
    report = _rule_based_compliance(
        extracted={"credit_score": None, "active_loans": None},
        form_data={"declared_income": 50000, "declared_loans": 1},
        mismatches={},
        extraction_assessment={
            "review_required": True,
            "missing_critical_fields": ["credit_score", "active_loans"],
            "warnings": ["Missing critical extracted fields: credit_score, active_loans."],
        },
    )

    assert report["compliance_status"] == "Manual Review Required"
    assert report["manual_review_required"] is True
    assert report["warnings"]
