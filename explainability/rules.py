from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

try:
    from aix360.algorithms.rule_induction.ripper import RipperExplainer
except ImportError:  # pragma: no cover
    RipperExplainer = None


@dataclass
class RuleArtifact:
    status: str
    message: str = ""
    prediction: int | None = None
    alignment_with_model: str = ""
    matched_rule: str = ""
    matched_rule_summary: list[str] | None = None
    case_summary: list[str] | None = None
    learned_rules: str = ""


class RuleBasedExplainer:
    def __init__(self, train_frame: pd.DataFrame, train_target: pd.Series) -> None:
        self.available = False
        self.message = ""
        self.rule_model = None
        self.rule_set = None
        self.train_frame = train_frame.copy()
        self.train_target = train_target.copy()
        self.amount_cutoffs = None
        self.burden_cutoffs = None
        self.age_cutoffs = [25, 40, 60]
        if RipperExplainer is None:
            self.message = (
                "AIX360 RIPPER is not installed in this environment. "
                "Install AIX360 in a compatible Python 3.10 environment to enable rule-based explanations."
            )
            return
        try:
            prepared = self._prepare_rule_frame(self.train_frame, fit=True)
            self.rule_model = RipperExplainer(random_state=42)
            self.rule_model.fit(prepared, self.train_target.astype(int), target_label=1)
            self.rule_set = self.rule_model.explain()
            self.available = True
        except Exception as exc:
            self.message = f"RIPPER initialization failed: {exc}"

    def explain_case(self, input_frame: pd.DataFrame, model_prediction: int) -> dict[str, Any]:
        if not self.available or self.rule_model is None:
            return RuleArtifact(status="unavailable", message=self.message).__dict__
        try:
            prepared = self._prepare_rule_frame(input_frame, fit=False)
            rule_prediction = int(self.rule_model.predict(prepared)[0])
            artifact = RuleArtifact(
                status="ready",
                prediction=rule_prediction,
                alignment_with_model="Matches main model" if rule_prediction == model_prediction else "Differs from main model",
                matched_rule=self._matched_rule(prepared.iloc[0], rule_prediction),
                matched_rule_summary=self._matched_rule_summary(prepared.iloc[0], rule_prediction),
                case_summary=self._case_summary(prepared.iloc[0]),
                learned_rules=self._format_rules(),
            )
            return artifact.__dict__
        except Exception as exc:
            return RuleArtifact(status="error", message=f"RIPPER explanation failed: {exc}").__dict__

    def _prepare_rule_frame(self, frame: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = frame.copy()
        if fit:
            self.amount_cutoffs = [
                float(df["credit_amount"].quantile(0.33)),
                float(df["credit_amount"].quantile(0.66)),
            ]
            self.burden_cutoffs = [
                float(df["monthly_burden"].quantile(0.33)),
                float(df["monthly_burden"].quantile(0.66)),
            ]

        def bucket(values: pd.Series, cutoffs: list[float], labels: list[str]) -> pd.Series:
            bins = [-float("inf"), *cutoffs, float("inf")]
            return pd.cut(values.astype(float), bins=bins, labels=labels, include_lowest=True).astype(str)

        prepared = pd.DataFrame(
            {
                "Age Group": pd.cut(
                    df["age"].astype(float),
                    bins=[-float("inf"), *self.age_cutoffs, float("inf")],
                    labels=["Young", "Early Career", "Mid Career", "Senior"],
                    include_lowest=True,
                ).astype(str),
                "Credit Amount Band": bucket(df["credit_amount"], self.amount_cutoffs, ["Low", "Medium", "High"]),
                "Duration Band": pd.cut(
                    df["duration"].astype(float),
                    bins=[-float("inf"), 12, 24, float("inf")],
                    labels=["Short", "Medium", "Long"],
                    include_lowest=True,
                ).astype(str),
                "Monthly Burden Band": bucket(df["monthly_burden"], self.burden_cutoffs, ["Low", "Medium", "High"]),
                "Housing": df["housing"].astype(str).str.title(),
                "Savings": df["saving_accounts"].astype(str).str.title(),
                "Checking": df["checking_account"].astype(str).str.title(),
                "Purpose": df["purpose"].astype(str).str.replace("radio/tv", "Radio or TV").str.title(),
                "Savings Buffer": df["has_savings_buffer"].map({1: "Available", 0: "Limited"}).astype(str),
                "Checking Buffer": df["has_checking_buffer"].map({1: "Available", 0: "Limited"}).astype(str),
            }
        )
        return prepared

    def _case_summary(self, row: pd.Series) -> list[str]:
        summary = [
            f"Age group: {row['Age Group']}",
            f"Credit amount band: {row['Credit Amount Band']}",
            f"Duration band: {row['Duration Band']}",
            f"Monthly burden band: {row['Monthly Burden Band']}",
            f"Checking buffer: {row['Checking Buffer']}",
            f"Savings buffer: {row['Savings Buffer']}",
            f"Housing: {row['Housing']}",
        ]
        return summary

    def _format_rules(self) -> str:
        if self.rule_set is None:
            return ""
        raw = str(self.rule_set).replace("^", " AND ").replace(" v\n", "\nOR\n").replace(" v ", " OR ")
        raw = raw.replace("==", "=").replace("_", " ")
        return raw

    def _matched_rule(self, row: pd.Series, rule_prediction: int) -> str:
        reasons = self._matched_rule_summary(row, rule_prediction) or []
        if not reasons:
            return "No simple rule summary was available for this case."
        joined = " AND ".join(reasons[:3])
        outcome = "Decision risk is high" if rule_prediction == 1 else "Decision risk is low"
        return f"IF {joined} THEN {outcome}."

    def _matched_rule_summary(self, row: pd.Series, rule_prediction: int) -> list[str]:
        reasons: list[str] = []
        if row["Credit Amount Band"] == "High":
            reasons.append("credit amount is high")
        if row["Duration Band"] == "Long":
            reasons.append("repayment duration is long")
        if row["Monthly Burden Band"] == "High":
            reasons.append("monthly burden is high")
        if row["Checking Buffer"] == "Limited":
            reasons.append("checking buffer is limited")
        if row["Savings Buffer"] == "Limited":
            reasons.append("savings buffer is limited")
        if row["Housing"] == "Rent":
            reasons.append("housing is rent")
        if row["Purpose"] in {"Education", "Business", "Vacation/Others"}:
            reasons.append(f"loan purpose is {row['Purpose'].lower()}")
        if not reasons:
            if rule_prediction == 1:
                reasons.append("this case matches a higher-risk rule pattern")
            else:
                reasons.append("this case matches a lower-risk rule pattern")
        return reasons[:4]
