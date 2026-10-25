from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Message(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=30000)


class CaseInput(StrictModel):
    case_id: str = Field(min_length=1, max_length=160)
    group_id: str = Field(min_length=1, max_length=160)
    task: str = Field(min_length=1, max_length=3000)
    context: list[Message] = Field(min_length=1, max_length=40)
    candidate_output: str = Field(min_length=1, max_length=40000)
    task_type: str = "medical_dialogue"
    source: str = "user_supplied"
    source_version: str = "unversioned"
    split: Literal["dev", "validation", "test", "adhoc"] = "adhoc"


class Rubric(StrictModel):
    rubric_id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=6000)
    level: Literal["general", "task", "case"]
    task_type: str = "medical_dialogue"
    case_id: str = ""
    group_id: str = ""
    split: str = "dev"
    source: str
    version: str
    weight: float = Field(default=1.0, ge=-10, le=10)
    mandatory: bool = False
    approved: bool = False
    applicability: str = ""

    @model_validator(mode="after")
    def bind_case(self):
        if self.level == "case" and not self.case_id:
            raise ValueError("Case-specific rubric must have a case_id.")
        return self


class EvaluationRequest(StrictModel):
    case: CaseInput
    mode: Literal["oracle", "retrieval"] = "oracle"
    provider: Literal["mock", "deepseek"] = "mock"
    rubrics: list[Rubric] = Field(default_factory=list, max_length=60)
    top_k: int = Field(default=4, ge=1, le=12)
    max_run_cost_cny: float = Field(default=0.50, gt=0, le=30)
    benchmark_binary: bool = True

    @model_validator(mode="after")
    def scope(self):
        if self.mode == "oracle" and not self.rubrics:
            raise ValueError("Oracle evaluation requires explicit rubric inputs.")
        if self.mode == "retrieval" and self.rubrics:
            raise ValueError("Retrieval evaluation must not receive oracle rubrics.")
        if len({r.rubric_id for r in self.rubrics}) != len(self.rubrics):
            raise ValueError("Duplicate rubric IDs are rejected before any model call.")
        for r in self.rubrics:
            if not r.approved:
                raise ValueError("Only explicitly approved rubrics can be judged.")
            if r.level == "case" and r.case_id != self.case.case_id:
                raise ValueError("Cross-case rubric is not applicable.")
        if len(canonical(self.model_dump()).encode("utf-8")) > 120000:
            raise ValueError("Input exceeds the local evaluation payload bound.")
        return self


class Judgment(StrictModel):
    judgment: Literal["met", "not_met", "abstain"]
    reason: str = Field(min_length=1, max_length=2000)
    answer_evidence: list[str] = Field(default_factory=list, max_length=5)
    context_evidence: list[str] = Field(default_factory=list, max_length=5)
    review_required: bool = False


def validate_judgment(judgment: Judgment, case: CaseInput, binary: bool) -> None:
    if binary and judgment.judgment == "abstain":
        raise ValueError("Benchmark protocol requires binary met/not_met; abstention remains a failed prediction.")
    contexts = "\n".join(m.content for m in case.context)
    for quote in judgment.answer_evidence:
        if not quote or quote not in case.candidate_output:
            raise ValueError("Answer evidence is not an exact substring of the evaluated answer.")
    for quote in judgment.context_evidence:
        if not quote or quote not in contexts:
            raise ValueError("Context evidence is not an exact substring of supplied context.")
