"""OpenAI Responses API adapter. No request is made without an API key."""
from __future__ import annotations

import json
import os


SYSTEM_INSTRUCTIONS = """
너는 소상공인 경영진단 조수다. 모델 출력은 개별 점포의 폐업확률이 아니라 지역×업종 신호다.
관측 사실, 가능한 설명, 확인할 질문, 조건부 실험을 반드시 구분한다.
외부 사례의 성과를 해당 사업자의 예상 성과로 단정하지 말고, 질문은 최대 3개만 제시한다.
제공된 evidence와 business_answers만 근거로 사용하고, 없는 수치·사례·출처는 만들지 않는다.
retrieved_real_cases를 인용할 때에는 evidence_refs에 case_id와 source_url을 적는다.
근거가 부족한 원인은 '확인 필요'로 표현하고, 추천은 2~4주 동안 검증할 작은 실험으로 작성한다.
답변은 JSON으로 작성한다: summary, observed_signals, questions, recommended_experiments, kpis,
evidence_refs, cautions.
""".strip()

ADVICE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "observed_signals": {"type": "array", "items": {"type": "string"}},
        "questions": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "recommended_experiments": {"type": "array", "items": {"type": "string"}},
        "kpis": {"type": "array", "items": {"type": "string"}},
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
        "cautions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "observed_signals", "questions",
                 "recommended_experiments", "kpis", "evidence_refs", "cautions"],
    "additionalProperties": False,
}


class LLMAdvisor:
    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")
        self.api_key = os.getenv("OPENAI_API_KEY")

    def available(self) -> bool:
        return bool(self.api_key)

    def advise(self, context: dict, business_answers: dict | None = None) -> dict:
        if not self.available():
            return {"status": "LLM_NOT_CALLED", "reason": "OPENAI_API_KEY is not set",
                    "context": context, "business_answers": business_answers or {}}
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the openai package to call the LLM") from exc
        client = OpenAI(api_key=self.api_key)
        payload = {"evidence": context, "business_answers": business_answers or {}}
        response = client.responses.create(
            model=self.model,
            instructions=SYSTEM_INSTRUCTIONS,
            input=json.dumps(payload, ensure_ascii=False),
            text={"format": {"type": "json_schema", "name": "business_risk_advice",
                              "strict": True, "schema": ADVICE_SCHEMA}},
            store=False,
        )
        text = response.output_text
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = {"summary": text}
        parsed["status"] = "LLM_CALLED"
        parsed["model"] = self.model
        return parsed
