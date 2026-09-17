"""Orchestrate diagnosis, graph retrieval, similar cases and conversational advice."""
from __future__ import annotations

import pandas as pd

from .graph import GraphRepository
from .embeddings import OpenAIEmbedder
from .llm import LLMAdvisor
from .paths import ARTIFACT_ROOT
from .recommender import SimilarRegionRecommender


QUESTION_BANK = {
    "demand_decline": "최근 3개월 점포의 영업일당 주문건수와 재방문 고객이 줄었습니까?",
    "ticket_pressure": "가격·할인·세트구성 변경 후 주문당 마진이 변했습니까?",
    "customer_concentration": "특정 연령·시간대 외 고객의 방문이 줄었는지 확인할 수 있습니까?",
    "competition_pressure": "500m 안 신규 경쟁점과 배달·가격·메뉴가 겹치는 정도가 어떻습니까?",
}


class RiskAssistant:
    def __init__(self, graph: GraphRepository | None = None, llm: LLMAdvisor | None = None,
                 embedder: OpenAIEmbedder | None = None):
        self.graph = graph or GraphRepository()
        self.llm = llm or LLMAdvisor()
        self.embedder = embedder or OpenAIEmbedder()
        history_path = ARTIFACT_ROOT / "holdout_predictions.csv"
        self.recommender = SimilarRegionRecommender(pd.read_csv(history_path)) if history_path.exists() else None

    def diagnose(self, profile: dict, business_answers: dict | None = None) -> dict:
        risks = {key.removeprefix("p_"): float(value) for key, value in profile.items()
                 if key.startswith("p_") and value is not None and pd.notna(value)}
        ranked = sorted(risks.items(), key=lambda item: item[1], reverse=True)
        active = [name for name, score in ranked if score >= .5]
        # The model preserves simultaneous problems; the UI may still show one headline.
        # This is a presentation ranking, not a mutually-exclusive multiclass label.
        primary_risk = ranked[0][0] if ranked else None
        secondary_risks = [name for name in active if name != primary_risk]
        questions = [QUESTION_BANK[name] for name, _ in ranked[:3]]
        industry = profile.get("TP_BUZ_NM", "")
        answer_text = " ".join(f"{key}: {value}" for key, value in (business_answers or {}).items())
        query_text = f"업종 {industry}. 위험 " + ", ".join(
            f"{name} {score:.3f}" for name, score in ranked
        ) + (f". 사업자 답변 {answer_text}" if answer_text else "")
        cases = self.graph.retrieve_hybrid(risks, industry, query_text, self.embedder)
        history = self.graph.retrieve_history(profile.get("SIDO_NM"), profile.get("CCG_NM"),
                                              profile.get("TP_BUZ_NM", ""))
        store_history = self.graph.retrieve_store_history(
            profile.get("SIDO_NM"), profile.get("CCG_NM"), profile.get("TP_BUZ_NM", ""))
        trend_timeline = self.graph.retrieve_trend_timeline([case["case_id"] for case in cases])
        similar = self.recommender.recommend(profile, profile.get("TP_BUZ_NM", "")) if self.recommender else []
        context = {
            "unit": "region_x_industry",
            "region": {"sido": profile.get("SIDO_NM"), "sigungu": profile.get("CCG_NM")},
            "industry": profile.get("TP_BUZ_NM"),
            "risk_probabilities": risks,
            "active_risks": active,
            "primary_risk": primary_risk,
            "secondary_risks": secondary_risks,
            "questions": questions,
            "retrieved_real_cases": cases,
            "retrieval_mode": (
                "hybrid_vector_and_cypher"
                if any("semantic" in case.get("retrieval_sources", []) for case in cases)
                else "structured_cypher_or_local"
            ),
            "region_industry_history": history,
            "historical_store_dynamics": store_history,
            "matched_trend_timeline": trend_timeline,
            "similar_region_cases": similar,
            "critical_limitation": "Not an individual-store closure probability",
        }
        return {"diagnosis": context, "advisor": self.llm.advise(context, business_answers)}
