import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from risk_assistant.dataset import LABELS, build_monthly_panel, derive_labels
from risk_assistant.graph import GraphRepository
from risk_assistant.service import RiskAssistant
from risk_assistant.temporal import add_temporal_features


def test_temporal_labels_use_future_month():
    panel = build_monthly_panel()
    eligible = panel.target_month.notna() & panel.amt_log_growth_2m.notna()
    holdout = int(panel.loc[eligible, "target_month"].max())
    labeled, _ = derive_labels(panel, eligible & (panel.target_month < holdout))
    assert set(LABELS).issubset(labeled.columns)
    assert (labeled.loc[eligible, "target_month"] > labeled.loc[eligible, "STRD_YYMM"]).all()


def test_three_month_features_are_past_only():
    panel, numeric = add_temporal_features(build_monthly_panel())
    assert {"log_amt_lag1", "log_amt_lag2", "log_amt_3m_slope"}.issubset(numeric)
    row = panel.sort_values(["SIDO_NM", "CCG_NM", "TP_BUZ_NM", "STRD_YYMM"]).dropna(
        subset=["log_amt_lag2"]
    ).iloc[0]
    history = panel[
        panel["SIDO_NM"].eq(row["SIDO_NM"])
        & panel["CCG_NM"].eq(row["CCG_NM"])
        & panel["TP_BUZ_NM"].eq(row["TP_BUZ_NM"])
        & panel["STRD_YYMM"].lt(row["STRD_YYMM"])
    ].sort_values("STRD_YYMM")
    assert row["log_amt_lag1"] == history.iloc[-1]["log_amt"]
    assert row["log_amt_lag2"] == history.iloc[-2]["log_amt"]


def test_local_graph_fallback_and_no_llm_call():
    graph = GraphRepository(uri=None, user=None, password=None)
    graph.driver = None
    result = RiskAssistant(graph=graph).diagnose({
        "SIDO_NM": "서울특별시", "CCG_NM": "강남구", "TP_BUZ_NM": "제과점",
        "p_demand_decline": .8, "p_ticket_pressure": .7,
        "p_customer_concentration": .2, "p_competition_pressure": .6,
    })
    assert result["advisor"]["status"] == "LLM_NOT_CALLED"
    assert result["diagnosis"]["retrieved_real_cases"]
    assert result["diagnosis"]["retrieval_mode"] == "structured_cypher_or_local"
    assert all("structured" in case["retrieval_sources"]
               for case in result["diagnosis"]["retrieved_real_cases"])
    assert result["diagnosis"]["primary_risk"] == "demand_decline"
    assert set(result["diagnosis"]["secondary_risks"]) == {"ticket_pressure", "competition_pressure"}


def test_hybrid_reciprocal_rank_fusion_prefers_overlap():
    class FakeEmbedder:
        def available(self):
            return True

        def embed_one(self, text):
            return [0.1, 0.2]

    class FakeGraph(GraphRepository):
        def __init__(self):
            self.driver = object()

        def retrieve(self, risks, industry, limit=6):
            return [{"case_id": "relation_only"}, {"case_id": "overlap"}]

        def retrieve_vector(self, query_embedding, limit=6):
            return [{"case_id": "overlap", "vector_score": .9},
                    {"case_id": "semantic_only", "vector_score": .8}]

    result = FakeGraph().retrieve_hybrid(
        {"demand_decline": .8}, "제과점", "수요 위험", FakeEmbedder())
    assert result[0]["case_id"] == "overlap"
    assert result[0]["retrieval_sources"] == ["structured", "semantic"]
