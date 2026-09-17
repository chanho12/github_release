"""Neo4j graph store with a local CSV fallback for development."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .paths import ARTIFACT_ROOT, EXTERNAL_ROOT


PROBLEM_TO_CASES = {
    "demand_decline": ["M03", "M05", "M08", "M09"],
    "ticket_pressure": ["M01", "M07", "M09"],
    "customer_concentration": ["M02", "M04", "M05"],
    "competition_pressure": ["M04", "M08", "M09"],
}


class GraphRepository:
    def __init__(self, uri: str | None = None, user: str | None = None,
                 password: str | None = None, require_neo4j: bool = False):
        self.uri = uri or os.getenv("NEO4J_URI")
        self.user = user or os.getenv("NEO4J_USER")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        self.driver = None
        if self.uri and self.user and self.password:
            try:
                from neo4j import GraphDatabase
                self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            except ImportError as exc:
                raise RuntimeError("Install the neo4j package to use the graph database") from exc
        if require_neo4j and self.driver is None:
            raise RuntimeError("Neo4j is mandatory in production: set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD")
        case_path = EXTERNAL_ROOT / "시장트렌드_2026" / "실제_마케팅_사례_근거표.csv"
        self.local_cases = pd.read_csv(case_path)

    def close(self):
        if self.driver:
            self.driver.close()

    def verify_connectivity(self) -> bool:
        if not self.driver:
            return False
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            return False

    def apply_constraints(self) -> int:
        if not self.driver:
            return 0
        statements = [
            "CREATE CONSTRAINT case_id_unique IF NOT EXISTS FOR (n:Case) REQUIRE n.case_id IS UNIQUE",
            "CREATE CONSTRAINT problem_name_unique IF NOT EXISTS FOR (n:Problem) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT region_key_unique IF NOT EXISTS FOR (n:Region) REQUIRE n.key IS UNIQUE",
            "CREATE CONSTRAINT industry_name_unique IF NOT EXISTS FOR (n:Industry) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT observation_key_unique IF NOT EXISTS FOR (n:Observation) REQUIRE n.key IS UNIQUE",
            "CREATE CONSTRAINT market_history_key_unique IF NOT EXISTS FOR (n:MarketHistory) REQUIRE n.key IS UNIQUE",
            "CREATE CONSTRAINT trend_name_unique IF NOT EXISTS FOR (n:Trend) REQUIRE n.name IS UNIQUE",
        ]
        with self.driver.session() as session:
            for statement in statements:
                session.run(statement).consume()
        return len(statements)

    def create_vector_index(self, dimensions: int = 1536) -> bool:
        """Create the Case embedding index used by hybrid retrieval."""
        if not self.driver:
            return False
        query = f"""
        CREATE VECTOR INDEX case_embedding IF NOT EXISTS
        FOR (c:Case) ON (c.embedding)
        OPTIONS {{indexConfig: {{
          `vector.dimensions`: {int(dimensions)},
          `vector.similarity_function`: 'cosine'
        }}}}
        """
        with self.driver.session() as session:
            session.run(query).consume()
        return True

    def seed_cases(self) -> int:
        if not self.driver:
            return 0
        query = """
        MERGE (c:Case {case_id: $case_id})
        SET c.brand=$brand, c.period=$period, c.action=$action, c.result=$result,
            c.source_url=$source_url, c.caution=$caution
        WITH c
        UNWIND $problems AS problem
        MERGE (p:Problem {name: problem})
        MERGE (c)-[:ADDRESSES]->(p)
        """
        reverse = {case_id: problem for problem, ids in PROBLEM_TO_CASES.items() for case_id in ids}
        count = 0
        with self.driver.session() as session:
            for row in self.local_cases.to_dict("records"):
                problems = [p for p, ids in PROBLEM_TO_CASES.items() if row["case_id"] in ids]
                session.run(query, case_id=row["case_id"], brand=row["brand"], period=row["case_period"],
                            action=row["actual_action"], result=row["reported_result"],
                            source_url=row["source_url"], caution=row["caution"], problems=problems)
                count += 1
        return count

    def seed_case_embeddings(self, embedder, batch_size: int = 32) -> int:
        """Embed grounded case text and store vectors on existing Case nodes."""
        if not self.driver or not embedder.available():
            return 0
        rows = self.local_cases.to_dict("records")
        update = """
        UNWIND $rows AS row
        MATCH (c:Case {case_id: row.case_id})
        SET c.embedding=row.embedding, c.embedding_model=$model,
            c.embedding_text=row.embedding_text
        """
        count = 0
        with self.driver.session() as session:
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                texts = [
                    " | ".join(str(row.get(key, "")) for key in
                               ["brand", "actual_action", "reported_result", "caution"])
                    for row in batch
                ]
                vectors = embedder.embed(texts)
                payload = [
                    {"case_id": row["case_id"], "embedding": vector, "embedding_text": text}
                    for row, vector, text in zip(batch, vectors, texts)
                ]
                session.run(update, rows=payload, model=embedder.model).consume()
                count += len(payload)
        return count

    def seed_observations(self, batch_size: int = 500) -> int:
        if not self.driver:
            return 0
        path = ARTIFACT_ROOT / "historical_observations.csv"
        history = pd.read_csv(path)
        query = """
        UNWIND $rows AS row
        MERGE (r:Region {key: row.region_key})
        SET r.sido=row.sido, r.sigungu=row.sigungu
        MERGE (i:Industry {name: row.industry})
        MERGE (o:Observation {key: row.observation_key})
        SET o.month=row.month, o.target_month=row.target_month,
            o.amt_growth=row.amt_growth, o.cnt_growth=row.cnt_growth,
            o.ticket_growth=row.ticket_growth, o.customer_hhi=row.customer_hhi,
            o.supply_count=row.supply_count, o.demand_decline=row.demand_decline,
            o.ticket_pressure=row.ticket_pressure,
            o.customer_concentration=row.customer_concentration,
            o.competition_pressure=row.competition_pressure
        MERGE (o)-[:OBSERVED_IN]->(r)
        MERGE (o)-[:FOR_INDUSTRY]->(i)
        """
        def clean(value):
            return None if pd.isna(value) else value

        rows = []
        for row in history.to_dict("records"):
            region_key = f"{row['SIDO_NM']}|{row['CCG_NM']}"
            rows.append({
                "region_key": region_key, "sido": row["SIDO_NM"], "sigungu": row["CCG_NM"],
                "industry": row["TP_BUZ_NM"],
                "observation_key": f"{row['STRD_YYMM']}|{region_key}|{row['TP_BUZ_NM']}",
                "month": int(row["STRD_YYMM"]), "target_month": int(row["target_month"]),
                "amt_growth": clean(row["amt_log_growth_1m"]), "cnt_growth": clean(row["cnt_log_growth_1m"]),
                "ticket_growth": clean(row["ticket_log_growth_1m"]), "customer_hhi": clean(row["customer_hhi"]),
                "supply_count": clean(row["supply_count"]), "demand_decline": clean(row["demand_decline"]),
                "ticket_pressure": clean(row["ticket_pressure"]),
                "customer_concentration": clean(row["customer_concentration"]),
                "competition_pressure": clean(row["competition_pressure"]),
            })
        with self.driver.session() as session:
            for start in range(0, len(rows), batch_size):
                session.run(query, rows=rows[start:start + batch_size]).consume()
        return len(rows)

    def seed_store_history(self, batch_size: int = 1000) -> int:
        if not self.driver:
            return 0
        history = pd.read_csv(EXTERNAL_ROOT / "재현용_가공패널" / "과거_점포동학_201901_202606.csv")
        query = """
        UNWIND $rows AS row
        MERGE (r:Region {key: row.region_key})
        SET r.sido=row.sido, r.sigungu=row.sigungu
        MERGE (i:Industry {name: row.industry})
        MERGE (h:MarketHistory {key: row.history_key})
        SET h.month=row.month, h.active_start=row.active_start, h.openings=row.openings,
            h.closures=row.closures, h.active_end=row.active_end, h.net_change=row.net_change,
            h.closure_rate=row.closure_rate, h.churn_rate=row.churn_rate
        MERGE (h)-[:OBSERVED_IN]->(r)
        MERGE (h)-[:FOR_INDUSTRY]->(i)
        """
        def clean(value):
            return None if pd.isna(value) else value

        rows = []
        for row in history.to_dict("records"):
            region_key = f"{row['SIDO_NM']}|{row['CCG_NM']}"
            rows.append({
                "region_key": region_key, "sido": row["SIDO_NM"], "sigungu": row["CCG_NM"],
                "industry": row["TP_BUZ_NM"],
                "history_key": f"{row['STRD_YYMM']}|{region_key}|{row['TP_BUZ_NM']}",
                "month": int(row["STRD_YYMM"]), "active_start": int(row["active_start"]),
                "openings": int(row["openings"]), "closures": int(row["closures"]),
                "active_end": int(row["active_end"]), "net_change": int(row["net_change"]),
                "closure_rate": clean(row["closure_rate"]), "churn_rate": clean(row["churn_rate"]),
            })
        with self.driver.session() as session:
            for start in range(0, len(rows), batch_size):
                session.run(query, rows=rows[start:start + batch_size]).consume()
        return len(rows)

    def seed_trends(self) -> int:
        if not self.driver:
            return 0
        timeline = pd.read_csv(EXTERNAL_ROOT / "시장트렌드_2026" / "과거_트렌드_타임라인.csv")
        query = """
        MERGE (t:Trend {name: $trend_name})
        MERGE (c:Case {case_id: $case_id})
        MERGE (c)-[:EVIDENCES]->(t)
        SET t.last_evidence_period=$period
        """
        with self.driver.session() as session:
            for row in timeline.to_dict("records"):
                session.run(query, trend_name=row["trend_name"], case_id=row["case_id"],
                            period=row["case_period"]).consume()
        return len(timeline)

    def retrieve_history(self, sido: str, sigungu: str, industry: str, limit: int = 6) -> list[dict]:
        if self.driver:
            query = """
            MATCH (o:Observation)-[:OBSERVED_IN]->(r:Region),
                  (o)-[:FOR_INDUSTRY]->(i:Industry)
            WHERE r.sido=$sido AND r.sigungu=$sigungu AND i.name=$industry
            RETURN o.month AS month, o.amt_growth AS amt_growth,
                   o.cnt_growth AS cnt_growth, o.ticket_growth AS ticket_growth,
                   o.customer_hhi AS customer_hhi, o.supply_count AS supply_count
            ORDER BY o.month DESC LIMIT $limit
            """
            with self.driver.session() as session:
                return [dict(record) for record in session.run(
                    query, sido=sido, sigungu=sigungu, industry=industry, limit=limit)]
        path = ARTIFACT_ROOT / "historical_observations.csv"
        if not path.exists():
            return []
        history = pd.read_csv(path)
        subset = history[(history["SIDO_NM"] == sido) & (history["CCG_NM"] == sigungu) &
                         (history["TP_BUZ_NM"] == industry)].sort_values("STRD_YYMM", ascending=False)
        columns = ["STRD_YYMM", "amt_log_growth_1m", "cnt_log_growth_1m",
                   "ticket_log_growth_1m", "customer_hhi", "supply_count"]
        return subset[columns].head(limit).where(pd.notna, None).to_dict("records")

    def retrieve_store_history(self, sido: str, sigungu: str, industry: str, limit: int = 24) -> list[dict]:
        if self.driver:
            query = """
            MATCH (h:MarketHistory)-[:OBSERVED_IN]->(r:Region),
                  (h)-[:FOR_INDUSTRY]->(i:Industry)
            WHERE r.sido=$sido AND r.sigungu=$sigungu AND i.name=$industry
            RETURN h.month AS month, h.active_start AS active_start, h.openings AS openings,
                   h.closures AS closures, h.active_end AS active_end, h.net_change AS net_change,
                   h.closure_rate AS closure_rate, h.churn_rate AS churn_rate
            ORDER BY h.month DESC LIMIT $limit
            """
            with self.driver.session() as session:
                return [dict(record) for record in session.run(
                    query, sido=sido, sigungu=sigungu, industry=industry, limit=limit)]
        path = EXTERNAL_ROOT / "재현용_가공패널" / "과거_점포동학_201901_202606.csv"
        if not path.exists():
            return []
        history = pd.read_csv(path)
        subset = history[(history["SIDO_NM"] == sido) & (history["CCG_NM"] == sigungu) &
                         (history["TP_BUZ_NM"] == industry)].sort_values("STRD_YYMM", ascending=False)
        columns = ["STRD_YYMM", "active_start", "openings", "closures", "active_end",
                   "net_change", "closure_rate", "churn_rate"]
        return subset[columns].head(limit).where(pd.notna, None).to_dict("records")

    def retrieve_trend_timeline(self, case_ids: list[str] | None = None) -> list[dict]:
        case_ids = case_ids or self.local_cases["case_id"].tolist()
        if self.driver:
            query = """
            MATCH (c:Case)-[:EVIDENCES]->(t:Trend)
            WHERE c.case_id IN $case_ids
            RETURN c.case_id AS case_id, t.name AS trend_name, c.period AS period,
                   c.brand AS brand, c.action AS action, c.result AS result,
                   c.source_url AS source_url
            ORDER BY period DESC
            """
            with self.driver.session() as session:
                return [dict(record) for record in session.run(query, case_ids=case_ids)]
        path = EXTERNAL_ROOT / "시장트렌드_2026" / "과거_트렌드_타임라인.csv"
        timeline = pd.read_csv(path)
        return timeline[timeline["case_id"].isin(case_ids)].to_dict("records")

    def retrieve(self, risks: dict[str, float], industry: str, limit: int = 6) -> list[dict]:
        active = [name for name, score in risks.items() if score >= .5]
        if not active:
            active = [max(risks, key=risks.get)] if risks else []
        if self.driver:
            query = """
            MATCH (c:Case)-[:ADDRESSES]->(p:Problem)
            WHERE p.name IN $problems
            RETURN c.case_id AS case_id, c.brand AS brand, c.period AS period,
                   c.action AS action, c.result AS result, c.source_url AS source_url,
                   c.caution AS caution, collect(p.name) AS matched_problems
            ORDER BY size(matched_problems) DESC LIMIT $limit
            """
            with self.driver.session() as session:
                return [dict(record) for record in session.run(query, problems=active, limit=limit)]
        wanted = {case_id for problem in active for case_id in PROBLEM_TO_CASES.get(problem, [])}
        cases = self.local_cases[self.local_cases["case_id"].isin(wanted)].head(limit).copy()
        cases["matched_problems"] = cases["case_id"].map(
            lambda cid: [p for p in active if cid in PROBLEM_TO_CASES.get(p, [])])
        return cases.rename(columns={"actual_action": "action", "reported_result": "result"})[
            ["case_id", "brand", "case_period", "action", "result", "source_url", "caution", "matched_problems"]
        ].to_dict("records")

    def retrieve_vector(self, query_embedding: list[float], limit: int = 6) -> list[dict]:
        if not self.driver or not query_embedding:
            return []
        query = """
        CALL db.index.vector.queryNodes('case_embedding', $limit, $embedding)
        YIELD node AS c, score
        OPTIONAL MATCH (c)-[:ADDRESSES]->(p:Problem)
        RETURN c.case_id AS case_id, c.brand AS brand, c.period AS case_period,
               c.action AS action, c.result AS result, c.source_url AS source_url,
               c.caution AS caution, collect(DISTINCT p.name) AS matched_problems,
               score AS vector_score
        ORDER BY vector_score DESC
        """
        try:
            with self.driver.session() as session:
                return [dict(record) for record in session.run(
                    query, limit=limit, embedding=query_embedding)]
        except Exception:
            # Missing index/embeddings must not disable grounded Cypher retrieval.
            return []

    def retrieve_hybrid(self, risks: dict[str, float], industry: str, query_text: str,
                        embedder=None, limit: int = 6) -> list[dict]:
        """Fuse risk-relation and semantic rankings with reciprocal-rank fusion."""
        structured = self.retrieve(risks, industry, limit=max(limit, 10))
        semantic = []
        if self.driver and embedder is not None and embedder.available() and query_text.strip():
            semantic = self.retrieve_vector(embedder.embed_one(query_text), limit=max(limit, 10))

        fused: dict[str, dict] = {}
        for source, rows in (("structured", structured), ("semantic", semantic)):
            for rank, row in enumerate(rows, start=1):
                case_id = row["case_id"]
                item = fused.setdefault(case_id, {**row, "retrieval_sources": [], "rrf_score": 0.0})
                item["retrieval_sources"].append(source)
                item["rrf_score"] += 1.0 / (60 + rank)
                if row.get("vector_score") is not None:
                    item["vector_score"] = float(row["vector_score"])
        return sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)[:limit]
