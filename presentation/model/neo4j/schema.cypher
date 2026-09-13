CREATE CONSTRAINT case_id_unique IF NOT EXISTS
FOR (c:Case) REQUIRE c.case_id IS UNIQUE;

CREATE CONSTRAINT problem_name_unique IF NOT EXISTS
FOR (p:Problem) REQUIRE p.name IS UNIQUE;

CREATE CONSTRAINT region_key_unique IF NOT EXISTS
FOR (r:Region) REQUIRE r.key IS UNIQUE;

CREATE CONSTRAINT industry_name_unique IF NOT EXISTS
FOR (i:Industry) REQUIRE i.name IS UNIQUE;

CREATE CONSTRAINT intervention_id_unique IF NOT EXISTS
FOR (i:Intervention) REQUIRE i.intervention_id IS UNIQUE;

CREATE CONSTRAINT observation_key_unique IF NOT EXISTS
FOR (o:Observation) REQUIRE o.key IS UNIQUE;

CREATE CONSTRAINT market_history_key_unique IF NOT EXISTS
FOR (h:MarketHistory) REQUIRE h.key IS UNIQUE;

CREATE CONSTRAINT trend_name_unique IF NOT EXISTS
FOR (t:Trend) REQUIRE t.name IS UNIQUE;

// Keep this dimension aligned with OPENAI_EMBEDDING_DIMENSIONS.
CREATE VECTOR INDEX case_embedding IF NOT EXISTS
FOR (c:Case) ON (c.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 1536,
  `vector.similarity_function`: 'cosine'
}};

// Future feedback graph:
// (:Business)-[:TRIED {started_at, ended_at}]->(:Intervention)
// (:Intervention)-[:RESULTED_IN {cnt_lift, amt_lift, margin_lift}]->(:Outcome)
// (:Business)-[:LOCATED_IN]->(:Region)
// (:Business)-[:IN_INDUSTRY]->(:Industry)
// (:Observation)-[:OBSERVED_IN]->(:Region)
// (:Observation)-[:FOR_INDUSTRY]->(:Industry)
