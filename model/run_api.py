#!/usr/bin/env python3
"""Validate configuration, optionally seed Neo4j, and start the diagnosis API."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from risk_assistant.embeddings import OpenAIEmbedder
from risk_assistant.graph import GraphRepository


def validate_environment() -> None:
    required = ("OPENAI_API_KEY", "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise SystemExit(
            "Missing configuration: " + ", ".join(missing)
            + ". Copy model/.env.example to model/.env and fill the values."
        )


def validate_connection(graph: GraphRepository) -> None:
    if not graph.verify_connectivity():
        raise SystemExit(
            "Neo4j connection failed. Start Neo4j and check NEO4J_URI, "
            "NEO4J_USER, and NEO4J_PASSWORD in model/.env."
        )


def seed(graph: GraphRepository) -> None:
    embedder = OpenAIEmbedder()
    result = {
        "constraints": graph.apply_constraints(),
        "cases": graph.seed_cases(),
        "trends": graph.seed_trends(),
        "observations": graph.seed_observations(),
        "store_history": graph.seed_store_history(),
    }
    graph.create_vector_index(embedder.dimensions)
    result["case_embeddings"] = graph.seed_case_embeddings(embedder)
    print("Neo4j seed completed:", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", action="store_true", help="Seed graph and embeddings before serving")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    validate_environment()
    repository = GraphRepository(require_neo4j=True)
    try:
        validate_connection(repository)
        if args.seed:
            seed(repository)
    finally:
        repository.close()

    import uvicorn

    uvicorn.run("risk_assistant.api:app", host=args.host, port=args.port, reload=False)
