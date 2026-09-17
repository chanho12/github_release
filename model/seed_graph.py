#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from risk_assistant.graph import GraphRepository
from risk_assistant.embeddings import OpenAIEmbedder


if __name__ == "__main__":
    graph = GraphRepository()
    if graph.driver is None:
        raise SystemExit("Set NEO4J_URI, NEO4J_USER and NEO4J_PASSWORD first")
    try:
        embedder = OpenAIEmbedder()
        seeded = {"constraints": graph.apply_constraints(),
                  "seeded_cases": graph.seed_cases(), "seeded_trends": graph.seed_trends(),
                  "seeded_observations": graph.seed_observations(),
                  "seeded_store_history": graph.seed_store_history()}
        if embedder.available():
            graph.create_vector_index(embedder.dimensions)
            seeded["seeded_case_embeddings"] = graph.seed_case_embeddings(embedder)
        else:
            seeded["seeded_case_embeddings"] = 0
            seeded["embedding_warning"] = "OPENAI_API_KEY is not set; structured graph was seeded"
        print(seeded)
    finally:
        graph.close()
