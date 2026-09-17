"""Local case-based retrieval; upgradeable to collaborative filtering with feedback logs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


PROBABILITY_COLUMNS = [
    "p_demand_decline", "p_ticket_pressure", "p_customer_concentration", "p_competition_pressure"
]


class SimilarRegionRecommender:
    def __init__(self, history: pd.DataFrame):
        available = [c for c in PROBABILITY_COLUMNS if c in history]
        if not available:
            raise ValueError("At least one risk probability column is required")
        self.columns = available
        self.history = history.reset_index(drop=True).copy()
        self.scaler = StandardScaler().fit(self.history[self.columns].fillna(0))
        matrix = self.scaler.transform(self.history[self.columns].fillna(0))
        self.index = NearestNeighbors(metric="cosine").fit(matrix)

    def recommend(self, profile: dict, industry: str, n: int = 5) -> list[dict]:
        candidates = self.history.index[self.history["TP_BUZ_NM"].eq(industry)].to_numpy()
        if not len(candidates):
            candidates = self.history.index.to_numpy()
        query = pd.DataFrame([{c: profile.get(c, 0.0) for c in self.columns}])
        query_vector = self.scaler.transform(query)
        candidate_vectors = self.scaler.transform(self.history.loc[candidates, self.columns].fillna(0))
        distance = 1 - np.dot(candidate_vectors, query_vector[0]) / (
            np.linalg.norm(candidate_vectors, axis=1) * np.linalg.norm(query_vector[0]) + 1e-12)
        chosen = candidates[np.argsort(distance)[:n]]
        columns = [c for c in ["SIDO_NM", "CCG_NM", "TP_BUZ_NM", "target_month",
                               "recommended_case_ids", "observed_outcome"] + self.columns if c in self.history]
        output = self.history.loc[chosen, columns].copy()
        output["similarity"] = 1 - distance[np.argsort(distance)[:n]]
        return output.to_dict("records")


def collaborative_status(feedback: pd.DataFrame | None) -> dict:
    """Do not call nearest-neighbor retrieval collaborative filtering without interactions."""
    required = {"business_id", "intervention_id", "outcome_score"}
    if feedback is None or not required.issubset(feedback.columns) or len(feedback) < 100:
        return {"mode": "CASE_BASED_RETRIEVAL", "reason": "user-intervention-outcome interactions are insufficient"}
    return {"mode": "COLLABORATIVE_FILTERING_READY", "interactions": int(len(feedback))}
