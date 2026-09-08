#!/usr/bin/env python3
"""Deterministic diagnosis engine used by the validation package.

The engine never computes a business cause. It turns a precomputed Evidence
Card into an ordered verification workflow. Missing store information results
in abstention, not imputation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_RULE_ORDER = ["F_PEAK", "C_TOTAL_INDUSTRY", "A_CNT", "B_TICKET", "D_CONCENTRATION", "E_SUPPLY"]
MAX_QUESTIONS = 3


def _metric(card: dict, path: str, default=None):
    value: Any = card
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def triggered_rule_ids(card: dict) -> list[str]:
    ids: list[str] = []
    if _metric(card, "trend.peak_dependent", False):
        ids.append("F_PEAK")
    if _metric(card, "region_total_vs_industry.classification") != "같은방향":
        ids.append("C_TOTAL_INDUSTRY")
    if _metric(card, "trend.cnt_direction") == "감소":
        ids.append("A_CNT")
    if _metric(card, "trend.ticket_direction") == "감소":
        ids.append("B_TICKET")
    if (_metric(card, "composition.age_concentration_up_all_numeric", False) or
            _metric(card, "composition.industry_concentration_up", False)):
        ids.append("D_CONCENTRATION")
    supply = _metric(card, "external_modules.supply", {}) or {}
    if supply.get("applicable") and supply.get("supply_direction") == "상승" and _metric(card, "trend.cnt_direction") in {"감소", "정체"}:
        ids.append("E_SUPPLY")
    return [rid for rid in DEFAULT_RULE_ORDER if rid in ids]


def _rules_by_id(rules: list[dict]) -> dict[str, dict]:
    return {r["rule_id"]: r for r in rules}


def baseline_output(card: dict) -> dict:
    """A: a reasonable general dashboard and checklist using the same card."""
    t = card["trend"]
    facts = [
        f"2026년 1~6월 AMT 로그 월성장률 {t['amt_growth_pct_m']:+.2f}% ({t['amt_direction']}).",
        f"CNT 로그 월성장률 {t['cnt_growth_pct_m']:+.2f}% ({t['cnt_direction']}).",
        f"건당 평균 결제금액 로그 월성장률 {t['ticket_growth_pct_m']:+.2f}% ({t['ticket_direction']}).",
        f"전국 동일업종 대비 AMT 점유율 월기울기 {card['national_same_industry']['amt_share_slope_pp_m']:+.4f}%p.",
        f"지역총량과 업종 방향 관계: {card['region_total_vs_industry']['classification']}.",
    ]
    checklist = [
        "월별 AMT·CNT·건당 평균 결제금액과 추세 강건성을 함께 확인한다.",
        "전국 동일업종 흐름 및 지역총량과 해당 업종의 차이를 확인한다.",
        "연령·업종 구성비가 절대 건수와 함께 변했는지 확인한다.",
        "가용한 경우 외부 공급자료의 기간·업종 매핑·모집단을 확인한다.",
        "점포 영업일·주문구성·가격/할인·고유고객수 자료를 확인한 뒤 개선안을 정한다.",
    ]
    questions = [
        "분석기간에 실제 영업일·영업시간 또는 휴점 변화가 있었습니까?",
        "가격·할인·상품/주문구성·결제방식 중 바뀐 항목이 있었습니까?",
        "점포 주문건수·고유고객수·재방문 또는 주변 경쟁점 변화를 확인할 수 있습니까?",
    ]
    return {
        "method": "A_GENERAL_DASHBOARD_CHECKLIST",
        "case_id": card["case_id"],
        "confirmed_facts": facts,
        "checklist": checklist[:MAX_QUESTIONS],
        "followup_questions": questions,
        "possible_explanations": ["거래건수, 건당 결제금액, 구성 변화가 함께 또는 상쇄해 나타났을 수 있다."],
        "unknown_information": card["missing_information"],
        "conditional_improvement_candidates": [
            "영업노출 변화가 확인되면 영업일·시간당 지표로 다시 비교한다.",
            "상품·가격·할인 변화가 확인되면 한 요소만 2~4주 소규모 시험한다.",
            "실제 고객·경쟁 변화가 확인되면 해당 경로에 맞는 유지·확장 시험을 정한다.",
        ],
        "abstentions": [
            "점포 운영·주문·고객 자료가 없으면 지역 수준 진단으로 제한한다.",
            "건당 결제금액만으로 가격·마진 원인을 판단하지 않는다.",
            "개선효과와 폐업위험을 추정하지 않는다.",
        ],
        "limitations": card["limitations"],
        "first_priority_code": "GENERAL_REVIEW",
        "source_card_hash": card["card_hash"],
    }


def c0_output(card: dict, rules: list[dict], store_answers: dict | None = None) -> dict:
    """C0: rule/template only; no LLM and no invented store facts."""
    store_answers = store_answers or {}
    by_id = _rules_by_id(rules)
    ids = triggered_rule_ids(card)
    active_ids = ids[:MAX_QUESTIONS]
    checks, questions, branches, actions, metrics, abstentions = [], [], [], [], [], []
    for rank, rid in enumerate(active_ids, 1):
        rule = by_id[rid]
        checks.append({
            "rank": rank,
            "rule_id": rid,
            "observed_signal": rule["observed_fact_template"],
            "priority_reason": rule["priority_rationale"],
        })
        questions.append({"rank": rank, "rule_id": rid, "question": rule["followup_question"]})
        answer_key = rule["answer_key"]
        answer = store_answers.get(answer_key, "모름")
        branch_map = json.loads(rule["answer_branch_json"])
        branch = branch_map.get(str(answer), branch_map.get("모름"))
        branches.append({"rule_id": rid, "answer_key": answer_key, "answer": answer,
                         "answer_source": "STORE_PROVIDED" if answer_key in store_answers else "UNKNOWN",
                         "next_branch": branch})
        if answer_key in store_answers and answer != "모름":
            actions.append({"rule_id": rid, "candidate": rule["conditional_action"],
                            "condition": branch, "status": "CONDITIONAL_CANDIDATE"})
        else:
            actions.append({"rule_id": rid, "candidate": rule["conditional_action"],
                            "condition": branch, "status": "PENDING_REQUIRED_ANSWER"})
            abstentions.append({"rule_id": rid, "reason": rule["abstention_condition"]})
        metrics.append({"rule_id": rid, "metric": rule["validation_metric"]})
    if not ids:
        checks.append({"rank": 1, "rule_id": "G_GENERAL", "observed_signal": "뚜렷한 규칙 신호 없음",
                       "priority_reason": "신호를 만들지 않고 월별 경로와 점포 기본정보를 확인한다."})
        questions.append({"rank": 1, "rule_id": "G_GENERAL",
                          "question": "해당 기간 영업일·영업시간·가격·상품구성에 변화가 있었습니까?"})
        abstentions.append({"rule_id": "G_GENERAL", "reason": "점포정보가 없어 지역 수준 진단으로 제한"})
    return {
        "method": "C0_RULE_TEMPLATE_NO_LLM",
        "case_id": card["case_id"],
        "confirmed_facts": {
            "period": card["period"], "amt_growth_pct_m": card["trend"]["amt_growth_pct_m"],
            "cnt_growth_pct_m": card["trend"]["cnt_growth_pct_m"],
            "ticket_growth_pct_m": card["trend"]["ticket_growth_pct_m"],
            "directions": {k: card["trend"][k] for k in ["amt_direction", "cnt_direction", "ticket_direction"]},
        },
        "triggered_rules": ids,
        "active_rules": active_ids,
        "deferred_rules": ids[MAX_QUESTIONS:],
        "prioritized_checks": checks,
        "followup_questions": questions,
        "answer_branches": branches,
        "conditional_improvement_candidates": actions,
        "validation_metrics": metrics,
        "abstentions": abstentions,
        "possible_explanations": [by_id[r]["candidate_explanations"] for r in active_ids],
        "prohibited_claims": [by_id[r]["prohibited_claims"] for r in active_ids],
        "limitations": card["limitations"],
        "first_priority_code": checks[0]["rule_id"],
        "source_card_hash": card["card_hash"],
    }


def load_cards(path: Path) -> dict[str, dict]:
    cards = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            card = json.loads(line)
            cards[card["case_id"]] = card
    return cards


def load_rules(path: Path) -> list[dict]:
    import csv
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))
