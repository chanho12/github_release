#!/usr/bin/env python3
"""Minimal CLI prototype: region/industry -> evidence -> questions -> candidates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from diagnosis_engine import baseline_output, c0_output, load_cards, load_rules


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-dir",required=True)
    ap.add_argument("--list",action="store_true")
    ap.add_argument("--region",help="시도 시군구 또는 case_id의 앞 두 부분")
    ap.add_argument("--industry",help="업종코드 또는 업종명")
    ap.add_argument("--method",choices=["A","C0"],default="C0")
    ap.add_argument("--answers-json",default="{}",help="실제/시뮬레이션 답변 JSON. 출처를 별도 기록할 것")
    args=ap.parse_args()
    run=Path(args.run_dir).resolve(); cards=load_cards(run/"evidence_cards.jsonl")
    if args.list:
        for c in list(cards.values())[:50]: print(c["case_id"])
        print(f"총 {len(cards)}개. --region과 --industry로 선택하세요.")
        return
    if not args.region or not args.industry:
        ap.error("--list 또는 --region/--industry가 필요합니다")
    matched=[]
    for c in cards.values():
        region=f"{c['region']['sido']} {c['region']['sigungu']}"
        ind=c["industry"]
        if args.region in {region,c["case_id"],f"{c['region']['sido']}|{c['region']['sigungu']}"} and args.industry in {str(ind['code']),ind['name']}:
            matched.append(c)
    if len(matched)!=1:
        raise SystemExit(f"정확히 1개가 필요하지만 {len(matched)}개가 일치했습니다")
    card=matched[0]
    if args.method=="A": result=baseline_output(card)
    else:
        answers=json.loads(args.answers_json); result=c0_output(card,load_rules(run/"diagnosis_rules.csv"),answers)
    envelope={"prototype_mode":"RULE_TEMPLATE_NO_LLM" if args.method=="C0" else "GENERAL_DASHBOARD_NO_LLM",
              "personalization_status":"REGION_DIAGNOSIS_ONLY" if args.answers_json=="{}" else "STORE_ANSWERS_SUPPLIED_NOT_VERIFIED",
              "period":"2026-01~2026-06","evidence_card":card,"diagnosis":result,
              "warning":"개별 점포 원인·효과·폐업확률·예상 매출증가율을 제공하지 않습니다."}
    print(json.dumps(envelope,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
