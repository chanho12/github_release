#!/usr/bin/env python3
"""Independent structural validator for a diagnosis-validation run."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run-dir",required=True); args=ap.parse_args(); run=Path(args.run_dir).resolve()
    required=["README.md","project_scope.md","execution_plan.md","artifact_manifest.csv","claim_corrections.csv","external_data_scope.csv","unresolved_data_issues.md","analysis_config.yaml","frozen_split.json","diagnosis_rules.csv","evidence_cards.jsonl","development_cases.csv","evaluation_cases.csv","sampling_protocol.md","scenario_registry.csv","conversation_protocol.md","evaluation_rubric.md","human_ratings_template.csv","execution_status.json","reports/final_validation_report.md","outputs/method_A_outputs.jsonl","outputs/method_C0_outputs.jsonl","outputs/machine_fact_checks.csv","outputs/neutral_band_sensitivity.csv","outputs/prototype_smoke_test.json"]
    rows=[]
    def add(t,s,v):rows.append({"test":t,"status":s,"value":v})
    missing=[x for x in required if not (run/x).exists()]; add("required_artifacts","PASS" if not missing else "FAIL",json.dumps(missing,ensure_ascii=False))
    cards=[json.loads(x) for x in (run/"evidence_cards.jsonl").read_text().splitlines() if x.strip()]; add("evidence_card_count","PASS" if len(cards)==2327 else "FAIL",len(cards))
    split=json.loads((run/"frozen_split.json").read_text()); dev={tuple(x) for x in split["development_regions"]}; eva={tuple(x) for x in split["evaluation_regions"]}; add("region_split","PASS" if not dev&eva else "FAIL",len(dev&eva))
    evaldf=pd.read_csv(run/"evaluation_cases.csv"); add("evaluation_n","PASS" if len(evaldf)==24 else "FAIL",len(evaldf)); add("evaluation_unique_regions","PASS" if len(evaldf)==len(evaldf[["SIDO_NM","CCG_NM"]].drop_duplicates()) else "FAIL",len(evaldf[["SIDO_NM","CCG_NM"]].drop_duplicates()))
    dataset_hash=(run/"analysis_dataset_v1.sha256").read_text().split()[0]; add("dataset_hash","PASS" if dataset_hash==sha(run/"analysis_dataset_v1.csv") else "FAIL",dataset_hash)
    report=(run/"reports/final_validation_report.md").read_text(); refs=re.findall(r"\[[^]]+\]\(([^)]+)\)",report); broken=[r for r in refs if not r.startswith("http") and not (run/"reports"/r).resolve().exists()]; add("report_links","PASS" if not broken else "FAIL",json.dumps(broken,ensure_ascii=False))
    status=json.loads((run/"execution_status.json").read_text()); add("truthful_execution_status","PASS" if status["B"]=="NOT_RUN_API" and status["C"]=="NOT_RUN_API" and status["human_evaluation"]=="AWAITING_HUMAN_EVAL" else "FAIL",json.dumps(status,ensure_ascii=False))
    text="\n".join((run/x).read_text(errors="ignore") for x in required if (run/x).is_file())
    secrets=re.findall(r"(?i)(?:api[_-]?key|secret|bearer)\s*[:=]\s*[A-Za-z0-9_-]{12,}",text); add("secret_scan","PASS" if not secrets else "FAIL",len(secrets))
    fact=pd.read_csv(run/"outputs/machine_fact_checks.csv")
    fact_cols=[c for c in fact.columns if c!="case_id"]
    fact_ok=bool(fact[fact_cols].all(axis=None)); add("machine_fact_checks","PASS" if fact_ok else "FAIL",int(fact_ok))
    sensitivity=pd.read_csv(run/"outputs/neutral_band_sensitivity.csv")
    add("neutral_band_sensitivity","PASS" if sensitivity.neutral_band_pct_m.tolist()==[0.0,0.25,0.5] else "FAIL",json.dumps(sensitivity.neutral_band_pct_m.tolist()))
    d=pd.DataFrame(rows); d.to_csv(run/"outputs/package_validation_results.csv",index=False,encoding="utf-8-sig")
    old=pd.read_csv(run/"artifact_manifest.csv")
    manifest=old[old.role.eq("FROZEN_INPUT")].copy()
    outrows=[]
    for p in sorted(run.rglob("*")):
        if not p.is_file() or p.name=="artifact_manifest.csv": continue
        status="EXECUTED"
        if p.name=="human_ratings_template.csv": status="AWAITING_HUMAN_EVAL"
        if p.name in {"method_B_general_llm.md","method_C_procedure_llm.md"}: status="NOT_RUN_API"
        project_root=run.parents[2]
        outrows.append({"path":str(p.relative_to(project_root)),"role":"OUTPUT","sha256":sha(p),"status":status,"linked_code":"validate_diagnosis_package.py"})
    pd.concat([manifest,pd.DataFrame(outrows)],ignore_index=True).to_csv(run/"artifact_manifest.csv",index=False,encoding="utf-8-sig")
    print(d.to_string(index=False)); raise SystemExit(1 if (d.status=="FAIL").any() else 0)


if __name__=="__main__":main()
