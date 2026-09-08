#!/usr/bin/env python3
"""Aggregate real blind ratings; never imputes blank ratings as zero."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCORES=["evidence_alignment_1_5","check_relevance_1_5","priority_validity_1_5","unsupported_claims_1_5","uncertainty_scope_1_5","conditional_action_fit_1_5","understandability_1_5","operational_realism_1_5","question_burden_1_5"]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run-dir",required=True); ap.add_argument("--ratings",required=True); args=ap.parse_args()
    run=Path(args.run_dir).resolve(); ratings=pd.read_csv(args.ratings,dtype=str).replace("",np.nan)
    observed=ratings.dropna(subset=["reviewer_id"])
    if observed.empty:
        result={"status":"AWAITING_HUMAN_EVAL","rated_rows":0,"message":"Blank template is not a zero score and no human result was generated."}
        (run/"outputs/human_evaluation_status.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(result,ensure_ascii=False)); return
    key=pd.read_csv(run/"blind_order_key_internal.csv")
    d=observed.merge(key,on=["packet_id","output_label"],validate="many_to_one")
    for c in SCORES+["decision_time_seconds"]: d[c]=pd.to_numeric(d[c],errors="coerce")
    long=[]
    for c in SCORES+["decision_time_seconds"]:
        for (role,method),g in d.groupby(["reviewer_role","method"]):
            long.append({"metric":c,"role":role,"method":method,"n_ratings":g[c].notna().sum(),"mean":g[c].mean(),"median":g[c].median()})
    summary=pd.DataFrame(long); summary.to_csv(run/"outputs/human_evaluation_summary.csv",index=False,encoding="utf-8-sig")
    # Paired differences at case-reviewer level; no claim of independent reviewer rows.
    paired=[]
    for c in SCORES+["decision_time_seconds"]:
        w=d.pivot_table(index=["case_id","reviewer_id","reviewer_role"],columns="method",values=c,aggfunc="first").dropna(subset=["A","C0"])
        if len(w): paired.append({"metric":c,"paired_case_reviewer_n":len(w),"C0_minus_A_mean":(w.C0-w.A).mean(),"C0_minus_A_median":(w.C0-w.A).median()})
    pd.DataFrame(paired).to_csv(run/"outputs/human_paired_differences.csv",index=False,encoding="utf-8-sig")
    result={"status":"EXECUTED","rated_rows":len(d),"unique_reviewers":d.reviewer_id.nunique(),"warning":"Small internal paired review; not field impact."}
    (run/"outputs/human_evaluation_status.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
