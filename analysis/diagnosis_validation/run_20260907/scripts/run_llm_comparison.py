#!/usr/bin/env python3
"""Provider-neutral B/C comparison runner.

No call occurs unless both --adapter-command and --confirm-api-run are supplied.
The adapter receives one JSON request on stdin and must return one JSON response
on stdout. The same command, model id, budgets and Evidence Card are used for B/C.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run-dir",required=True); ap.add_argument("--status-only",action="store_true")
    ap.add_argument("--adapter-command"); ap.add_argument("--confirm-api-run",action="store_true")
    ap.add_argument("--model-id",default="UNSPECIFIED"); ap.add_argument("--max-output-tokens",type=int,default=1200); ap.add_argument("--question-budget",type=int,default=3)
    args=ap.parse_args(); run=Path(args.run_dir).resolve(); status_path=run/"execution_status.json"
    status=json.loads(status_path.read_text())
    if args.status_only or not (args.adapter_command and args.confirm_api_run):
        print(json.dumps({"B":status.get("B","NOT_RUN_API"),"C":status.get("C","NOT_RUN_API"),"reason":"Provide an authorized adapter command and --confirm-api-run to execute."},ensure_ascii=False,indent=2)); return
    if args.model_id=="UNSPECIFIED": raise SystemExit("--model-id must be fixed before B/C execution")
    cards=read_jsonl(run/"outputs/shared_evaluation_inputs.jsonl"); logs=[]
    common=(run/"prompts/shared_constraints.md").read_text()
    prompts={"B":(run/"prompts/method_B_general_llm.md").read_text(),"C":(run/"prompts/method_C_procedure_llm.md").read_text()}
    rules=(run/"diagnosis_rules.csv").read_text()
    cmd=shlex.split(args.adapter_command)
    for card in cards:
        for method in ["B","C"]:
            req={"model":args.model_id,"max_output_tokens":args.max_output_tokens,"question_budget":args.question_budget,
                 "system":common,"method_prompt":prompts[method]+("\n\nRULES:\n"+rules if method=="C" else ""),"evidence_card":card}
            start=time.perf_counter(); now=datetime.now(timezone.utc).isoformat()
            try:
                p=subprocess.run(cmd,input=json.dumps(req,ensure_ascii=False),text=True,capture_output=True,timeout=180,check=False)
                latency=time.perf_counter()-start
                response=json.loads(p.stdout) if p.returncode==0 else None
                logs.append({"case_id":card["case_id"],"method":method,"model":args.model_id,"called_at":now,"latency_seconds":latency,
                             "returncode":p.returncode,"response":response,"stderr":p.stderr[-2000:],"cost":response.get("cost") if isinstance(response,dict) else None,
                             "usage":response.get("usage") if isinstance(response,dict) else None})
            except Exception as exc:
                logs.append({"case_id":card["case_id"],"method":method,"model":args.model_id,"called_at":now,"error":type(exc).__name__+": "+str(exc),"response":None})
    path=run/"outputs/llm_B_C_execution_log.jsonl"
    with path.open("w",encoding="utf-8") as f:
        for x in logs:f.write(json.dumps(x,ensure_ascii=False)+"\n")
    ok={m:sum(x.get("response") is not None for x in logs if x["method"]==m) for m in ["B","C"]}
    status["B"]="EXECUTED" if ok["B"]==len(cards) else "FAILED"; status["C"]="EXECUTED" if ok["C"]==len(cards) else "FAILED"
    status["llm_configuration"]={"adapter_command":cmd[0],"model_id":args.model_id,"max_output_tokens":args.max_output_tokens,"question_budget":args.question_budget}
    status_path.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps({"cases":len(cards),"successful":ok,"log":str(path)},ensure_ascii=False,indent=2))


if __name__=="__main__": main()
