# Diagnosis validation run

Run: `run_20260907`

## Reproduce

```bash
python3 scripts/build_diagnosis_validation.py --out analysis/diagnosis_validation/run_20260907
python3 scripts/validate_diagnosis_package.py --run-dir analysis/diagnosis_validation/run_20260907
python3 scripts/diagnosis_cli.py --run-dir analysis/diagnosis_validation/run_20260907 --list
```

LLM adapter and human aggregation:

```bash
python3 scripts/run_llm_comparison.py --run-dir analysis/diagnosis_validation/run_20260907 --status-only
python3 scripts/aggregate_human_evaluation.py --run-dir analysis/diagnosis_validation/run_20260907 --ratings analysis/diagnosis_validation/run_20260907/human_ratings_template.csv
```

## Status

- Data/version freeze: EXECUTED
- Evidence Cards (2,327): EXECUTED
- Development/evaluation split: EXECUTED
- A and C0: EXECUTED
- B and C: NOT_RUN_API
- Automated validation: EXECUTED
- Human evaluation: AWAITING_HUMAN_EVAL
- Prototype CLI: EXECUTED

기존 run과 원자료를 덮어쓰지 않는다. `blind_order_key_internal.csv`는 평가 중 평가자에게 제공하지 않는다.
