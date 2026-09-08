#!/usr/bin/env python3
"""Build the reproducible diagnosis-validation package.

Inputs are frozen raw BC data and the verified 2026-09-07 audit run. No new
external data, API call, model call, or human rating is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

from diagnosis_engine import baseline_output, c0_output


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "analysis/audit_same_sales_different_actions/run_20260907"
RKEY = ["SIDO_NM", "CCG_NM"]
MONTHS = list(range(202601, 202607))
SEED = 20260907
EPS = 0.25
SENSITIVITY_EPS = [0.0, 0.25, 0.5]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def direction(v: float, eps: float = EPS) -> str:
    return "상승" if v > eps else "감소" if v < -eps else "정체"


def log_metrics(y) -> dict:
    y = np.asarray(y, float); x = np.arange(len(y))
    fit = linregress(x, np.log(y))
    return {"beta": float(fit.slope), "growth_pct_m": float(np.expm1(fit.slope)*100), "r2": float(fit.rvalue**2),
            "raw_slope": float(linregress(x, y).slope)}


def concentration_panel(bc: pd.DataFrame, domestic_only: bool) -> pd.DataFrame:
    q = bc[bc.AGE_CD.ne("x")].copy()
    if domestic_only:
        q = q[q.GENDER_CD.isin(["1", "2"])]
    q = q.groupby(RKEY + ["TP_BUZ_NO", "STRD_YYMM", "AGE_CD"], as_index=False).cnt.sum()
    q["share"] = q.cnt / q.groupby(RKEY + ["TP_BUZ_NO", "STRD_YYMM"]).cnt.transform("sum")
    m = (q.assign(hhi=lambda d:d.share**2, ent=lambda d:-d.share*np.log(d.share))
         .groupby(RKEY + ["TP_BUZ_NO", "STRD_YYMM"], as_index=False)
         .agg(hhi=("hhi","sum"), entropy=("ent","sum")))
    m.entropy /= math.log(6)
    rows=[]
    for key,g in m.groupby(RKEY+["TP_BUZ_NO"]):
        if g.STRD_YYMM.nunique()!=6: continue
        g=g.sort_values("STRD_YYMM")
        rows.append({**dict(zip(RKEY+["TP_BUZ_NO"],key)),
                     "age_hhi_slope_domestic" if domestic_only else "age_hhi_slope_all":linregress(range(6),g.hhi).slope,
                     "age_entropy_slope_domestic" if domestic_only else "age_entropy_slope_all":linregress(range(6),g.entropy).slope})
    return pd.DataFrame(rows)


def build_rules() -> pd.DataFrame:
    fields=["rule_id","trigger","required_fields","applicability_gate","observed_fact_template","candidate_explanations",
            "priority_rationale","followup_question","answer_key","answer_branch_json","conditional_action","validation_metric",
            "abstention_condition","prohibited_claims"]
    rows=[
      ["A_CNT","cnt_direction == 감소","monthly_CNT;cnt_growth","6개월 완전관측",
       "결제건수 감소 신호가 관찰됨","영업일 감소;결제분할 변화;고유고객수 또는 구매빈도 변화",
       "금액보다 실제 거래 흐름을 먼저 확인해야 건당결제액 상승에 가려진 변화를 구분할 수 있다.",
       "같은 기간 실제 영업일과 점포 주문건수, 가능하면 고유고객수는 어떻게 변했습니까?","store_cnt_context",
       json.dumps({"영업일감소":"영업일 보정 후 다시 비교","고유고객감소":"재방문·신규고객 경로 확인","빈도감소":"구매주기·편의성 확인","변화없음":"BC 가맹점/결제수단 구성 확인","모름":"원인 판단 유보"},ensure_ascii=False),
       "확인된 원인 한 가지를 대상으로 2~4주 소규모 운영·재방문 시험","점포 주문건수;고유고객수;재방문율;영업일당 CNT",
       "영업일·점포 주문·고유고객 자료가 없으면 고객 이탈 원인을 판단하지 않는다.","고객 이탈 확정;재방문 감소 확정;광고비 즉시 증액"],
      ["B_TICKET","ticket_direction == 감소","monthly_AMT;monthly_CNT;monthly_ticket","AMT와 CNT 양수",
       "건당 평균 결제금액 감소 신호가 관찰됨","할인;저가상품 비중;주문구성;결제분할;고객구성 변화",
       "가격 인상 전에 금액 하락이 가격·수량·상품믹스 중 어디에서 왔는지 확인해야 한다.",
       "가격, 할인, 상품·주문구성, 결제방식 중 분석기간에 바뀐 항목이 있습니까?","ticket_context",
       json.dumps({"가격하락":"마진과 수량을 함께 확인","할인증가":"할인 유입과 재구매를 확인","저가상품증가":"교차판매 가능성 확인","결제분할":"주문단위 지표로 재계산","변화없음":"고객·가맹점 구성 확인","모름":"원인 판단 유보"},ensure_ascii=False),
       "원인이 확인된 경우 구성·번들·할인 중 하나만 작은 범위에서 시험","건당결제액;상품당 마진;구매수량;재구매율",
       "상품·가격·할인 자료가 없으면 가격 또는 수익성 원인을 판단하지 않는다.","가격 하락 확정;수익 악화 확정;즉시 가격 인상"],
      ["C_TOTAL_INDUSTRY","region_total_direction != industry_direction","region_total;industry_trend","동일 기간·제공 11업종 범위",
       "지역총량과 해당 업종의 방향이 다름","업종별 계절성;지역 내 업종 전환;전국 공통 업종 흐름",
       "지역총량을 점포 업종에 적용하기 전에 해당 업종 고유 흐름을 분리해야 한다.",
       "점포의 실제 업종·주력상품은 BC 연결업종과 일치하며, 다른 업종 매출원이 섞여 있습니까?","industry_scope_match",
       json.dumps({"일치":"업종 CNT·건단가를 우선 적용","혼합업종":"매출원을 분리해 재검토","불일치":"현재 업종 진단 적용 중단","모름":"업종 적용 판단 유보"},ensure_ascii=False),
       "업종 범위가 일치할 때 해당 업종의 CNT·건당결제액 개선 후보만 검토","업종별 AMT/CNT;주력상품 비중",
       "실제 업종 범위를 확인하지 못하면 개별 점포로 일반화하지 않는다.","지역 성장=해당 점포 성장;기존 서비스 오류율"],
      ["D_CONCENTRATION","age or industry concentration up","age shares;industry shares","고정 구성범위·절대건수 병기",
       "연령 또는 업종 소비 비중의 집중 방향이 관찰됨","특정 집단 절대성장;다른 집단 정체;구성 범위 변화",
       "비중만으로 위험을 판정하지 않고 절대 건수와 점포 목표고객의 관련성을 확인한다.",
       "집중이 커진 집단의 실제 고객수·상품수요가 늘었습니까, 다른 집단이 줄었습니까?","concentration_context",
       json.dumps({"집중집단증가":"유지와 분산의 목적을 구분","타집단감소":"이탈 구간·상품 적합성 확인","둘다":"절대규모와 비중을 함께 추적","무관":"점포 처방에서 제외","모름":"원인 판단 유보"},ensure_ascii=False),
       "점포 목표와 관련성이 확인된 경우에만 타깃 유지 또는 고객층 확장 실험","연령별 고유고객;절대 CNT;상품별 전환율",
       "연령코드·고객수·점포 관련성이 불명확하면 위험판정을 유보한다.","고령화 위험 확정;다양성 붕괴 확정;주민 구성과 동일"],
      ["E_SUPPLY","supply up and CNT non-growth","external supply;mapping","MODULE_ONLY이고 6개월 유효",
       "외부 공급 증가와 BC 결제건수 비성장이 함께 관찰됨","인허가 재분류;실제 경쟁점 증가;BC 가맹점 모집단 변화",
       "수요 대응보다 공급이 빠른지 보기 전에 외부 사업자와 BC 가맹점 모집단 차이를 확인한다.",
       "실제 영업권 내 경쟁점 수·개폐점과 BC 가맹점 포함범위가 자료와 일치합니까?","supply_context",
       json.dumps({"실제경쟁증가":"상권·상품 차별성 확인","재분류":"공급 신호 제외","BC범위변화":"BC 추세 해석 유보","변화없음":"외부자료 매핑 재검토","모름":"과잉경쟁 판단 유보"},ensure_ascii=False),
       "실제 경쟁증가가 확인될 때만 상품·시간대·고객층 차별화 시험","실제 경쟁점;상권 내 주문;점유율;시험 전후 CNT",
       "매핑·시점·모집단이 확인되지 않으면 과잉경쟁을 판단하지 않는다.","과잉경쟁 확정;개별 점포 평균매출 하락 확정"],
      ["F_PEAK","peak_dependent == true","monthly_AMT;leave-one-month-out","6개월 완전관측",
       "특정 월 제거 시 AMT 성장 방향이 중립 이하로 바뀜","행사;영업일;관측범위;대량결제;계절성",
       "장기 방향을 논하기 전에 한 달 민감도와 관측범위 변화를 먼저 확인한다.",
       "피크월에 행사·휴점·영업일·대량주문·결제수단 범위 변화가 있었습니까?","peak_context",
       json.dumps({"행사":"행사 제외·반복 가능성을 분리","대량주문":"일반 영업과 분리","영업일변화":"영업일당 지표로 재검토","범위변화":"추세 판단 중단","변화없음":"더 긴 기간 확인","모름":"장기 추세 판단 유보"},ensure_ascii=False),
       "원인이 반복 가능한 경우에만 같은 조건의 소규모 재현 시험","비행사월 AMT/CNT;영업일당 지표;12개월 지속성",
       "피크월 원인과 12개월 자료가 없으면 장기 성장·쇠퇴 판단을 유보한다.","이벤트 성공 확정;장기 쇠퇴 확정;피크 원인 확정"],
    ]
    return pd.DataFrame(rows,columns=fields)


def build_cards(bc: pd.DataFrame, panel: pd.DataFrame, region: pd.DataFrame, q4: pd.DataFrame, stores: pd.DataFrame) -> tuple[list[dict],pd.DataFrame]:
    monthly=bc.groupby(RKEY+["TP_BUZ_NO","TP_BUZ_NM","STRD_YYMM"],as_index=False)[["amt","cnt"]].sum()
    national=monthly.groupby(["TP_BUZ_NO","STRD_YYMM"],as_index=False)[["amt","cnt"]].sum()
    nat_rows=[]
    for ind,g in national.groupby("TP_BUZ_NO"):
        g=g.sort_values("STRD_YYMM"); a,c=log_metrics(g.amt),log_metrics(g.cnt)
        nat_rows.append({"TP_BUZ_NO":ind,"national_amt_growth_pct_m":a["growth_pct_m"],"national_cnt_growth_pct_m":c["growth_pct_m"]})
    nat=pd.DataFrame(nat_rows)
    all_age=concentration_panel(bc,False); dom_age=concentration_panel(bc,True)
    age=all_age.merge(dom_age,on=RKEY+["TP_BUZ_NO"],how="outer")
    reg_comp=region[RKEY+["hhi_slope","entropy_slope","industry_concentration_up","concentration_increase"]]
    store_cols=RKEY+["연결업종","confidence","external_source","source_scope","supply_growth_pct_m","demand_supply_gap_cnt","h1_type"]
    smap=stores[store_cols].copy()
    living_path=AUDIT/"reproduction/fourth_external_integration/02_living_population_monthly.csv"
    living=pd.read_csv(living_path)
    living_regions=set(map(tuple,living[RKEY].drop_duplicates().to_numpy()))
    name_norm={"편 의 점":"편의점","슈퍼 마켓":"슈퍼마켓","중국음식":"중국음식","제 과 점":"제과점"}
    base=(panel.merge(q4[RKEY+["TP_BUZ_NO","region_total_amt_growth_pct_m","region_total_direction","direction_divergence"]],on=RKEY+["TP_BUZ_NO"],validate="one_to_one")
          .merge(nat,on="TP_BUZ_NO").merge(age,on=RKEY+["TP_BUZ_NO"],how="left").merge(reg_comp,on=RKEY,how="left"))
    base["연결업종"]=base.TP_BUZ_NM.map(name_norm)
    base=base.merge(smap,on=RKEY+["연결업종"],how="left")
    cards=[]; index_rows=[]
    for _,r in base.iterrows():
        g=monthly[(monthly.SIDO_NM==r.SIDO_NM)&(monthly.CCG_NM==r.CCG_NM)&(monthly.TP_BUZ_NO==r.TP_BUZ_NO)].sort_values("STRD_YYMM")
        # Region share within national same industry; percentage-point slope.
        n=national[national.TP_BUZ_NO==r.TP_BUZ_NO].sort_values("STRD_YYMM")
        share_amt=g.amt.to_numpy()/n.amt.to_numpy()*100; share_cnt=g.cnt.to_numpy()/n.cnt.to_numpy()*100
        ma,mc,mt=log_metrics(g.amt),log_metrics(g.cnt),log_metrics(g.amt/g.cnt)
        supply_app=pd.notna(r.external_source)
        ext={"applicable":bool(supply_app),"source":str(r.external_source) if supply_app else None,
             "mapping_confidence":str(r.confidence) if supply_app else None,"source_scope":str(r.source_scope) if supply_app else None,
             "supply_growth_pct_m":float(r.supply_growth_pct_m) if supply_app and pd.notna(r.supply_growth_pct_m) else None,
             "supply_direction":direction(float(r.supply_growth_pct_m)) if supply_app and pd.notna(r.supply_growth_pct_m) else None,
             "limitation":"외부 사업자/인허가와 BC 가맹점은 동일 모집단이 아님" if supply_app else "해당 업종에 실행 가능한 HIGH 공급모듈 없음"}
        classification="같은방향"
        if r.region_total_direction!=r.amt_direction:
            classification="정반대" if {r.region_total_direction,r.amt_direction}=={"상승","감소"} else "정체포함차이"
        card={
          "case_id":f"{r.SIDO_NM}|{r.CCG_NM}|{int(r.TP_BUZ_NO)}","region":{"sido":r.SIDO_NM,"sigungu":r.CCG_NM},
          "industry":{"code":int(r.TP_BUZ_NO),"name":r.TP_BUZ_NM},"period":"2026-01~2026-06",
          "source_ids":["BC_RAW","AUDIT_RUN_20260907"]+([str(r.external_source)] if supply_app else []),
          "monthly":{"unit":{"amt":"원","cnt":"건","ticket":"원/건"},"values":[{"yyyymm":int(x.STRD_YYMM),"amt":int(x.amt),"cnt":int(x.cnt),"ticket":float(x.amt/x.cnt)} for _,x in g.iterrows()]},
          "trend":{"method":"log OLS, x=0..5","amt_beta":ma["beta"],"cnt_beta":mc["beta"],"ticket_beta":mt["beta"],
                   "log_identity_error":ma["beta"]-mc["beta"]-mt["beta"],"amt_growth_pct_m":ma["growth_pct_m"],
                   "cnt_growth_pct_m":mc["growth_pct_m"],"ticket_growth_pct_m":mt["growth_pct_m"],
                   "amt_direction":direction(ma["growth_pct_m"]),"cnt_direction":direction(mc["growth_pct_m"]),"ticket_direction":direction(mt["growth_pct_m"]),
                   "neutral_band":"±0.25%/월, 실무적 설정이며 유의성 기준 아님","amt_robust_sign":bool(r.amt_robust_sign),
                   "cnt_robust_sign":bool(r.cnt_robust_sign),"peak_month":int(r.peak_month),"peak_dependent":bool(r.peak_dependent),
                   "amt_growth_without_peak_pct_m":float(r.amt_growth_without_peak_pct_m)},
          "national_same_industry":{"national_amt_growth_pct_m":float(r.national_amt_growth_pct_m),"national_cnt_growth_pct_m":float(r.national_cnt_growth_pct_m),
                                    "amt_share_slope_pp_m":float(linregress(range(6),share_amt).slope),"cnt_share_slope_pp_m":float(linregress(range(6),share_cnt).slope),
                                    "denominator":"전국 동일 BC업종 월별 AMT/CNT"},
          "region_total_vs_industry":{"region_total_amt_growth_pct_m":float(r.region_total_amt_growth_pct_m),"region_total_direction":r.region_total_direction,
                                      "industry_direction":r.amt_direction,"classification":classification,
                                      "scope":"제공된 11개 BC 업종 합계, 전체 지역경제 아님"},
          "composition":{"age_hhi_slope_all_numeric":None if pd.isna(r.age_hhi_slope_all) else float(r.age_hhi_slope_all),
                         "age_entropy_slope_all_numeric":None if pd.isna(r.age_entropy_slope_all) else float(r.age_entropy_slope_all),
                         "age_hhi_slope_domestic_personal":None if pd.isna(r.age_hhi_slope_domestic) else float(r.age_hhi_slope_domestic),
                         "age_entropy_slope_domestic_personal":None if pd.isna(r.age_entropy_slope_domestic) else float(r.age_entropy_slope_domestic),
                         "age_concentration_up_all_numeric":bool((r.age_hhi_slope_all>0)&(r.age_entropy_slope_all<0)),
                         "age_concentration_up_domestic_personal":bool((r.age_hhi_slope_domestic>0)&(r.age_entropy_slope_domestic<0)),
                         "industry_hhi_slope_fixed7":None if pd.isna(r.hhi_slope) else float(r.hhi_slope),
                         "industry_entropy_slope_fixed7":None if pd.isna(r.entropy_slope) else float(r.entropy_slope),
                         "industry_concentration_up":bool(r.industry_concentration_up),
                         "age_warning":"외국인 GENDER_CD=3도 숫자 연령코드에 포함됨; 국내개인 민감도 별도. 연령코드 1·2 구간은 문서상 불명확."},
          "external_modules":{"supply":ext,"population":{"applicable":r.CCG_NM not in ["화성시 동탄구","화성시 만세구","화성시 병점구","화성시 효행구"],"classification":"CONTROL","limitation":"주민과 결제자는 동일 모집단이 아님"},
                              "living_population":{"applicable":(r.SIDO_NM,r.CCG_NM) in living_regions,"classification":"MODULE_ONLY","period":"2026-01~03","limitation":"89개 인구감소지역만; Q1 공통계절성 가능"}},
          "coverage":{"months":6,"complete_region_industry":True,"total_cnt_6m":int(r.total_cnt_6m),"total_amt_6m":int(r.total_amt_6m)},
          "mapping_confidence":ext["mapping_confidence"],
          "missing_information":["점포 실제 영업일·영업시간","점포 주문건수·고유고객수·재방문","상품·수량·가격·할인·마진","BC 가맹점 포함범위","개선 실행 전후 성과"],
          "limitations":["CNT는 결제건수이며 고객수/방문자수 아님","AMT/CNT는 건당 평균 결제금액이며 가격/이익 아님","6개월로 계절성과 장기 지속성 분리 불가","지역×업종 결과를 개별 점포 원인으로 일반화 불가"],
          "trace":{"raw_file":"dataset/ABP_CONTEST_DATA.csv","group_key":{"SIDO_NM":r.SIDO_NM,"CCG_NM":r.CCG_NM,"TP_BUZ_NO":int(r.TP_BUZ_NO)},"months":MONTHS},
        }
        card["card_hash"]=canonical_hash(card); cards.append(card)
        index_rows.append({"case_id":card["case_id"],"SIDO_NM":r.SIDO_NM,"CCG_NM":r.CCG_NM,"TP_BUZ_NO":int(r.TP_BUZ_NO),"TP_BUZ_NM":r.TP_BUZ_NM,
                           "amt_direction":card["trend"]["amt_direction"],"total_cnt_6m":int(r.total_cnt_6m),"supply_module":bool(supply_app),
                           "age_sensitivity_disagrees":card["composition"]["age_concentration_up_all_numeric"]!=card["composition"]["age_concentration_up_domestic_personal"],"card_hash":card["card_hash"]})
    return cards,pd.DataFrame(index_rows)


def freeze_samples(index: pd.DataFrame, dev: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,dict,pd.DataFrame]:
    dev_regions=set(map(tuple,dev[RKEY].drop_duplicates().to_numpy()))
    development=index.merge(dev[RKEY+["TP_BUZ_NO"]].drop_duplicates(),on=RKEY+["TP_BUZ_NO"],how="inner")
    pool=index[~index[RKEY].apply(tuple,axis=1).isin(dev_regions)].copy()
    pool["volume_group"]=np.where(pool.total_cnt_6m>=100_000,"충분","소규모")
    pool["external_group"]=np.where(pool.supply_module,"공급모듈","BC만")
    pool["stratum"]=pool.amt_direction+"|"+pool.volume_group+"|"+pool.external_group
    rng=np.random.default_rng(SEED); selected=[]; used=set(); stats=[]
    # Target two per 3x2x2 stratum. Shortages are filled later, preserving unique regions.
    for stratum,g in pool.groupby("stratum",sort=True):
        order=rng.permutation(g.index)
        chosen=[]
        for idx in order:
            reg=tuple(pool.loc[idx,RKEY])
            if reg in used: continue
            chosen.append(idx); used.add(reg)
            if len(chosen)==2: break
        selected.extend(chosen); stats.append({"stratum":stratum,"eligible_cases":len(g),"target_n":2,"selected_initial_n":len(chosen)})
    target=24
    remaining=pool.loc[~pool.index.isin(selected)].copy()
    for idx in rng.permutation(remaining.index):
        if len(selected)>=target: break
        reg=tuple(pool.loc[idx,RKEY])
        if reg in used: continue
        selected.append(idx); used.add(reg)
    evaluation=pool.loc[selected].copy().sort_values(["amt_direction","case_id"]).reset_index(drop=True)
    counts=evaluation.stratum.value_counts(); eligible=pool.stratum.value_counts()
    evaluation["stratum_eligible_n"]=evaluation.stratum.map(eligible)
    evaluation["stratum_selected_n"]=evaluation.stratum.map(counts)
    evaluation["conditional_inclusion_fraction"]=evaluation.stratum_selected_n/evaluation.stratum_eligible_n
    development=development.assign(sample_role="DEVELOPMENT_EXPLANATION")
    evaluation=evaluation.assign(sample_role="INTERNAL_EVALUATION")
    frozen={"seed":SEED,"split_unit":"SIDO_NM+CCG_NM","development_regions":sorted([list(x) for x in dev_regions]),
            "evaluation_regions":sorted(evaluation[RKEY].drop_duplicates().values.tolist()),"evaluation_case_ids":evaluation.case_id.tolist(),
            "target_n":24,"actual_n":len(evaluation),"thresholds_frozen_before_output_review":{"neutral_band_pct_m":EPS,"volume_sufficient_cnt_6m":100000},
            "note":"전체 데이터는 기존 EDA에 사용됨; 규칙개발과 분리한 내부 평가이며 외부검증 아님"}
    return development,evaluation,frozen,pd.DataFrame(stats)


def render_output(out: dict) -> str:
    if out["method"].startswith("A_"):
        lines=["확인된 사실"]+[f"- {x}" for x in out["confirmed_facts"]]+["","우선 점검"]+[f"- {i}. {x}" for i,x in enumerate(out["checklist"],1)]
        lines += ["","추가 질문"]+[f"- {x}" for x in out["followup_questions"]]
        lines += ["","조건부 개선 후보"]+[f"- [PENDING_REQUIRED_ANSWER] {x}" for x in out["conditional_improvement_candidates"]]
        lines += ["","판단 유보"]+[f"- {x}" for x in out["abstentions"]]
    else:
        facts=out["confirmed_facts"]
        lines=["확인된 사실",f"- AMT {facts['amt_growth_pct_m']:+.2f}%/월, CNT {facts['cnt_growth_pct_m']:+.2f}%/월, 건당결제액 {facts['ticket_growth_pct_m']:+.2f}%/월.","","우선 점검"]
        lines += [f"- {x['rank']}. {x['observed_signal']} — {x['priority_reason']}" for x in out["prioritized_checks"]]
        lines += ["","추가 질문"]+[f"- {x['question']}" for x in out["followup_questions"]]
        lines += ["","조건부 개선 후보"]+[f"- [{x['status']}] {x['candidate']}" for x in out["conditional_improvement_candidates"]]
        lines += ["","판단 유보"]+[f"- {x['reason']}" for x in out["abstentions"]]
    return "\n".join(lines)


def render_shared_card(card:dict) -> str:
    monthly=["| 월 | AMT(원) | CNT(건) | 건당 결제금액(원/건) |","|---:|---:|---:|---:|"]
    for x in card["monthly"]["values"]:
        monthly.append(f"| {x['yyyymm']} | {x['amt']:,.0f} | {x['cnt']:,.0f} | {x['ticket']:,.0f} |")
    t=card["trend"]; n=card["national_same_industry"]; rv=card["region_total_vs_industry"]; comp=card["composition"]; supply=card["external_modules"]["supply"]
    lines=["## 공통 Evidence Card",*monthly,"",
           f"- 기간/방법: {card['period']}, {t['method']}; 방향 중립구간 {t['neutral_band']}",
           f"- 추세: AMT {t['amt_growth_pct_m']:+.2f}%/월({t['amt_direction']}), CNT {t['cnt_growth_pct_m']:+.2f}%/월({t['cnt_direction']}), 건당 결제금액 {t['ticket_growth_pct_m']:+.2f}%/월({t['ticket_direction']})",
           f"- 전국 동일업종: AMT {n['national_amt_growth_pct_m']:+.2f}%/월, CNT {n['national_cnt_growth_pct_m']:+.2f}%/월; 지역 점유율 기울기 AMT {n['amt_share_slope_pp_m']:+.4f}%p/월, CNT {n['cnt_share_slope_pp_m']:+.4f}%p/월",
           f"- 지역총량 대비: 제공 11업종 지역총량 AMT {rv['region_total_amt_growth_pct_m']:+.2f}%/월({rv['region_total_direction']}), 해당 업종과 {rv['classification']}",
           f"- 구성: 전체 숫자연령 집중 증가={comp['age_concentration_up_all_numeric']}, 국내개인 민감도 집중 증가={comp['age_concentration_up_domestic_personal']}, 고정 7업종 집중 증가={comp['industry_concentration_up']}",
           f"- 공급모듈: 적용={supply['applicable']}, 출처={supply.get('source')}, 매핑={supply.get('mapping_confidence')}, 한계={supply['limitation']}",
           "- 해석 제한: CNT는 고객수가 아닌 결제건수이며, 건당 결제금액은 가격·마진이 아니다. 지역×업종 결과를 개별 점포 원인으로 일반화하지 않는다."]
    return "\n".join(lines)


def comparisons(cards_by_id:dict,rules:list[dict],evaluation:pd.DataFrame,out:Path) -> tuple[list[dict],list[dict],pd.DataFrame]:
    shared=[]; aouts=[]; c0outs=[]
    for cid in evaluation.case_id:
        card=cards_by_id[cid]; shared.append(card); aouts.append(baseline_output(card)); c0outs.append(c0_output(card,rules))
    write_jsonl(out/"outputs/shared_evaluation_inputs.jsonl",shared)
    write_jsonl(out/"outputs/method_A_outputs.jsonl",aouts); write_jsonl(out/"outputs/method_C0_outputs.jsonl",c0outs)
    rows=[]
    for a,c in zip(aouts,c0outs):
        rows.append({"case_id":a["case_id"],"A_first_priority":a["first_priority_code"],"C0_first_priority":c["first_priority_code"],
                     "C0_trigger_count":len(c["triggered_rules"]),"C0_question_count":len(c["followup_questions"]),
                     "C0_pending_action_count":sum(x["status"]=="PENDING_REQUIRED_ANSWER" for x in c["conditional_improvement_candidates"]),
                     "first_priority_differs":a["first_priority_code"]!=c["first_priority_code"]})
    return aouts,c0outs,pd.DataFrame(rows)


def automated_tests(cards:list[dict],evaluation:pd.DataFrame,dev:pd.DataFrame,rules:list[dict],aouts:list[dict],c0outs:list[dict],out:Path) -> pd.DataFrame:
    tests=[]
    def add(test,status,value,meaning): tests.append({"test":test,"status":status,"value":value,"meaning":meaning})
    max_identity=max(abs(c["trend"]["log_identity_error"]) for c in cards)
    add("log_identity", "PASS" if max_identity<1e-12 else "FAIL",max_identity,"beta_AMT=beta_CNT+beta_ticket")
    add("card_count","PASS" if len(cards)==2327 else "FAIL",len(cards),"완전 지역×업종 카드 수")
    devregs=set(map(tuple,dev[RKEY].drop_duplicates().to_numpy())); evalregs=set(map(tuple,evaluation[RKEY].drop_duplicates().to_numpy()))
    add("region_split_leakage","PASS" if not devregs&evalregs else "FAIL",len(devregs&evalregs),"개발·평가 시군구 중복")
    add("evaluation_unique_regions","PASS" if len(evalregs)==len(evaluation) else "FAIL",len(evalregs),"평가사례별 고유 지역")
    c0repeat=[c0_output(c,rules) for c in [next(x for x in cards if x['case_id']==cid) for cid in evaluation.case_id]]
    add("C0_repeat_determinism","PASS" if canonical_hash({"x":c0outs})==canonical_hash({"x":c0repeat}) else "FAIL",canonical_hash({"x":c0outs}),"동일 입력 반복")
    eval_cards=[next(x for x in cards if x['case_id']==cid) for cid in evaluation.case_id]
    exact=all(a["source_card_hash"]==c["card_hash"] and z["source_card_hash"]==c["card_hash"] for a,z,c in zip(aouts,c0outs,eval_cards))
    add("source_card_binding","PASS" if exact else "FAIL",int(exact),"A/C0가 동일 카드 해시에 결합")
    numeric_exact=all(
        abs(z["confirmed_facts"]["amt_growth_pct_m"]-c["trend"]["amt_growth_pct_m"])<1e-12 and
        abs(z["confirmed_facts"]["cnt_growth_pct_m"]-c["trend"]["cnt_growth_pct_m"])<1e-12 and
        abs(z["confirmed_facts"]["ticket_growth_pct_m"]-c["trend"]["ticket_growth_pct_m"])<1e-12
        for z,c in zip(c0outs,eval_cards)
    )
    add("C0_numeric_fact_consistency","PASS" if numeric_exact else "FAIL",int(numeric_exact),"C0 수치가 Evidence Card와 정확히 일치")
    directions_ok=all(c["trend"][f"{m}_direction"]==direction(c["trend"][f"{m}_growth_pct_m"],EPS) for c in cards for m in ["amt","cnt","ticket"])
    add("direction_threshold_consistency","PASS" if directions_ok else "FAIL",int(directions_ok),"±0.25%/월 방향 규칙 재계산")
    budget_ok=all(len(x["followup_questions"])<=3 for x in c0outs) and all(len(x["followup_questions"])<=3 for x in aouts)
    add("question_budget","PASS" if budget_ok else "FAIL",max([len(x["followup_questions"]) for x in aouts+c0outs]),"A/C0 질문 최대 3개")
    missing_supply=[c for c in cards if not c["external_modules"]["supply"]["applicable"]]
    invalid_supply=sum("E_SUPPLY" in c0_output(c,rules)["triggered_rules"] for c in missing_supply)
    add("missing_module_abstention","PASS" if invalid_supply==0 else "FAIL",invalid_supply,"공급자료 없을 때 공급 규칙 비활성")
    text="\n".join(render_output(x) for x in aouts+c0outs)
    secret=bool(re.search(r"(?i)(api[_-]?key|secret|bearer)\s*[:=]\s*[A-Za-z0-9_-]{12,}",text))
    add("secret_scan","PASS" if not secret else "FAIL",int(secret),"출력의 비밀키 패턴")
    prohibited=["폐업확률","예상 매출증가율","고객 이탈 확정","과잉경쟁 확정"]
    hits=sum(text.count(x) for x in prohibited)
    add("unsupported_claim_keyword_review","REVIEW" if hits else "PASS",hits,"자동 키워드 검사는 사람평가 대체 아님")
    add("LLM_B_execution","NOT_RUN_API","no adapter/model/key/budget authorized","프롬프트·실행기만 작성")
    add("LLM_C_execution","NOT_RUN_API","no adapter/model/key/budget authorized","프롬프트·실행기만 작성")
    add("human_evaluation","AWAITING_HUMAN_EVAL",0,"점수·만족도·시간을 생성하지 않음")
    return pd.DataFrame(tests)


def machine_fact_checks(cards_by_id:dict,aouts:list[dict],c0outs:list[dict]) -> pd.DataFrame:
    rows=[]
    for a,c0 in zip(aouts,c0outs):
        c=cards_by_id[a["case_id"]]
        rows.append({
            "case_id":a["case_id"],
            "same_card_hash":a["source_card_hash"]==c0["source_card_hash"]==c["card_hash"],
            "amt_exact":abs(c0["confirmed_facts"]["amt_growth_pct_m"]-c["trend"]["amt_growth_pct_m"])<1e-12,
            "cnt_exact":abs(c0["confirmed_facts"]["cnt_growth_pct_m"]-c["trend"]["cnt_growth_pct_m"])<1e-12,
            "ticket_exact":abs(c0["confirmed_facts"]["ticket_growth_pct_m"]-c["trend"]["ticket_growth_pct_m"])<1e-12,
            "direction_exact":all(c["trend"][f"{m}_direction"]==direction(c["trend"][f"{m}_growth_pct_m"],EPS) for m in ["amt","cnt","ticket"]),
            "question_budget_ok":len(a["followup_questions"])<=3 and len(c0["followup_questions"])<=3,
            "source_trace_present":bool(c.get("trace",{}).get("raw_file")) and len(c.get("trace",{}).get("months",[]))==6,
        })
    return pd.DataFrame(rows)


def neutral_band_sensitivity(panel:pd.DataFrame,q4:pd.DataFrame) -> pd.DataFrame:
    x=panel.merge(q4[RKEY+["TP_BUZ_NO","region_total_amt_growth_pct_m"]],on=RKEY+["TP_BUZ_NO"],validate="one_to_one")
    rows=[]
    for eps in SENSITIVITY_EPS:
        ind=x.amt_growth_pct_m.map(lambda v:direction(v,eps)); reg=x.region_total_amt_growth_pct_m.map(lambda v:direction(v,eps))
        mismatch=ind.ne(reg); opposite=((ind=="상승")&(reg=="감소"))|((ind=="감소")&(reg=="상승"))
        rows.append({"neutral_band_pct_m":eps,"n":len(x),"direction_mismatch_n":int(mismatch.sum()),
                     "exact_opposite_n":int(opposite.sum()),"neutral_involved_mismatch_n":int((mismatch&~opposite).sum()),
                     "region_up_industry_down_n":int(((reg=="상승")&(ind=="감소")).sum()),
                     "amt_up_cnt_nonup_n":int(((x.amt_growth_pct_m>eps)&(x.cnt_growth_pct_m<=eps)).sum())})
    return pd.DataFrame(rows)


def development_pair_review(dev_details:pd.DataFrame,cards_by_id:dict,rules:list[dict]) -> pd.DataFrame:
    rows=[]
    for pid,g in dev_details.groupby("pair_id"):
        outs=[]
        for _,r in g.sort_values("side").iterrows():
            cid=f"{r.SIDO_NM}|{r.CCG_NM}|{int(r.TP_BUZ_NO)}"; outs.append(c0_output(cards_by_id[cid],rules))
        qsets=[tuple(x["question"] for x in o["followup_questions"]) for o in outs]
        orders=[tuple(x["rule_id"] for x in o["prioritized_checks"]) for o in outs]
        rows.append({"pair_id":pid,"first_check_differs":outs[0]["first_priority_code"]!=outs[1]["first_priority_code"],
                     "additional_questions_differ":qsets[0]!=qsets[1],"full_order_differs":orders[0]!=orders[1],
                     "interpretation":"차이가 없어도 실패가 아님; 동일 점검이 합리적일 수 있음"})
    return pd.DataFrame(rows)


def write_static_docs(out:Path,inputs:list[Path],claim:pd.DataFrame,scope:pd.DataFrame,dev:pd.DataFrame,evaluation:pd.DataFrame,stats:pd.DataFrame) -> None:
    (out/"project_scope.md").write_text("""# Project scope\n\n주제는 「같은 매출 변화, 다른 처방」이다. 검증 대상은 동일 Evidence Card에서 구조화된 규칙·질문 절차가 일반 지표표보다 점검을 더 명시적으로 조직하는지다. 개별 점포 원인, 폐업예측, 매출개선, 처방효과는 범위 밖이다.\n\n- 데이터 기간: 2026년 1~6월\n- 분석 단위: 지역×BC 업종\n- 개발 사례: 기존 설명용 4쌍\n- 내부 평가: 개발 지역을 제외한 24개 실제 사례\n- LLM·사람평가: 권한/평가자 미확보 시 실행하지 않음\n""",encoding="utf-8")
    (out/"execution_plan.md").write_text("""# Execution plan and recorded status\n\n1. 기존 감사 run과 원자료 확인 — EXECUTED\n2. 주장 수정·입력 해시 고정 — EXECUTED\n3. Evidence Card·진단규칙 구현 — EXECUTED\n4. 개발/평가 시군구 분리 — EXECUTED\n5. A·C0 동일입력 비교 — EXECUTED\n6. B·C 동일 LLM 실행 — NOT_RUN_API\n7. 자동 검증 — EXECUTED\n8. 익명 사람평가 패키지 — AWAITING_HUMAN_EVAL\n9. CLI 시제품·최종보고서 — EXECUTED\n""",encoding="utf-8")
    lines=["# Unresolved data issues","","- CNT로 고유고객수·방문·재방문을 구분할 수 없다.","- 건당 결제금액으로 상품가격·수량·마진을 구분할 수 없다.","- 외국인이 숫자 연령코드에 포함되며 연령코드 1·2의 정확한 경계가 불명확하다.","- 주민 인구와 지역 결제자 모집단은 다르다.","- 공급자료는 BC 가맹점과 동일 모집단이 아니고 HIGH는 업종명 대응 신뢰도일 뿐이다.","- 생활인구는 89개 인구감소지역·1~3월만 있다.","- 실제 점주정보·개선 실행·성과자료가 없다."]
    (out/"unresolved_data_issues.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    yaml=f"""run_id: {out.name}\nseed: {SEED}\nperiod: [202601, 202602, 202603, 202604, 202605, 202606]\nneutral_band_pct_month: {EPS}\nsensitivity_neutral_bands_pct_month: {SENSITIVITY_EPS}\nevaluation_target_n: 24\nsplit_unit: SIDO_NM+CCG_NM\nminimum_complete_months: 6\nllm_status: NOT_RUN_API\nhuman_evaluation_status: AWAITING_HUMAN_EVAL\n"""
    (out/"analysis_config.yaml").write_text(yaml,encoding="utf-8")
    (out/"sampling_protocol.md").write_text(f"""# Sampling protocol\n\n- Seed: {SEED}\n- 규칙·임계값 고정 후 평가출력을 생성했다.\n- 기존 개발 4쌍의 8개 시군구를 평가 pool에서 제외했다.\n- 평가 단위는 지역×업종이며 24개 서로 다른 시군구를 선택했다.\n- 층화축은 AMT 방향(±{EPS}%/월 중립), 6개월 CNT 10만 기준, HIGH 공급모듈 가용 여부다.\n- 각 층에서 최대 2개를 무작위 추출한 뒤 부족분을 남은 pool에서 채웠다.\n- 층별 모집단·선정수·조건부 포함비율은 evaluation_cases.csv에 있다.\n- 이는 이미 EDA에 사용된 전체 데이터 안의 내부 분리이며 외부검증이나 전국 발생률 추정 표본이 아니다.\n\n{stats.to_csv(index=False)}\n""",encoding="utf-8")
    (out/"conversation_protocol.md").write_text("""# Conversation protocol\n\n## Step 1\n모든 방식에 동일 Evidence Card와 품질정보를 제공한다. A는 지표표+체크리스트, B는 일반 LLM, C는 동일 LLM+진단절차, C0는 규칙만 사용한다.\n\n## Step 2\nB와 C의 질문예산은 최대 3개로 맞춘다. 실제 익명 점포답변이 없으면 모두 `모름`이다. 시뮬레이션 답변은 scenario_registry에만 두고 실제 지역 발견에 합치지 않는다.\n\n## Step 3\n답변 출처를 STORE_PROVIDED / SIMULATED / UNKNOWN으로 기록하고 조건부 개선 후보를 갱신한다. 숫자·원인·성과를 새로 생성하지 않는다.\n""",encoding="utf-8")
    (out/"evaluation_rubric.md").write_text("""# Blind human evaluation rubric\n\n방식명은 숨기고 사례별 두 출력을 같은 평가자가 짝지어 본다. 출력 순서는 seed로 무작위화했다. 우리 규칙의 문구를 정답으로 사용하지 않으며 복수의 합리적 순서를 허용한다.\n\n전문가(1~5): 근거 일치, 점검항목 적절성, 우선순위 타당성, 원인단정 억제, 불확실성, 조건부 후보 적합성.\n점주(1~5): 이해도, 현실성, 질문 부담. 실제 소요시간(초)은 직접 측정한다.\n\n최소 2명 이상을 권장한다. 사례를 독립 표본으로 두고 평가자 반복을 지역표본으로 세지 않는다. 출력 확인 전에 rubric을 고정한다. 현재 상태는 AWAITING_HUMAN_EVAL이다.\n""",encoding="utf-8")


def prompts_and_runner_docs(out:Path) -> None:
    prompts=out/"prompts"; prompts.mkdir(exist_ok=True)
    shared="""동일 Evidence Card만 사용한다. 숫자를 재계산하거나 원인을 만들지 않는다. 확인된 사실/가능한 설명/미확인 정보/다음 행동/조건부 개선 후보를 구분한다. CNT는 결제건수, AMT/CNT는 건당 평균 결제금액이다. 질문은 최대 3개다. 개선효과·폐업확률·예상 매출증가율을 생성하지 않는다. JSON으로 답한다.\n"""
    (prompts/"shared_constraints.md").write_text(shared,encoding="utf-8")
    (prompts/"method_A_spec.md").write_text("# A\n\n동일 지표와 품질정보를 표로 제공하고 합리적인 일반 체크리스트를 제시한다. LLM 없음.\n",encoding="utf-8")
    (prompts/"method_B_general_llm.md").write_text("# B general LLM\n\n"+shared+"일반적인 상권 상담가로서 우선 점검, 추가 질문, 답변별 조건부 후보를 제시하라. 별도 진단 규칙은 제공하지 않는다.\n",encoding="utf-8")
    (prompts/"method_C_procedure_llm.md").write_text("# C procedure-guided LLM\n\n"+shared+"diagnosis_rules.csv의 적용 gate와 우선순위, 질문 분기를 따르되 자료에 맞지 않는 규칙은 유보하라. 같은 점검이 적절하면 억지로 다르게 만들지 말라.\n",encoding="utf-8")
    (prompts/"method_C0_spec.md").write_text("# C0\n\n진단 규칙과 템플릿만 사용하며 LLM을 사용하지 않는다. 실제 답변이 없으면 후보를 PENDING_REQUIRED_ANSWER로 둔다.\n",encoding="utf-8")
    (prompts/"method_name_mapping.md").write_text("""# Name mapping\n\n- 과거 B0 일반 지표표 → 이번 A\n- 이번 B → 동일 카드의 일반 LLM\n- 이번 C → 동일 카드+진단절차의 동일 LLM\n- 이번 C0 → 진단규칙만, LLM 없음\n- 과거 B0E와 사람 비교는 미실시이며 이번 blind A/C0 패키지로 준비만 함\n""",encoding="utf-8")


def blind_packets(out:Path,evaluation:pd.DataFrame,aouts:list[dict],c0outs:list[dict],cards_by_id:dict) -> pd.DataFrame:
    folder=out/"blind_review_packets"; folder.mkdir(exist_ok=True)
    rng=random.Random(SEED); maprows=[]; ratingrows=[]
    for i,(a,c) in enumerate(zip(aouts,c0outs),1):
        order=[("A",a),("C0",c)]; rng.shuffle(order)
        labels=[]
        for j,(method,obj) in enumerate(order,1):
            label=f"Output {j}"; labels.append((label,method,obj)); maprows.append({"packet_id":f"P{i:03d}","output_label":label,"method":method,"case_id":a["case_id"]})
        case=evaluation.iloc[i-1]
        text=f"# Blind packet P{i:03d}\n\nCase: {case.TP_BUZ_NM}, 방향층 {case.amt_direction}. 지역명은 평가 맥락상 표시하되 방식명은 숨김: {case.SIDO_NM} {case.CCG_NM}.\n\n{render_shared_card(cards_by_id[a['case_id']])}\n\n"
        for label,_,obj in labels: text+=f"## {label}\n\n{render_output(obj)}\n\n"
        text+="평가 점수는 human_ratings_template.csv에 입력한다.\n"
        (folder/f"packet_{i:03d}.md").write_text(text,encoding="utf-8")
        for label,_,_ in labels:
            for role in ["EXPERT","OWNER"]:
                ratingrows.append({"packet_id":f"P{i:03d}","output_label":label,"reviewer_id":"","reviewer_role":role,
                                   "evidence_alignment_1_5":"","check_relevance_1_5":"","priority_validity_1_5":"","unsupported_claims_1_5":"",
                                   "uncertainty_scope_1_5":"","conditional_action_fit_1_5":"","understandability_1_5":"","operational_realism_1_5":"",
                                   "question_burden_1_5":"","decision_time_seconds":"","comments":""})
    pd.DataFrame(ratingrows).to_csv(out/"human_ratings_template.csv",index=False,encoding="utf-8-sig")
    mapping=pd.DataFrame(maprows); mapping.to_csv(out/"blind_order_key_internal.csv",index=False,encoding="utf-8-sig")
    return mapping


def scenario_registry(out:Path,cards_by_id:dict,rules:list[dict],evaluation:pd.DataFrame) -> pd.DataFrame:
    cid=evaluation.case_id.iloc[0]; scenarios=[
      ("S00",{},"UNKNOWN","모든 답변 모름"),("S01",{"store_cnt_context":"영업일감소"},"SIMULATED","영업일 감소 가상분기"),
      ("S02",{"ticket_context":"할인증가"},"SIMULATED","할인 증가 가상분기"),("S03",{"industry_scope_match":"불일치"},"SIMULATED","업종 불일치 가상분기"),
      ("S04",{"concentration_context":"무관"},"SIMULATED","점포와 집중 무관 가상분기"),("S05",{"peak_context":"범위변화"},"SIMULATED","관측범위 변화 가상분기")]
    rows=[]; logs=[]
    for sid,ans,src,note in scenarios:
        result=c0_output(cards_by_id[cid],rules,ans); rows.append({"scenario_id":sid,"base_case_id":cid,"answer_source":src,"answers_json":json.dumps(ans,ensure_ascii=False),"purpose":note,"is_real_store_discovery":False})
        logs.append({"scenario_id":sid,"result":result})
    pd.DataFrame(rows).to_csv(out/"scenario_registry.csv",index=False,encoding="utf-8-sig"); write_jsonl(out/"outputs/scenario_branch_logs.jsonl",logs)
    return pd.DataFrame(rows)


def final_report(out:Path,claims:pd.DataFrame,scope:pd.DataFrame,auto:pd.DataFrame,dev_review:pd.DataFrame,compare:pd.DataFrame,evaluation:pd.DataFrame,sensitivity:pd.DataFrame) -> None:
    def md(d): return d.to_markdown(index=False) if False else markdown(d)
    report=f"""# 「같은 매출 변화, 다른 처방」 진단 구현·비교검증 보고서

## 1. 검증 결론

동일 Evidence Card에서 규칙 기반 C0는 일반 지표표 A보다 **적용 규칙, 추가 질문, 판단 유보 조건, 답변 분기, 확인 지표를 재현 가능하게 구조화했다.** 그러나 이 구조가 사람의 판단을 실제로 개선하는지는 아직 입증되지 않았다. B·C LLM은 API·모델·예산이 지정되지 않아 `NOT_RUN_API`, 익명 사람평가는 `AWAITING_HUMAN_EVAL`이다.

따라서 현재 주제는 지역×업종 수준의 **점검 지원 시제품**까지 방어 가능하다. 맞춤형 개선효과·폐업예측·개별 점포 원인 확정은 방어할 수 없다.

## 2. 기존 주장 재현·수정

{md(claims)}

679개 불일치는 정반대 496개와 정체 포함 차이 183개다. 지역총량 상승·업종 감소 432개는 전체 2,327조합의 18.56%, 지역총량 상승 2,027조합의 21.31%이며 지역 수 비율이 아니다. 기존 4쌍은 구조 차이가 큰 개발 사례이지 성능표본이 아니다.

## 3. 데이터와 외부자료 범위

{md(scope)}

BC 242,574행, 255개 시군구쌍, 11개 업종, 6개월과 완전 지역×업종 2,327개를 고정했다. 외국인 70,178행이 숫자 연령코드를 사용하므로 전체 숫자연령과 국내개인(GENDER 1·2) 집중도 민감도를 Evidence Card에 함께 넣었다. 어느 쪽도 주민 구성과 동일 집단으로 해석하지 않는다.

## 4. Evidence Card와 진단 절차

2,327개 카드에는 월별 AMT·CNT·건당 결제금액, 로그 slope와 항등식 오차, 강건성, 전국 동일업종 점유율, 지역총량과 업종 관계, 전체 숫자연령/국내개인 연령집중 민감도, 고정 7업종 집중도, 적용 가능한 공급모듈과 한계, 원자료 키를 저장했다.

규칙은 `F 피크민감도 → C 지역총량/업종 차이 → A CNT → B 건당결제액 → D 구성집중 → E 공급` 순이다. 각 규칙은 관찰, 필요한 점포정보, 질문, 답변분기, 조건부 후보, 확인지표, 유보조건, 금지단정을 갖는다. 원인이 확인되지 않으면 후보 상태는 `PENDING_REQUIRED_ANSWER`다.

## 5. 개발·평가 분리

- 개발: 기존 4쌍 8개 지역×업종.
- 평가: 개발지역을 제외하고 AMT 방향·거래규모·공급모듈 가용성으로 층화한 24개 실제 사례, 모두 서로 다른 시군구.
- 전체 데이터는 과거 EDA에 사용됐으므로 ‘규칙 개발과 분리한 내부 평가’이지 외부검증이 아니다.
- 평가사례를 본 뒤 규칙·±0.25% 임계값을 바꾸지 않았다. 0·0.5%는 민감도만 기록한다.

중립구간 민감도:

{md(sensitivity)}

±0.25%에서 679개였던 방향 불일치는 중립구간 정의에 따라 달라진다. 따라서 679를 자연적 경계나 위험률로 사용하지 않는다.

## 6. 방식별 실제 실행

| 방식 | 동일 카드 | 상태 | 이번에 확인한 것 |
|---|---:|---|---|
| A 일반 지표표+체크리스트 | 예 | EXECUTED | 합리적 공통 점검목록 제공 |
| B 일반 LLM | 예 | NOT_RUN_API | 프롬프트·어댑터 실행기만 작성 |
| C 진단절차+동일 LLM | 예 | NOT_RUN_API | 프롬프트·어댑터 실행기만 작성 |
| C0 규칙·템플릿 | 예 | EXECUTED | 사례별 우선규칙·질문·유보·분기 생성 |

자동검증:

{md(auto)}

A와 C0 모두 같은 카드 해시에 묶였고 숫자를 새로 만들지 않았다. C0의 반복 출력은 동일했다. 이는 소프트웨어 정합성 증거이지 점검 적절성의 사람 검증이 아니다.

## 7. 차이가 난 사례·같은 사례·유보

개발 4쌍의 분리 판정:

{md(dev_review)}

이번 개발 사례에서 Pair 2·4는 첫 점검뿐 아니라 추가 질문과 전체 순서도 같았다. Pair 1·3은 세 항목이 모두 달랐다. 같은 결과도 합리적일 수 있으므로 ‘4/4 성공’을 계산하지 않았다. 평가 24건에서 A의 첫 항목은 공통 검토이고 C0는 관측 신호별 우선규칙을 냈지만, 어느 순서가 더 적절한지는 사람 평가 전에는 미확인이다.

{md(compare.head(24))}

공급모듈이 없으면 E 규칙은 작동하지 않았고, 점포 답변이 없으면 개선 후보는 대기 상태로 남았다. 시뮬레이션 분기 5개는 코드 경로 시험용이며 실제 지역·점포 발견이 아니다.

## 8. 최소 시제품

`diagnosis_cli.py`는 지역·업종을 선택하면 분석기간과 Evidence Card를 확인하고 A 또는 C0 지역 진단을 출력한다. 점포답변 JSON을 선택적으로 받을 수 있지만 답변이 없으면 지역 진단 모드로 명시한다. 폐업확률·예상 매출증가율은 출력하지 않는다.

## 9. 추가 가치 판정

- **데이터 분해:** 지역총량과 업종 방향, CNT와 건당결제액 상쇄, 전체/국내개인 연령민감도를 한 카드에서 구분했다. 구현 완료.
- **진단 규칙:** 동일 정보에서 적용 gate·질문·유보·분기를 결정적으로 생성했다. 구현 및 자동 재현성 확인.
- **LLM:** B/C를 실행하지 않았으므로 추가 가치 미입증. C0만으로 충분한지 역시 사람평가 전에는 판단 보류.
- **사업효과:** 점주·POS·현장실험이 없어 미입증.

## 10. 다음 검증

1. 최소 2명의 전문가와 점주 평가자가 blind packet을 평가한다.
2. 동일 모델·출력예산·질문예산을 고정한 B/C 호출 권한을 명시적으로 제공한다.
3. 실제 동의받은 익명 점포의 영업일·주문·고유고객·상품/가격 자료를 입력한다.
4. 평가 완료 후 paired 사례 단위 집계를 실행한다. 미측정값을 0점으로 바꾸지 않는다.

## 11. 주요 산출물과 재현 명령

- [Evidence Cards](../evidence_cards.jsonl), [진단 규칙](../diagnosis_rules.csv)
- [내부 평가사례](../evaluation_cases.csv), [블라인드 평가표](../human_ratings_template.csv)
- [A/C0 구조 비교](../outputs/A_vs_C0_structural_comparison.csv), [사례별 기계 검증](../outputs/machine_fact_checks.csv)
- [중립구간 민감도](../outputs/neutral_band_sensitivity.csv), [패키지 검증](../outputs/package_validation_results.csv)
- [재현 README](../README.md), [산출물 manifest](../artifact_manifest.csv)
- [분석 생성 코드](../scripts/build_diagnosis_validation.py), [진단 엔진](../scripts/diagnosis_engine.py), [CLI 시제품](../scripts/diagnosis_cli.py)

```bash
python3 scripts/build_diagnosis_validation.py --out analysis/diagnosis_validation/run_20260907
python3 scripts/run_llm_comparison.py --run-dir analysis/diagnosis_validation/run_20260907 --status-only
python3 scripts/aggregate_human_evaluation.py --run-dir analysis/diagnosis_validation/run_20260907 --ratings analysis/diagnosis_validation/run_20260907/human_ratings_template.csv
python3 scripts/validate_diagnosis_package.py --run-dir analysis/diagnosis_validation/run_20260907
```

분석일: 2026-09-07  
상태: A/C0 및 자동검증 완료, B/C `NOT_RUN_API`, 사람평가 `AWAITING_HUMAN_EVAL`.
"""
    (out/"reports/final_validation_report.md").write_text(report,encoding="utf-8")


def markdown(d:pd.DataFrame)->str:
    x=d.copy().replace({np.nan:""}); cols=list(map(str,x.columns)); lines=["| "+" | ".join(cols)+" |","|"+"|".join(["---"]*len(cols))+"|"]
    for row in x.astype(str).itertuples(index=False,name=None): lines.append("| "+" | ".join(v.replace("|","\\|").replace("\n"," ") for v in row)+" |")
    return "\n".join(lines)


def write_readme(out:Path,status:dict) -> None:
    rel=out.relative_to(ROOT)
    text=f"""# Diagnosis validation run\n\nRun: `{out.name}`\n\n## Reproduce\n\n```bash\npython3 scripts/build_diagnosis_validation.py --out {rel}\npython3 scripts/validate_diagnosis_package.py --run-dir {rel}\npython3 scripts/diagnosis_cli.py --run-dir {rel} --list\n```\n\nLLM adapter and human aggregation:\n\n```bash\npython3 scripts/run_llm_comparison.py --run-dir {rel} --status-only\npython3 scripts/aggregate_human_evaluation.py --run-dir {rel} --ratings {rel}/human_ratings_template.csv\n```\n\n## Status\n\n- Data/version freeze: EXECUTED\n- Evidence Cards (2,327): EXECUTED\n- Development/evaluation split: EXECUTED\n- A and C0: EXECUTED\n- B and C: NOT_RUN_API\n- Automated validation: EXECUTED\n- Human evaluation: AWAITING_HUMAN_EVAL\n- Prototype CLI: EXECUTED\n\n기존 run과 원자료를 덮어쓰지 않는다. `blind_order_key_internal.csv`는 평가 중 평가자에게 제공하지 않는다.\n"""
    (out/"README.md").write_text(text,encoding="utf-8")


def manifest(out:Path,input_paths:list[Path]) -> pd.DataFrame:
    rows=[]
    for p in input_paths:
        rows.append({"path":str(p.relative_to(ROOT)),"role":"FROZEN_INPUT","sha256":sha256(p),"status":"EXECUTED","linked_code":"build_diagnosis_validation.py"})
    for p in sorted(out.rglob("*")):
        if not p.is_file() or p.name=="artifact_manifest.csv": continue
        status="AWAITING_HUMAN_EVAL" if p.name=="human_ratings_template.csv" else "EXECUTED"
        if "prompts" in p.parts or p.name=="run_llm_comparison.py": status="NOT_RUN_API" if "method_B" in p.name or "method_C_" in p.name else status
        rows.append({"path":str(p.relative_to(ROOT)),"role":"OUTPUT","sha256":sha256(p),"status":status,"linked_code":"build_diagnosis_validation.py"})
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default="analysis/diagnosis_validation/run_20260907"); args=ap.parse_args()
    out=(ROOT/args.out).resolve();
    for sub in [out,out/"outputs",out/"reports",out/"scripts",out/"prompts",out/"blind_review_packets"]: sub.mkdir(parents=True,exist_ok=True)
    inputs=[ROOT/"dataset/ABP_CONTEST_DATA.csv",AUDIT/"final_report.md",AUDIT/"claim_reproduction.csv",AUDIT/"external_data_decisions.csv",AUDIT/"region_industry_audit_panel.csv",AUDIT/"region_growth_quality.csv",AUDIT/"region_total_vs_industry.csv",AUDIT/"case_pair_details.csv"]
    missing=[p for p in inputs if not p.exists()]
    if missing: raise FileNotFoundError("Missing frozen inputs: "+", ".join(map(str,missing)))
    bc=pd.read_csv(inputs[0]); panel=pd.read_csv(inputs[4]); region=pd.read_csv(inputs[5]); q4=pd.read_csv(inputs[6]); dev_details=pd.read_csv(inputs[7])
    stores=pd.read_csv(AUDIT/"reproduction/fourth_external_integration/04_market_store_metrics_combined.csv")
    # Version-fixed compact analysis dataset.
    dataset=panel.merge(q4[RKEY+["TP_BUZ_NO","region_total_amt_growth_pct_m","region_total_direction","direction_divergence"]],on=RKEY+["TP_BUZ_NO"],validate="one_to_one")
    dataset.to_csv(out/"analysis_dataset_v1.csv",index=False,encoding="utf-8-sig")
    (out/"analysis_dataset_v1.sha256").write_text(sha256(out/"analysis_dataset_v1.csv")+"  analysis_dataset_v1.csv\n",encoding="utf-8")
    rulesdf=build_rules(); rulesdf.to_csv(out/"diagnosis_rules.csv",index=False,encoding="utf-8-sig"); rules=rulesdf.to_dict("records")
    cards,index=build_cards(bc,panel,region,q4,stores); write_jsonl(out/"evidence_cards.jsonl",cards); index.to_csv(out/"evidence_card_index.csv",index=False,encoding="utf-8-sig")
    cards_by_id={c["case_id"]:c for c in cards}
    dev,evaluation,frozen,stats=freeze_samples(index,dev_details); dev.to_csv(out/"development_cases.csv",index=False,encoding="utf-8-sig"); evaluation.to_csv(out/"evaluation_cases.csv",index=False,encoding="utf-8-sig")
    frozen["analysis_dataset_sha256"]=sha256(out/"analysis_dataset_v1.csv"); frozen["rules_sha256"]=sha256(out/"diagnosis_rules.csv")
    (out/"frozen_split.json").write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding="utf-8")
    aouts,c0outs,compare=comparisons(cards_by_id,rules,evaluation,out); compare.to_csv(out/"outputs/A_vs_C0_structural_comparison.csv",index=False,encoding="utf-8-sig")
    dev_review=development_pair_review(dev_details,cards_by_id,rules); dev_review.to_csv(out/"outputs/development_pair_review.csv",index=False,encoding="utf-8-sig")
    scenario_registry(out,cards_by_id,rules,evaluation)
    auto=automated_tests(cards,evaluation,dev,rules,aouts,c0outs,out); auto.to_csv(out/"outputs/automatic_validation_results.csv",index=False,encoding="utf-8-sig")
    machine=machine_fact_checks(cards_by_id,aouts,c0outs); machine.to_csv(out/"outputs/machine_fact_checks.csv",index=False,encoding="utf-8-sig")
    sensitivity=neutral_band_sensitivity(panel,q4); sensitivity.to_csv(out/"outputs/neutral_band_sensitivity.csv",index=False,encoding="utf-8-sig")
    prototype={"status":"EXECUTED","mode":"REGION_DIAGNOSIS_ONLY","case_id":evaluation.case_id.iloc[0],"result":c0_output(cards_by_id[evaluation.case_id.iloc[0]],rules)}
    (out/"outputs/prototype_smoke_test.json").write_text(json.dumps(prototype,ensure_ascii=False,indent=2),encoding="utf-8")
    scope=pd.read_csv(AUDIT/"external_data_decisions.csv"); scope.to_csv(out/"external_data_scope.csv",index=False,encoding="utf-8-sig")
    old=pd.read_csv(AUDIT/"claim_reproduction.csv")
    cross=q4.groupby(["region_total_direction","amt_direction"]).size()
    corrections=pd.DataFrame([
      ["지역총량·업종 불일치 679", "679/2327", "정반대 496, 정체 포함 183", "EXECUTED", "region_total_vs_industry.csv"],
      ["지역총량 상승·업종 감소", "432", "432/2327=18.56%; 432/2027=21.31%; 조합 비율", "EXECUTED", "region_total_vs_industry.csv"],
      ["대표 사례 4쌍", "성능 수치로 사용 불가", "구조차이 상위 개발사례; Pair 1·3만 세 비교항목이 다르고 Pair 2·4는 모두 같음", "EXECUTED", "development_pair_review.csv"],
      ["원단위/로그 성장착시", "10/9", "별도 정의 유지; 다른 플래그와 합산 금지", "EXECUTED", "claim_reproduction.csv"],
      ["3방법 일치", "강건성", "동일 6개월 재사용; 독립 검증 아님", "EXECUTED", "claim_reproduction.csv"],
      ["within/mix 63.7/36.3", "기여율", "지역별 절대기여 비중 중앙값; 전국 총액 아님", "EXECUTED", "claim_reproduction.csv"],
      ["연령 구성", "숫자연령", "외국인 포함 전체와 국내개인 민감도 병기; 주민과 동일시 금지", "EXECUTED", "BC raw crosstab"],
    ],columns=["previous_claim","confirmed_basis","corrected_expression","status","evidence"])
    corrections.to_csv(out/"claim_corrections.csv",index=False,encoding="utf-8-sig")
    write_static_docs(out,inputs,corrections,scope,dev,evaluation,stats); prompts_and_runner_docs(out); blind_packets(out,evaluation,aouts,c0outs,cards_by_id)
    execution={"run_id":out.name,"data_freeze":"EXECUTED","evidence_cards":"EXECUTED","A":"EXECUTED","C0":"EXECUTED","B":"NOT_RUN_API","C":"NOT_RUN_API","llm_reason":"모델·API 어댑터·키·예산 미지정","human_evaluation":"AWAITING_HUMAN_EVAL","improvement_effect":"NOT_MEASURED","generated_at":"2026-09-07 Asia/Seoul"}
    (out/"execution_status.json").write_text(json.dumps(execution,ensure_ascii=False,indent=2),encoding="utf-8")
    final_report(out,corrections,scope,auto,dev_review,compare,evaluation,sensitivity); write_readme(out,execution)
    # Copy executable scripts into run package for exact provenance.
    for src in [ROOT/"scripts/build_diagnosis_validation.py",ROOT/"scripts/diagnosis_engine.py",ROOT/"scripts/diagnosis_cli.py",ROOT/"scripts/run_llm_comparison.py",ROOT/"scripts/aggregate_human_evaluation.py",ROOT/"scripts/validate_diagnosis_package.py"]:
        if src.exists(): (out/"scripts"/src.name).write_bytes(src.read_bytes())
    man=manifest(out,inputs); man.to_csv(out/"artifact_manifest.csv",index=False,encoding="utf-8-sig")
    print(json.dumps({"out":str(out),"cards":len(cards),"development":len(dev),"evaluation":len(evaluation),"auto":auto.status.value_counts().to_dict(),"B":"NOT_RUN_API","C":"NOT_RUN_API","human":"AWAITING_HUMAN_EVAL"},ensure_ascii=False,indent=2))


if __name__=="__main__": main()
