#!/usr/bin/env python3
"""Independent final-topic validation from BC raw data and project external files."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress, theilslopes


ROOT = Path(__file__).resolve().parents[1]
BC_PATH = ROOT / "dataset/ABP_CONTEST_DATA.csv"
MONTHS = list(range(202601, 202607))
RKEY = ["SIDO_NM", "CCG_NM"]
EPS = 0.25


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def direction(value: float, eps: float = EPS) -> str:
    return "상승" if value > eps else "감소" if value < -eps else "정체"


def log_trend(values, x=None) -> dict:
    y = np.asarray(values, float)
    xx = np.arange(len(y), dtype=float) if x is None else np.asarray(x, float)
    if len(y) < 2 or np.any(~np.isfinite(y)) or np.any(y <= 0):
        return {"beta": np.nan, "growth_pct_m": np.nan, "r2": np.nan, "theil_beta": np.nan}
    fit = linregress(xx, np.log(y))
    return {
        "beta": float(fit.slope),
        "growth_pct_m": float(np.expm1(fit.slope) * 100),
        "r2": float(fit.rvalue**2),
        "theil_beta": float(theilslopes(np.log(y), xx).slope),
    }


def slope(values) -> float:
    y = np.asarray(values, float)
    return float(linregress(np.arange(len(y)), y).slope)


def md(df: pd.DataFrame) -> str:
    x = df.copy().replace({np.nan: ""})
    cols = list(map(str, x.columns))
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in x.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(v.replace("|", "\\|").replace("\n", " ") for v in row) + " |")
    return "\n".join(lines)


def load_bc() -> pd.DataFrame:
    d = pd.read_csv(BC_PATH, dtype={"GENDER_CD": str, "AGE_CD": str})
    d["TP_BUZ_NM_CLEAN"] = d.TP_BUZ_NM.str.replace(" ", "", regex=False)
    return d


def audit_bc(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    expected = {"STRD_YYMM", "SIDO_NM", "CCG_NM", "GENDER_CD", "AGE_CD", "TP_BUZ_NO", "TP_BUZ_NM", "amt", "cnt"}
    months = sorted(d.STRD_YYMM.unique().tolist())
    regions = d[RKEY].drop_duplicates()
    ccg_multi = regions.groupby("CCG_NM").SIDO_NM.nunique()
    industries = d[["TP_BUZ_NO", "TP_BUZ_NM_CLEAN"]].drop_duplicates().sort_values("TP_BUZ_NO")
    grain = ["STRD_YYMM"] + RKEY + ["GENDER_CD", "AGE_CD", "TP_BUZ_NO"]
    duplicate_grain = int(d.duplicated(grain).sum())
    missing = d.isna().sum()
    checks = [
        ["필수 컬럼", "확인됨" if set(d.columns) >= expected else "틀림", len(expected), ", ".join([c for c in d.columns if c != "TP_BUZ_NM_CLEAN"])],
        ["폐업/개업/점포수 라벨 부재", "확인됨", 0, "해당 컬럼 없음"],
        ["공급점포수 부재", "확인됨", 0, "해당 컬럼 없음"],
        ["소액표본 silent suppression", "추정만 가능", int(d.cnt.min()), "cnt 1~10이 없고 최솟값 11이나 제공 규칙 문서 없음"],
        ["SIDO_NM+CCG_NM 복합키 필요", "확인됨", int((ccg_multi > 1).sum()), "동일 CCG_NM이 복수 시도에 등장"],
        ["2026.01~06 6개월만 존재", "확인됨" if months == MONTHS else "틀림", len(months), str(months)],
        ["전년동월 계절성 통제 불가", "확인됨", 0, "2025년 동월 부재"],
        ["11개 업종 한정", "확인됨" if len(industries) == 11 else "틀림", len(industries), ", ".join(industries.TP_BUZ_NM_CLEAN)],
        ["고객수/재방문/신규고객 부재", "확인됨", 0, "CNT는 결제건수"],
        ["원자료 결측", "확인됨" if missing.sum() == 0 else "틀림", int(missing.sum()), json.dumps(missing[missing > 0].to_dict(), ensure_ascii=False)],
        ["원자료 분석 grain 중복", "확인됨" if duplicate_grain == 0 else "틀림", duplicate_grain, "+".join(grain)],
        ["AMT/CNT 양수", "확인됨" if (d.amt.gt(0) & d.cnt.gt(0)).all() else "틀림", int((d.amt.le(0) | d.cnt.le(0)).sum()), "0 이하 행 수"],
    ]
    coverage = (d.groupby(["TP_BUZ_NO", "TP_BUZ_NM_CLEAN"] + RKEY).STRD_YYMM.nunique().reset_index(name="months")
                .groupby(["TP_BUZ_NO", "TP_BUZ_NM_CLEAN"]).agg(observed_regions=("months", "size"), complete6_regions=("months", lambda x: int((x == 6).sum()))).reset_index())
    coverage["theoretical_regions"] = len(regions)
    coverage["complete6_pct"] = coverage.complete6_regions / len(regions) * 100
    facts = {
        "rows": len(d), "columns": len(expected), "months": months, "regions": len(regions),
        "sidos": d.SIDO_NM.nunique(), "industries": len(industries), "duplicate_grain": duplicate_grain,
        "null_cells": int(missing.sum()), "min_cnt": int(d.cnt.min()), "cnt_1_10_rows": int(d.cnt.between(1, 10).sum()),
        "amt_multiple_10000_pct": float(d.amt.mod(10000).eq(0).mean() * 100),
        "ambiguous_ccg_names": int((ccg_multi > 1).sum()), "unique_ccg_only": d.CCG_NM.nunique(),
    }
    return pd.DataFrame(checks, columns=["limit_or_check", "judgement", "value", "evidence"]), coverage, facts


def build_core(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    monthly = d.groupby(RKEY + ["TP_BUZ_NO", "TP_BUZ_NM_CLEAN", "STRD_YYMM"], as_index=False)[["amt", "cnt"]].sum()
    monthly["ticket"] = monthly.amt / monthly.cnt
    complete_keys = monthly.groupby(RKEY + ["TP_BUZ_NO"]).STRD_YYMM.nunique()
    complete_keys = complete_keys[complete_keys == 6].index
    p = monthly.set_index(RKEY + ["TP_BUZ_NO"])
    p = p[p.index.isin(complete_keys)].reset_index()

    national = monthly.groupby(["TP_BUZ_NO", "TP_BUZ_NM_CLEAN", "STRD_YYMM"], as_index=False)[["amt", "cnt"]].sum()
    national["ticket"] = national.amt / national.cnt
    nat_rows = []
    for (code, name), g in national.groupby(["TP_BUZ_NO", "TP_BUZ_NM_CLEAN"]):
        g = g.sort_values("STRD_YYMM")
        row = {"TP_BUZ_NO": code, "TP_BUZ_NM_CLEAN": name}
        for col in ["amt", "cnt", "ticket"]:
            t = log_trend(g[col]); row[f"national_{col}_beta"] = t["beta"]; row[f"national_{col}_growth_pct_m"] = t["growth_pct_m"]
        nat_rows.append(row)
    nat = pd.DataFrame(nat_rows)

    region_month = d.groupby(RKEY + ["STRD_YYMM"], as_index=False)[["amt", "cnt"]].sum()
    region_rows = []
    for key, g in region_month.groupby(RKEY):
        g = g.sort_values("STRD_YYMM")
        a = log_trend(g.amt); c = log_trend(g.cnt)
        region_rows.append({**dict(zip(RKEY, key)), "region_amt_growth_pct_m": a["growth_pct_m"], "region_cnt_growth_pct_m": c["growth_pct_m"]})
    region = pd.DataFrame(region_rows)

    rows = []
    for key, g in p.groupby(RKEY + ["TP_BUZ_NO", "TP_BUZ_NM_CLEAN"]):
        g = g.sort_values("STRD_YYMM"); x = np.arange(6)
        a, c, t = (log_trend(g[col]) for col in ["amt", "cnt", "ticket"])
        peak_i = int(np.argmax(g.amt.to_numpy())); keep = x != peak_i
        no_peak = log_trend(g.amt.to_numpy()[keep], x[keep])
        half_beta = (np.log(g.amt.iloc[3:].mean()) - np.log(g.amt.iloc[:3].mean())) / 3
        signs = [np.sign(a["beta"]), np.sign(a["theil_beta"]), np.sign(half_beta)]
        row = {**dict(zip(RKEY + ["TP_BUZ_NO", "TP_BUZ_NM_CLEAN"], key)),
               "amt_beta": a["beta"], "cnt_beta": c["beta"], "ticket_beta": t["beta"],
               "amt_growth_pct_m": a["growth_pct_m"], "cnt_growth_pct_m": c["growth_pct_m"], "ticket_growth_pct_m": t["growth_pct_m"],
               "amt_r2": a["r2"], "cnt_r2": c["r2"], "ticket_r2": t["r2"],
               "amt_direction": direction(a["growth_pct_m"]), "cnt_direction": direction(c["growth_pct_m"]), "ticket_direction": direction(t["growth_pct_m"]),
               "identity_error": a["beta"] - c["beta"] - t["beta"],
               "total_amt_6m": int(g.amt.sum()), "total_cnt_6m": int(g.cnt.sum()),
               "peak_month": int(g.STRD_YYMM.iloc[peak_i]), "amt_without_peak_growth_pct_m": no_peak["growth_pct_m"],
               "peak_dependent": direction(a["growth_pct_m"]) == "상승" and direction(no_peak["growth_pct_m"]) != "상승",
               "amt_three_method_same_sign": abs(sum(signs)) == 3,
               "amt_residual_sd": float(np.std(np.log(g.amt) - (linregress(x, np.log(g.amt)).intercept + a["beta"] * x), ddof=1))}
        rows.append(row)
    core = pd.DataFrame(rows).merge(nat, on=["TP_BUZ_NO", "TP_BUZ_NM_CLEAN"], validate="many_to_one").merge(region, on=RKEY, validate="many_to_one")
    for col in ["amt", "cnt", "ticket"]:
        core[f"relative_{col}_beta"] = core[f"{col}_beta"] - core[f"national_{col}_beta"]
        core[f"relative_{col}_growth_pct_m"] = np.expm1(core[f"relative_{col}_beta"]) * 100
        core[f"relative_{col}_direction"] = core[f"relative_{col}_growth_pct_m"].map(direction)
    core["region_amt_direction"] = core.region_amt_growth_pct_m.map(direction)
    core["region_industry_relation"] = np.where(core.region_amt_direction == core.amt_direction, "같은방향",
        np.where(((core.region_amt_direction == "상승") & (core.amt_direction == "감소")) | ((core.region_amt_direction == "감소") & (core.amt_direction == "상승")), "정반대", "정체포함차이"))
    core["decomposition_type"] = core.cnt_direction + " CNT·" + core.ticket_direction + " 건당결제액"
    facts = {
        "monthly_rows": len(p), "complete_combinations": len(core), "max_identity_error": float(core.identity_error.abs().max()),
        "strict_amt_up_cnt_down": int(((core.amt_direction == "상승") & (core.cnt_direction == "감소")).sum()),
        "amt_up_cnt_nonup": int(((core.amt_direction == "상승") & (core.cnt_direction != "상승")).sum()),
        "amt_up": int((core.amt_direction == "상승").sum()),
        "cnt_up_ticket_down": int(((core.cnt_direction == "상승") & (core.ticket_direction == "감소")).sum()),
        "region_industry_mismatch": int((core.region_industry_relation != "같은방향").sum()),
        "region_industry_opposite": int((core.region_industry_relation == "정반대").sum()),
        "region_up_industry_down": int(((core.region_amt_direction == "상승") & (core.amt_direction == "감소")).sum()),
        "relative_amt_up_cnt_down": int(((core.relative_amt_direction == "상승") & (core.relative_cnt_direction == "감소")).sum()),
        "peak_dependent": int(core.peak_dependent.sum()),
        "ticket_down": int((core.ticket_direction == "감소").sum()),
    }
    return monthly, national, core, facts


def composition_metrics(d: pd.DataFrame, core: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    numeric = d[d.AGE_CD.ne("x")].copy()
    variants = [("all_numeric", numeric), ("domestic_personal", numeric[numeric.GENDER_CD.isin(["1", "2"])])]
    out = core[RKEY + ["TP_BUZ_NO"]].copy()
    for label, q in variants:
        a = q.groupby(RKEY + ["TP_BUZ_NO", "STRD_YYMM", "AGE_CD"], as_index=False).cnt.sum()
        a["share"] = a.cnt / a.groupby(RKEY + ["TP_BUZ_NO", "STRD_YYMM"]).cnt.transform("sum")
        m = a.assign(hhi=a.share**2, ent=-a.share * np.log(a.share)).groupby(RKEY + ["TP_BUZ_NO", "STRD_YYMM"], as_index=False).agg(hhi=("hhi", "sum"), entropy=("ent", "sum"))
        m.entropy /= math.log(6)
        rows = []
        for key, g in m.groupby(RKEY + ["TP_BUZ_NO"]):
            if g.STRD_YYMM.nunique() != 6: continue
            g = g.sort_values("STRD_YYMM")
            rows.append({**dict(zip(RKEY + ["TP_BUZ_NO"], key)), f"age_hhi_slope_{label}": slope(g.hhi), f"age_entropy_slope_{label}": slope(g.entropy)})
        out = out.merge(pd.DataFrame(rows), on=RKEY + ["TP_BUZ_NO"], how="left", validate="one_to_one")
    # 성별 비교는 국내개인 코드 1·2만 사용한다. 3은 외국인, x는 법인이다.
    g = d[d.GENDER_CD.isin(["1", "2"])].groupby(RKEY + ["TP_BUZ_NO", "STRD_YYMM", "GENDER_CD"], as_index=False).cnt.sum()
    g["share"] = g.cnt / g.groupby(RKEY + ["TP_BUZ_NO", "STRD_YYMM"]).cnt.transform("sum")
    gm = g.assign(hhi=g.share**2).groupby(RKEY + ["TP_BUZ_NO", "STRD_YYMM"], as_index=False).hhi.sum()
    grows = []
    for key, z in gm.groupby(RKEY + ["TP_BUZ_NO"]):
        if z.STRD_YYMM.nunique() == 6:
            grows.append({**dict(zip(RKEY + ["TP_BUZ_NO"], key)), "gender_hhi_slope": slope(z.sort_values("STRD_YYMM").hhi)})
    out = out.merge(pd.DataFrame(grows), on=RKEY + ["TP_BUZ_NO"], how="left", validate="one_to_one")
    out["age_concentration_up_all"] = (out.age_hhi_slope_all_numeric > 0) & (out.age_entropy_slope_all_numeric < 0)
    out["age_concentration_up_domestic"] = (out.age_hhi_slope_domestic_personal > 0) & (out.age_entropy_slope_domestic_personal < 0)
    out["age_sensitivity_disagrees"] = out.age_concentration_up_all != out.age_concentration_up_domestic
    out["gender_concentration_up"] = out.gender_hhi_slope > 0
    facts = {"n": len(out), "age_up_all": int(out.age_concentration_up_all.sum()), "age_up_domestic": int(out.age_concentration_up_domestic.sum()),
             "age_disagree": int(out.age_sensitivity_disagrees.sum()), "gender_up": int(out.gender_concentration_up.sum())}
    return out, facts


def sensitivity(core: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for eps in [0.0, 0.25, 0.5]:
        for threshold in [0, 10_000, 100_000, 500_000]:
            q = core[core.total_cnt_6m >= threshold].copy()
            ad = q.amt_growth_pct_m.map(lambda x: direction(x, eps)); cd = q.cnt_growth_pct_m.map(lambda x: direction(x, eps)); td = q.ticket_growth_pct_m.map(lambda x: direction(x, eps)); rd = q.region_amt_growth_pct_m.map(lambda x: direction(x, eps))
            rad = q.relative_amt_growth_pct_m.map(lambda x: direction(x, eps)); rcd = q.relative_cnt_growth_pct_m.map(lambda x: direction(x, eps))
            rows.append({"neutral_band_pct_m": eps, "min_total_cnt_6m": threshold, "n": len(q),
                         "amt_up_cnt_down_n": int(((ad == "상승") & (cd == "감소")).sum()),
                         "amt_up_cnt_nonup_n": int(((ad == "상승") & (cd != "상승")).sum()),
                         "cnt_up_ticket_down_n": int(((cd == "상승") & (td == "감소")).sum()),
                         "region_industry_mismatch_n": int((rd != ad).sum()),
                         "region_up_industry_down_n": int(((rd == "상승") & (ad == "감소")).sum()),
                         "relative_amt_up_cnt_down_n": int(((rad == "상승") & (rcd == "감소")).sum())})
    return pd.DataFrame(rows)


def representative_cases(core: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    z = core.merge(comp, on=RKEY + ["TP_BUZ_NO"], validate="one_to_one")
    choices = []
    rules = [
        ("거래량증가·건당결제액하락", (z.amt_direction == "상승") & (z.cnt_direction == "상승") & (z.ticket_direction == "감소") & (z.total_cnt_6m >= 100000), "amt_growth_pct_m", False),
        ("엄격_AMT상승·CNT감소", (z.amt_direction == "상승") & (z.cnt_direction == "감소") & (z.total_cnt_6m >= 10000), "total_cnt_6m", False),
        ("지역상승·업종감소", (z.region_amt_direction == "상승") & (z.amt_direction == "감소") & (z.total_cnt_6m >= 100000), "total_cnt_6m", False),
        ("전국대비_AMT상승·CNT하락", (z.relative_amt_direction == "상승") & (z.relative_cnt_direction == "감소") & (z.total_cnt_6m >= 100000), "total_cnt_6m", False),
        ("피크의존", z.peak_dependent & (z.total_cnt_6m >= 100000), "total_cnt_6m", False),
        ("연령집중_민감도불일치", z.age_sensitivity_disagrees & (z.total_cnt_6m >= 100000), "total_cnt_6m", False),
        ("반례_동반성장·비피크", (z.amt_direction == "상승") & (z.cnt_direction == "상승") & (z.ticket_direction == "상승") & ~z.peak_dependent & (z.total_cnt_6m >= 100000), "total_cnt_6m", False),
    ]
    used = set()
    for label, mask, col, ascending in rules:
        q = z[mask].sort_values(col, ascending=ascending)
        q = q[~q.apply(lambda r: (r.SIDO_NM, r.CCG_NM, r.TP_BUZ_NO) in used, axis=1)]
        if q.empty: continue
        r = q.iloc[0]; used.add((r.SIDO_NM, r.CCG_NM, r.TP_BUZ_NO))
        choices.append({"case_type": label, "region": f"{r.SIDO_NM} {r.CCG_NM}", "industry": r.TP_BUZ_NM_CLEAN,
                        "amt_pct_m": round(r.amt_growth_pct_m, 2), "cnt_pct_m": round(r.cnt_growth_pct_m, 2), "ticket_pct_m": round(r.ticket_growth_pct_m, 2),
                        "national_amt_pct_m": round(r.national_amt_growth_pct_m, 2), "relative_amt_pct_m": round(r.relative_amt_growth_pct_m, 2),
                        "region_amt_pct_m": round(r.region_amt_growth_pct_m, 2), "total_cnt_6m": int(r.total_cnt_6m), "peak_month": int(r.peak_month)})
    return pd.DataFrame(choices)


def same_amt_matched_pairs(core: pd.DataFrame) -> pd.DataFrame:
    q = core[core.total_cnt_6m >= 100_000].copy().sort_values("amt_growth_pct_m")
    groups = {k: g.sort_values("amt_growth_pct_m") for k, g in q.groupby("decomposition_type")}
    candidates = []
    names = sorted(groups)
    for i, a_name in enumerate(names):
        for b_name in names[i + 1:]:
            a, b = groups[a_name], groups[b_name]
            if a.empty or b.empty: continue
            av, bv = a.amt_growth_pct_m.to_numpy(), b.amt_growth_pct_m.to_numpy(); ia = ib = 0; best = None
            while ia < len(av) and ib < len(bv):
                delta = abs(av[ia] - bv[ib])
                if best is None or delta < best[0]: best = (delta, a.iloc[ia], b.iloc[ib])
                if av[ia] < bv[ib]: ia += 1
                else: ib += 1
            if best is not None: candidates.append(best)
    rows = []
    used = set()
    for delta, a, b in sorted(candidates, key=lambda x: x[0]):
        aid = (a.SIDO_NM, a.CCG_NM, a.TP_BUZ_NO); bid = (b.SIDO_NM, b.CCG_NM, b.TP_BUZ_NO)
        if aid in used or bid in used: continue
        used.update([aid, bid])
        rows.append({"amt_gap_pp_m": round(delta, 4),
                     "case_A": f"{a.SIDO_NM} {a.CCG_NM}·{a.TP_BUZ_NM_CLEAN}", "A_amt": round(a.amt_growth_pct_m, 3), "A_cnt": round(a.cnt_growth_pct_m, 3), "A_ticket": round(a.ticket_growth_pct_m, 3), "A_structure": a.decomposition_type,
                     "case_B": f"{b.SIDO_NM} {b.CCG_NM}·{b.TP_BUZ_NM_CLEAN}", "B_amt": round(b.amt_growth_pct_m, 3), "B_cnt": round(b.cnt_growth_pct_m, 3), "B_ticket": round(b.ticket_growth_pct_m, 3), "B_structure": b.decomposition_type})
        if len(rows) == 5: break
    return pd.DataFrame(rows)


def audit_external(bc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted([p for p in (ROOT / "dataset").rglob("*") if p.is_file() and p != BC_PATH and p.name not in {".DS_Store", "dataset.md", "localdata_download.md"}])
    rows = []
    for p in files:
        rel = str(p.relative_to(ROOT)); suffix = p.suffix.lower(); group = "OTHER"; period = ""; level = ""; decision = "제외"; reason = "분석 데이터가 아닌 문서/중복 산출물"
        s = rel
        if "mois/202601_202606_age_population" in s: group, period, level, decision, reason = "주민등록 연령인구", "2026.01~06", "시군구×월×연령", "핵심 사용", "BC 연령구조의 인구 통제"
        elif "kosis/202601_202606" in s and suffix == ".csv": group, period, level, decision, reason = "CPI", "2026.01~06", "전국×월", "제한적 사용", "지역 차이 없이 명목/실질 민감도만 가능"
        elif "kosis/bc_cpi_mapping" in s: group, period, level, decision, reason = "CPI 매핑", "비시계열", "11 BC업종", "제한적 사용", "대부분 근사 매핑"
        elif "kosis/2026_" in s and suffix == ".pdf": group, period, level, decision, reason = "CPI 원문", "2026 상반기", "전국", "제한적 사용", "CSV 출처 검증용"
        elif "localdata_raw" in s: group, period, level, decision, reason = "LOCALDATA 원본", "최신 스냅샷+개폐업일자", "개별 인허가", "제한적 사용", "식품업종만; BC 가맹점 모집단과 다름"
        elif "processed/LOCALDATA" in s: group, period, level, decision, reason = "LOCALDATA 가공패널", "2026.01~06 재구성", "시군구×월×6 연결업종", "제한적 사용", "중국음식·제과점 우선, 나머지 근사"
        elif "external_raw/localdata" in s: group, period, level, decision, reason = "LOCALDATA 검증·매핑", "비시계열", "파일/업종", "제한적 사용", "원본 해시·ID·매핑 검증"
        elif "폐업률/월간 지역 경제지표" in s: group, period, level, decision, reason = "국세청 월간 원본", re.search(r"2026년 (\d+)월", p.name).group(0) if re.search(r"2026년 (\d+)월", p.name) else "", "시군구×101업종", "제한적 사용", "가동사업자는 사용 가능; 신규·폐업은 마스킹 심함"
        elif "processed/BC_국세청" in s: group, period, level, decision, reason = "국세청 가공패널", "2026.01~06", "시군구/시도×월×101업종", "제한적 사용", "편의점·슈퍼마켓 공급 방향 중심"
        elif "mois_living_population" in s: group, period, level, decision, reason = "생활인구", "2026.01~03", "89개 인구감소지역×월", "제외", "요청 분석기간 6개월 미충족; 사례 맥락만 가능"
        elif "reb_commercial" in s: group, period, level, decision, reason = "임대료·공실", "2026.Q1~Q2", "시도/대표상권×분기", "제외", "시군구×월×업종 매칭 불가"
        elif "semas" in s: group, period, level, decision, reason = "SEMAS", "2026.08 가이드/비시계열", "API 문서·매핑", "제외", "실제 2026.01~06 점포 패널 없음"
        elif "external_data_metadata" in s: group, period, level, decision, reason = "메타데이터", "비시계열", "자료목록", "제한적 사용", "출처 추적용"
        rows.append({"file": rel, "bytes": p.stat().st_size, "sha256": sha256(p), "dataset_group": group, "period": period, "spatial_industry_unit": level, "decision": decision, "reason": reason})
    inventory = pd.DataFrame(rows)

    bc_regions = bc[RKEY].drop_duplicates(); bc_region_set = set(map(tuple, bc_regions.to_numpy())); match_rows = []
    bc_month_ind = bc[RKEY + ["STRD_YYMM", "TP_BUZ_NO"]].drop_duplicates()
    # Population exact region match after the same official-name cleanup used in analysis.
    pop = pd.read_csv(ROOT / "dataset/external_raw/mois/202601_202606_age_population.csv", encoding="cp949")
    pop["full"] = pop["행정구역"].str.replace(r"\s*\(\d+\)", "", regex=True).str.strip().str.replace(r"\s+", " ", regex=True)
    pop_full = set(pop.full)
    total_cols = [c for c in pop.columns if "총인구수" in c]
    pop_valid = set(pop.loc[pop[total_cols].notna().all(axis=1) & pop[total_cols].astype(str).apply(lambda x: x.str.strip().ne("").all(), axis=1), "full"])
    pop_match = sum((s if s == "세종특별자치시" else f"{s} {c}") in pop_valid for s, c in bc_region_set)
    match_rows.append(["주민등록 연령인구", "2026.01~06", 255, pop_match, (255-pop_match)/255*100, 6, pop_match*6, 255*6, pop_match/255*100, (255-pop_match)/255*100, "없음", "핵심 사용", "화성 신설 4구 값 결측"])
    cpi = pd.read_csv(ROOT / "dataset/external_raw/kosis/202601_202606_detailed_cpi.csv")
    cmap = pd.read_csv(ROOT / "dataset/external_raw/kosis/bc_cpi_mapping.csv")
    match_rows.append(["CPI", "2026.01~06", 255, 255, 0.0, cpi.year_month.nunique(), 6, 6, 100.0, float(cpi.isna().mean().mean()*100), f"{cmap.BC_TP_BUZ_NO.nunique()}/11 근사", "제한적 사용", "전국 공통지수"])
    local = pd.read_excel(ROOT / "dataset/processed/LOCALDATA_식품업종_개폐업_202601_202606.xlsx", sheet_name="시군구_월별_개폐업")
    local_regions = set(map(tuple, local[["시도", "시군구"]].drop_duplicates().to_numpy())); local_match = len(bc_region_set & local_regions)
    local_code_map = {8001:"한식통합",8002:"한식통합",8003:"한식통합",8004:"일식회집",8005:"중국음식",8006:"서양음식",8021:"스넥",8301:"제과점"}
    bl = bc_month_ind[bc_month_ind.TP_BUZ_NO.isin(local_code_map)].copy(); bl["연결업종"] = bl.TP_BUZ_NO.map(local_code_map)
    lk = local.rename(columns={"기준년월":"STRD_YYMM","시도":"SIDO_NM","시군구":"CCG_NM"})[RKEY+["STRD_YYMM","연결업종"]].drop_duplicates()
    local_key_match = int(bl.merge(lk, on=RKEY+["STRD_YYMM","연결업종"], how="inner").shape[0])
    match_rows.append(["LOCALDATA 식품업종", "2026.01~06 재구성", 255, local_match, (255-local_match)/255*100, local.기준년월.nunique(), local_key_match, len(bl), local_key_match/len(bl)*100, float(local.isna().mean().mean()*100), f"8/11 BC코드→6 외부업종, HIGH 2", "제한적 사용", "식품 인허가와 BC 모집단 차이; 한식 3종 통합"])
    nts_path = ROOT / "dataset/processed/BC_국세청_폐업지표_202601_202606.xlsx"
    nts = pd.read_excel(nts_path, sheet_name="국세청_시군구_101업종")
    nts_regions = set(map(tuple, nts[["시도", "시군구"]].drop_duplicates().to_numpy())); nts_match = len(bc_region_set & nts_regions)
    closure_missing = float(nts.폐업_당월.isna().mean()*100)
    nts_map = pd.read_excel(nts_path, sheet_name="BC업종_매핑").dropna(subset=["국세청업종"])
    bn = bc_month_ind.merge(nts_map[["BC업종코드","국세청업종"]], left_on="TP_BUZ_NO", right_on="BC업종코드", how="inner")
    nk = nts.rename(columns={"기준년월":"STRD_YYMM","시도":"SIDO_NM","시군구":"CCG_NM"})[RKEY+["STRD_YYMM","국세청업종"]].drop_duplicates()
    nts_key_match = int(bn.merge(nk, on=RKEY+["STRD_YYMM","국세청업종"], how="inner").shape[0])
    match_rows.append(["국세청 월간 지역지표", "2026.01~06", 255, nts_match, (255-nts_match)/255*100, nts.기준년월.nunique(), nts_key_match, len(bn), nts_key_match/len(bn)*100, closure_missing, "BC 직접대응 10/11, 정확 3(주모듈 2)", "제한적 사용", "가동사업자 사용; 폐업값 마스킹; 일부 행정명칭 차이"])
    living = pd.read_csv(ROOT / "dataset/external_raw/mois_living_population/202601_202603_living_population_by_region.csv")
    liv_regions = set(map(tuple, living[RKEY].drop_duplicates().to_numpy())); liv_match = len(bc_region_set & liv_regions)
    match_rows.append(["생활인구", "2026.01~03", 255, liv_match, (255-liv_match)/255*100, living.STRD_YYMM.nunique(), len(living), 255*6, len(living)/(255*6)*100, float(living.isna().mean().mean()*100), "업종 없음", "제외", "6개월 미충족"])
    reb = pd.read_csv(ROOT / "dataset/external_raw/reb_commercial/2026Q1_Q2_commercial_rent_vacancy_province.csv")
    match_rows.append(["임대료·공실", "2026.Q1~Q2", 255, 0, 100.0, 0, 0, 255*6, 0.0, float(reb.isna().mean().mean()*100), "업종 없음", "제외", "시군구·월 매칭 불가"])
    match_rows.append(["SEMAS", "과거 패널 없음", 255, 0, 100.0, 0, 0, len(bc_month_ind), 0.0, 100.0, "매핑 설계만", "제외", "API 문서만 존재"])
    match = pd.DataFrame(match_rows, columns=["dataset", "period", "bc_regions", "matched_regions", "region_missing_pct", "matched_months", "matched_join_keys", "eligible_join_keys", "join_key_match_pct", "target_value_missing_pct", "industry_coverage", "decision", "logic_limit"])
    return inventory, match


def scorecard(core: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        ["같은 AMT 변화라도 내부 구조가 다르다", "강함", "CNT·건당 결제금액 분해 항등식과 광범위한 상쇄 패턴; AMT 증가·CNT 증가·건당 결제금액 감소가 반복"],
        ["지역 전체 성장과 개별업종 성장은 다를 수 있다", "강함", "±0.25%에서 방향 불일치 679/2,327, 정반대 496, 지역상승·업종감소 432"],
        ["전국 benchmark로 공통 월 효과와 지역 특이를 어느 정도 구분", "중간", "동일업종 전국 분모의 상대추세는 계산 가능하나 6개월·동시기 비교로 계절성을 완전 제거하지 못함"],
        ["소비층 변화와 peak 의존성으로 점검 우선순위 생성", "중간", f"연령집중 전체 {int(comp.age_concentration_up_all.sum())}/{len(comp)}, 피크의존 {int(core.peak_dependent.sum())}/{len(core)}; 점포 관련성은 미확인"],
        ["최대 3개의 추가 질문을 제시하는 진단 시스템", "중간", "규칙·gate·질문예산·재현성은 구현 가능; 사람의 판단개선은 아직 평가하지 않음"],
        ["점주 답변 확인 후에만 조건부 개선 후보 제안", "강함", "현재 데이터에 고객·상품·운영변수가 없으므로 단정 대신 답변 gate를 두는 구조가 데이터 한계와 정합"],
    ], columns=["claim", "rating", "basis"])


def write_report(out: Path, bc_checks, coverage, bc_facts, core, core_facts, comp, comp_facts, sens, reps, matched_pairs, inventory, external_match, scores):
    s025 = sens[(sens.neutral_band_pct_m == .25) & (sens.min_total_cnt_6m == 0)].iloc[0]
    s100 = sens[(sens.neutral_band_pct_m == .25) & (sens.min_total_cnt_6m == 100000)].iloc[0]
    indcov = coverage[["TP_BUZ_NO", "TP_BUZ_NM_CLEAN", "observed_regions", "complete6_regions", "complete6_pct"]].round(2)
    ext_summary = external_match.copy().round(2)
    sensitivity_show = sens[sens.min_total_cnt_6m.isin([0, 100000])].copy()
    sensitivity_show["strict_pct"] = (sensitivity_show.amt_up_cnt_down_n / sensitivity_show.n * 100).round(3)
    sensitivity_show["mismatch_pct"] = (sensitivity_show.region_industry_mismatch_n / sensitivity_show.n * 100).round(2)
    sensitivity_show = sensitivity_show[["neutral_band_pct_m", "min_total_cnt_6m", "n", "amt_up_cnt_down_n", "strict_pct", "cnt_up_ticket_down_n", "region_industry_mismatch_n", "mismatch_pct", "relative_amt_up_cnt_down_n"]]
    report = f"""# BC카드 공모전 최종 주제 검증 보고서

## 1. 최종 판정

> **판정: MODIFY**

현재 데이터는 `같은 매출 변화라도 내부 소비구조가 다르며 확인할 점검항목도 달라진다`는 핵심 문제를 충분히 지지한다. 그러나 `소상공인에게 맞춤형 개선안을 제안한다`는 표현은 점포 운영·고객·상품·가격 자료와 사람평가가 없어 그대로는 방어하기 어렵다.

권장 최종 주제는 다음과 같다.

> **“같은 매출 변화, 다른 점검” — 지역×업종 소비구조 분해 기반 경영 점검 우선순위와 조건부 개선 후보**

변경해야 할 사항은 세 가지다.

1. `처방·개선안`을 `점검 우선순위·조건부 개선 후보`로 제한한다.
2. 엄격한 AMT+·CNT- 성장착시가 아니라 거래량·건당 결제금액·benchmark·구성·피크의 다중 경로를 중심에 둔다.
3. 외부데이터는 주민등록인구를 핵심 통제로, CPI·LOCALDATA·국세청을 제한 모듈로만 사용하고 생활인구·임대료·SEMAS는 핵심분석에서 제외한다.

## 2. 원본 BC 데이터 감사

원본 `{BC_PATH.relative_to(ROOT)}`에서 직접 다시 계산했다. 기존 분석 산출물은 입력으로 사용하지 않았다.

### 2.1 기본 품질

- 행: {bc_facts['rows']:,}, 원 컬럼: 9개
- 기간: {bc_facts['months']}
- 지역 복합키: {bc_facts['regions']}개, 시도: {bc_facts['sidos']}개
- 업종: {bc_facts['industries']}개
- 원자료 결측 셀: {bc_facts['null_cells']:,}
- 분석 grain 중복: {bc_facts['duplicate_grain']:,}
- CNT 최솟값: {bc_facts['min_cnt']}, CNT 1~10 행: {bc_facts['cnt_1_10_rows']}
- AMT가 10,000원 단위인 행: {bc_facts['amt_multiple_10000_pct']:.2f}%
- CCG_NM 단독 사용 시 복수 시도에 걸치는 이름: {bc_facts['ambiguous_ccg_names']}개

### 2.2 요청 한계 판정

{md(bc_checks)}

silent suppression은 **추정만 가능**하다. `cnt` 최솟값이 11이고 1~10이 전혀 없으며 소표본 업종의 지역 누락이 많지만, 원자료에 suppression 플래그나 제공규칙이 없다. 미관측 조합을 0으로 대체하지 않는다.

### 2.3 업종 커버리지

{md(indcov)}

갈비전문점과 한정식은 지역 누락이 특히 많다. 완전관측 조합만 slope 분석에 사용하고 누락 자체를 영업 부재로 해석하지 않는다.

## 3. 외부데이터 전수 감사

프로젝트 `dataset` 아래의 외부 원본·가공본·문서 {len(inventory)}개를 전수 목록화하고 해시를 기록했다. 파일별 결과는 `external_file_inventory.csv`, 분석자료별 매칭 결과는 아래와 같다.

{md(ext_summary)}

### 3.1 최종 사용 분류

**핵심 사용**

- 주민등록 연령인구: 2026.01~06을 포함하며 BC 255개 지역 중 251개 매칭. 소비층 구성과 주민구조를 구분하는 통제자료로만 사용한다.

**제한적 사용**

- CPI: 6개월 완전하지만 전국지수여서 지역 차이를 설명하지 못한다. 명목·실질 민감도만 가능하다.
- LOCALDATA: 255개 지역과 6개월을 재구성했으나 BC 11개 중 6개 연결업종뿐이다. 중국음식·제과점 매핑을 우선하고 나머지는 사례·민감도로 제한한다.
- 국세청: 6개월과 시군구를 포함하지만 BC와 정확히 직접 대응하는 핵심 업종은 편의점·슈퍼마켓이다. 가동사업자 수는 공급 방향에 사용할 수 있으나 폐업값은 마스킹이 심하다.

**핵심분석 제외**

- 생활인구: 89개 지역, 1~3월만 있어 6개월 핵심분석에서 제외한다.
- 임대료·공실: 분기·시도/대표상권 단위여서 시군구×월×업종 진단에 직접 매칭할 수 없다.
- SEMAS: 현재 프로젝트에는 API 가이드와 매핑 설계만 있고 2026.01~06 실제 점포 패널이 없다.
- 국세청 신규·폐업 세부값: 다수 마스킹으로 핵심 성과변수에서 제외한다.

## 4. 핵심 분석 재검증

### 4.1 AMT 분해

`AMT = CNT × AMT/CNT`이므로 로그 slope도 `AMT beta = CNT beta + 건당 결제금액 beta`로 분해된다. 2,327개 완전관측 조합의 최대 수치오차는 `{core_facts['max_identity_error']:.2e}`였다.

±0.25%/월 기준:

- AMT 상승 조합: {core_facts['amt_up']:,}
- 엄격한 AMT 상승·CNT 감소: {core_facts['strict_amt_up_cnt_down']:,}개
- AMT 상승·CNT 비상승: {core_facts['amt_up_cnt_nonup']:,}개
- CNT 상승·건당 결제금액 하락: {core_facts['cnt_up_ticket_down']:,}개
- 건당 결제금액 하락: {core_facts['ticket_down']:,}개

**해석:** 엄격한 성장착시는 드물지만, 거래량 증가와 건당 결제금액 하락이 함께 나타나는 구조는 광범위하다. 따라서 이 주제는 성장착시 탐지보다 `AMT 성장경로 분해`로 설명해야 한다.

거래량 10만건 이상에서 AMT 월성장률이 거의 같은데 CNT·건당 결제금액 경로가 다른 실제 쌍은 다음과 같다.

{md(matched_pairs)}

이는 AMT 차이가 작더라도 내부 경로가 서로 다를 수 있다는 직접적인 사례다. 사례쌍은 성능표본이 아니라 구조 차이 설명용이다.

### 4.2 전국 동일업종 benchmark

전국 동일업종의 월별 AMT·CNT를 분모로 지역 점유율 및 `지역 로그 beta-전국 로그 beta`를 계산했다.

- 상대 AMT 상승·상대 CNT 감소: {core_facts['relative_amt_up_cnt_down']:,}/{len(core):,}
- 지역총량·업종 방향 불일치: {core_facts['region_industry_mismatch']:,}/{len(core):,}
- 실제 정반대: {core_facts['region_industry_opposite']:,}
- 지역총량 상승·업종 감소: {core_facts['region_up_industry_down']:,}

benchmark는 같은 달 전국 업종 변화보다 빠른지 느린지를 구분한다. 하지만 6개월 안의 전국 공통 월 변화를 상대화하는 방법이지 전년동월 계절성이나 인과효과를 완전히 통제하는 방법은 아니다.

### 4.3 연령·성별 소비구조

- 전체 숫자연령 기준 집중 증가: {comp_facts['age_up_all']:,}/{comp_facts['n']:,}
- 국내개인(GENDER 1·2) 기준 집중 증가: {comp_facts['age_up_domestic']:,}/{comp_facts['n']:,}
- 두 연령 판정이 다른 조합: {comp_facts['age_disagree']:,}
- 국내개인 성별(남성·여성) CNT HHI 증가 조합: {comp_facts['gender_up']:,}/{comp_facts['n']:,}

외국인이 숫자 연령코드에 포함되므로 전체와 국내개인 결과를 함께 제시한다. 성별 집중도는 외국인·법인을 제외한 코드 1·2만 계산했다. 집중 증가는 점검 신호이지 위험판정이 아니다. 실제 고객수와 주민구조는 현재 BC 데이터의 결제건수 구성과 동일하지 않다.

### 4.4 월별 안정성과 피크

- AMT 상승이 특정 최대월 제거 후 유지되지 않는 피크 의존 조합: {core_facts['peak_dependent']:,}/{len(core):,} ({core_facts['peak_dependent']/len(core)*100:.2f}%)
- 피크월은 행사·휴점·영업일·대량결제·관측범위 변화 중 무엇 때문인지 원자료만으로 알 수 없다.

피크 의존성은 장기 원인을 설명하지 않지만 `6개월 slope를 바로 장기 성장으로 해석하지 말고 먼저 특정 월을 확인하라`는 점검 우선순위를 만들 수 있다.

### 4.5 거래규모·중립구간 민감도

{md(sensitivity_show)}

엄격한 AMT+·CNT-는 중립구간 없이 9개, ±0.25%에서 {int(s025.amt_up_cnt_down_n)}개이며 10만건 이상에서는 {int(s100.amt_up_cnt_down_n)}개로 줄었다. 과거 원단위 선형 slope 부호 정의의 10개와도 다르므로 정의를 섞지 않는다. 지역총량·업종 불일치는 임계값과 거래량 하한에도 상당수 남지만 정확한 발생률은 중립구간 정의에 민감하다.

### 4.6 대표 사례와 반례

{md(reps)}

대표 사례는 패턴의 존재를 보여주는 예시이며 전국 성능이나 위험 사례가 아니다. 총 CNT가 작은 엄격 성장착시는 별도 표시하고 큰 거래량 사례를 우선한다.

## 5. 주장별 최종 판정

{md(scores)}

`강함`은 현상 또는 설계 정합성이 원자료 재계산에서 반복된다는 뜻이다. 사람의 의사결정 개선이나 실제 매출 효과가 검증됐다는 뜻은 아니다.

## 6. 확실히 확인된 사실과 확인되지 않은 해석

### 확실히 확인된 사실

1. AMT 변화는 CNT와 건당 결제금액 변화의 합으로 정확히 분해된다.
2. 엄격한 AMT+·CNT-는 드물지만 CNT 증가·건당 결제금액 하락은 광범위하다.
3. 지역총량과 개별업종 방향이 다른 사례가 반복된다.
4. 전국 동일업종 benchmark로 지역의 상대적 강약을 계산할 수 있다.
5. 연령·성별 구성과 특정 월 의존성이 지역×업종별로 다르다.
6. 작은 거래량, 중립구간, 외국인 포함 여부에 따라 일부 판정이 달라진다.
7. 현재 BC 데이터에는 점포·고객·상품·성과 라벨이 없다.

### 확인되지 않은 해석

1. CNT 감소가 고객 이탈 또는 재방문 감소라는 해석
2. 건당 결제금액 하락이 가격·마진·이익 하락이라는 해석
3. 연령·성별 집중이 위험하다는 해석
4. 피크월이 행사 또는 관광 때문이라는 해석
5. 지역×업종 결과가 특정 점포에도 동일하다는 해석
6. 진단 질문이 일반 체크리스트보다 사람의 판단을 개선한다는 주장
7. 조건부 개선 후보가 매출을 높인다는 주장
8. 진단 플래그가 폐업 가능성을 예측한다는 주장

## 7. GO / MODIFY / NO-GO 판단

### 최종: MODIFY

**GO인 부분**

- 같은 AMT 변화의 내부 구조가 다르다는 문제정의
- 지역총량·개별업종·전국 benchmark를 분리하는 분석
- 연령·성별 구성과 피크 의존을 점검 신호로 사용하는 것
- 확인된 신호에 따라 최대 3개의 추가 질문을 제시하는 규칙 설계
- 점포 답변이 없으면 판단과 개선 후보를 유보하는 방식

**수정할 부분**

- 제목 또는 설명에서 `맞춤형 개선안`을 `조건부 개선 후보`로 변경
- 성장착시를 중심 개념에서 제외하고 `성장경로·소비구조 분해`를 중심에 배치
- 위험점수·폐업률·예상 매출증가율을 출력하지 않음
- 생활인구·임대료·SEMAS를 전국 핵심근거에서 제외
- 일반 방식 대비 우수성은 사람평가 전까지 미입증으로 표시

## 8. 발표에 사용할 핵심 분석 4개

1. **AMT 성장경로 분해**  
   같은 AMT 변화가 CNT와 건당 결제금액의 서로 다른 조합에서 만들어짐을 보여준다.

2. **지역총량·개별업종·전국 benchmark 3단 비교**  
   지역이 성장해도 개별업종은 감소할 수 있고, 절대성장과 전국 대비 상대성장이 다를 수 있음을 보여준다.

3. **소비층 구성 민감도**  
   연령·성별 집중 방향과 전체 숫자연령/국내개인 판정 차이를 함께 보여주어 단순 구성비 해석을 피한다.

4. **월별 안정성·피크 의존**  
   6개월 slope가 특정 월에 의존하는지를 표시해 장기성장 해석 전에 확인할 질문을 만든다.

공급점포수는 정확 매핑되는 업종에서만 보조 카드로 제시하고 핵심 4개에는 포함하지 않는 것이 안전하다.

## 9. 재현 파일

- 분석 코드: `scripts/validate_final_topic_from_raw.py`
- BC 한계 판정: `bc_limit_checks.csv`
- 업종 커버리지: `bc_industry_coverage.csv`
- 지역×업종 핵심 결과: `core_region_industry_metrics.csv`
- 연령·성별 구성: `composition_metrics.csv`
- 민감도: `sensitivity_results.csv`
- 대표·반례: `representative_and_counter_cases.csv`
- 유사 AMT 실제 쌍: `same_amt_matched_pairs.csv`
- 외부 파일 전수 목록: `external_file_inventory.csv`
- 외부 매칭 감사: `external_dataset_match_audit.csv`
- 주장 판정: `claim_scorecard.csv`
- 실행 검증 기록: `validation_checks.csv`
- 입력·출력 해시: `manifest.csv`

실행 명령:

```bash
PYTHONPYCACHEPREFIX=/tmp/bc_final_topic_validation_pycache python3 scripts/validate_final_topic_from_raw.py --out {out.relative_to(ROOT)}
```

분석 기준일: 2026-09-07  
분석 성격: 원자료 재계산 기반 경영데이터 분석. 인과·폐업·개선효과 검증이 아님.
"""
    (out / "final_topic_validation_report.md").write_text(report, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="analysis/final_topic_validation/run_20260907"); args = ap.parse_args()
    out = (ROOT / args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    bc = load_bc()
    bc_checks, coverage, bc_facts = audit_bc(bc)
    monthly, national, core, core_facts = build_core(bc)
    comp, comp_facts = composition_metrics(bc, core)
    sens = sensitivity(core)
    reps = representative_cases(core, comp)
    matched_pairs = same_amt_matched_pairs(core)
    inventory, external_match = audit_external(bc)
    scores = scorecard(core, comp)
    validation = pd.DataFrame([
        ["bc_rows", len(bc) == 242574, len(bc)],
        ["complete_region_industry", len(core) == 2327, len(core)],
        ["core_key_unique", not core.duplicated(RKEY + ["TP_BUZ_NO"]).any(), int(core.duplicated(RKEY + ["TP_BUZ_NO"]).sum())],
        ["log_identity", core.identity_error.abs().max() < 1e-12, float(core.identity_error.abs().max())],
        ["composition_key_unique", not comp.duplicated(RKEY + ["TP_BUZ_NO"]).any(), int(comp.duplicated(RKEY + ["TP_BUZ_NO"]).sum())],
        ["sensitivity_grid", len(sens) == 12, len(sens)],
        ["external_file_inventory", len(inventory) == 30, len(inventory)],
        ["external_decision_complete", external_match.decision.isin(["핵심 사용", "제한적 사용", "제외"]).all(), int(external_match.decision.isna().sum())],
    ], columns=["check", "passed", "value"])
    if not validation.passed.all(): raise AssertionError(validation.loc[~validation.passed].to_dict("records"))
    tables = {
        "bc_limit_checks.csv": bc_checks, "bc_industry_coverage.csv": coverage,
        "core_region_industry_metrics.csv": core, "composition_metrics.csv": comp,
        "sensitivity_results.csv": sens, "representative_and_counter_cases.csv": reps, "same_amt_matched_pairs.csv": matched_pairs,
        "external_file_inventory.csv": inventory, "external_dataset_match_audit.csv": external_match,
        "claim_scorecard.csv": scores, "validation_checks.csv": validation,
    }
    for name, table in tables.items(): table.to_csv(out / name, index=False, encoding="utf-8-sig")
    write_report(out, bc_checks, coverage, bc_facts, core, core_facts, comp, comp_facts, sens, reps, matched_pairs, inventory, external_match, scores)
    shutil.copy2(__file__, out / Path(__file__).name)
    manifest_rows = [{"path": str(BC_PATH.relative_to(ROOT)), "role": "INPUT", "sha256": sha256(BC_PATH)}]
    for p in sorted([x for x in (ROOT / "dataset").rglob("*") if x.is_file() and x != BC_PATH]):
        manifest_rows.append({"path": str(p.relative_to(ROOT)), "role": "EXTERNAL_INPUT_OR_DOCUMENT", "sha256": sha256(p)})
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name != "manifest.csv": manifest_rows.append({"path": str(p.relative_to(ROOT)), "role": "OUTPUT", "sha256": sha256(p)})
    pd.DataFrame(manifest_rows).to_csv(out / "manifest.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"out": str(out), "bc_rows": len(bc), "complete_combinations": len(core), "external_files": len(inventory), "verdict": "MODIFY"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
