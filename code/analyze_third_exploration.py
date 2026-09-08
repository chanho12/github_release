"""BC카드 2026년 상반기 3차 탐색 분석(H1~H15).

공식 외부자료(주민등록 인구, CPI, 국세청 가동사업자)를 결합한다.
누락된 BC 지역×업종은 0으로 대체하지 않는다. 업종 포트폴리오 분석은
전 지역에서 완전 관측되는 8개 업종을 주 분석으로 하고 9개 고관측 업종을
민감도 분석으로 병행한다.
"""
from __future__ import annotations

import json, math, os, tempfile, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.gettempdir()) / "bc_third_exploration"
TMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP / "xdg"))
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon, cosine
from scipy.stats import linregress, spearmanr, theilslopes
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from analyze_consumption_patterns import age_concentration, amt_decomposition
from analyze_growth_illusion import AGE_NAMES, configure_plot

DATA = ROOT / "dataset" / "ABP_CONTEST_DATA.csv"
POP_FILE = ROOT / "dataset" / "external_raw" / "mois" / "202601_202606_age_population.csv"
CPI_FILE = ROOT / "dataset" / "external_raw" / "kosis" / "202601_202606_cpi.csv"
NTS_FILE = ROOT / "dataset" / "processed" / "BC_국세청_폐업지표_202601_202606.xlsx"
OUT = ROOT / "analysis" / "third_exploration"
FIG = OUT / "figures"
MONTHS = list(range(202601, 202607))
RKEY = ["SIDO_NM", "CCG_NM"]
CORE7 = [4010, 4020, 8001, 8005, 8006, 8021, 8301]
FULL_COVERAGE = [4004] + CORE7
HIGH_COVERAGE = FULL_COVERAGE + [8004]
ALL11 = [4004,4010,4020,8001,8002,8003,8004,8005,8006,8021,8301]


def trend(y) -> dict:
    y = np.asarray(y, float)
    x = np.arange(len(y), dtype=float)
    if len(y) != 6 or np.any(~np.isfinite(y)) or np.any(y <= 0):
        return {k: np.nan for k in ["ols_beta", "ols_pct_m", "r2", "theil_beta", "first3_last3_pct", "robust_signs"]}
    ly = np.log(y)
    ols = linregress(x, ly)
    ts = theilslopes(ly, x)
    d = np.log(y[3:].mean() / y[:3].mean())
    signs = [np.sign(ols.slope), np.sign(ts.slope), np.sign(d)]
    return {"ols_beta": ols.slope, "ols_pct_m": np.expm1(ols.slope)*100,
            "r2": ols.rvalue**2, "theil_beta": ts.slope,
            "first3_last3_pct": np.expm1(d)*100,
            "robust_signs": int(abs(sum(signs)) == 3)}


def linear_trend(y) -> dict:
    y=np.asarray(y,float); x=np.arange(len(y),dtype=float)
    if len(y)!=6 or np.any(~np.isfinite(y)):
        return {"slope":np.nan,"r2":np.nan,"theil":np.nan,"first3_last3":np.nan,"robust_signs":0}
    o=linregress(x,y); t=theilslopes(y,x).slope; d=y[3:].mean()-y[:3].mean()
    return {"slope":o.slope,"r2":o.rvalue**2,"theil":t,"first3_last3":d,
            "robust_signs":int(abs(np.sign(o.slope)+np.sign(t)+np.sign(d))==3)}


def load_bc():
    d=pd.read_csv(DATA,dtype={"GENDER_CD":str,"AGE_CD":str})
    d["TP_BUZ_NM"]=d["TP_BUZ_NM"].str.replace(" ","",regex=False)
    return d


def parse_population(bc_regions):
    raw=pd.read_csv(POP_FILE,encoding="cp949")
    raw["full"]=raw["행정구역"].str.replace(r"\s*\(\d+\)","",regex=True).str.strip().str.replace(r"\s+"," ",regex=True)
    # 세종은 시도행과 시군구행이 같은 명칭·같은 값으로 중복 제공된다.
    raw=raw.drop_duplicates("full",keep="last")
    keys=[]
    for _,r in bc_regions.iterrows():
        full=r.SIDO_NM if r.SIDO_NM=="세종특별자치시" else f"{r.SIDO_NM} {r.CCG_NM}"
        keys.append((r.SIDO_NM,r.CCG_NM,full))
    key=pd.DataFrame(keys,columns=RKEY+["full"])
    z=key.merge(raw,on="full",how="left",validate="one_to_one")
    if z.filter(like="총인구수").isna().all(axis=1).any(): raise ValueError("주민등록 인구 지역 매칭 실패")
    rows=[]
    for _,r in z.iterrows():
        for m in MONTHS:
            ym=str(m); pref=f"{ym[:4]}년{ym[4:]}월_계_"
            def as_int(v):
                return 0 if pd.isna(v) or str(v).strip()=="" else int(str(v).replace(",",""))
            vals={a:as_int(r[pref+a]) for a in
                  ["총인구수","0~9세","10~19세","20~29세","30~39세","40~49세","50~59세","60~69세","70~79세","80~89세","90~99세","100세 이상"]}
            age=[vals["0~9세"]+vals["10~19세"],vals["20~29세"],vals["30~39세"],vals["40~49세"],vals["50~59세"],sum(vals[k] for k in ["60~69세","70~79세","80~89세","90~99세","100세 이상"])]
            shares=np.array(age)/vals["총인구수"]
            decade_keys=["0~9세","10~19세","20~29세","30~39세","40~49세","50~59세","60~69세","70~79세","80~89세","90~99세","100세 이상"]
            decade_midpoints=np.array([5,15,25,35,45,55,65,75,85,95,105],dtype=float)
            weighted_age=(np.array([vals[k] for k in decade_keys])*decade_midpoints).sum()/vals["총인구수"] if vals["총인구수"] else np.nan
            rows.append({"SIDO_NM":r.SIDO_NM,"CCG_NM":r.CCG_NM,"STRD_YYMM":m,"population":vals["총인구수"],"pop_60plus_share":shares[-1],"pop_weighted_age":weighted_age,"pop_age_hhi":sum(shares**2),"pop_age_entropy":-sum(shares*np.log(shares+1e-15))/math.log(6)})
    return pd.DataFrame(rows)


def h1_calendar_cpi(region_month):
    cpi=pd.read_csv(CPI_FILE).rename(columns={"year_month":"STRD_YYMM"})
    x=region_month.merge(cpi,on="STRD_YYMM",validate="many_to_one")
    x["days"]=pd.to_datetime(x.STRD_YYMM.astype(str)+"01").dt.days_in_month
    x["amt_per_day"]=x.amt/x.days; x["cnt_per_day"]=x.cnt/x.days
    x["real_amt"]=x.amt/(x.headline_cpi/100); x["real_amt_per_day"]=x.real_amt/x.days
    rows=[]
    for k,g in x.groupby(RKEY,sort=False):
        g=g.sort_values("STRD_YYMM"); row=dict(zip(RKEY,k))
        for col in ["amt","cnt","avg_ticket","amt_per_day","cnt_per_day","real_amt","real_amt_per_day"]:
            for kk,v in trend(g[col]).items(): row[f"{col}_{kk}"]=v
        rows.append(row)
    return pd.DataFrame(rows),x


def h2_ticket_mix(bc,codes,label):
    p=bc[bc.TP_BUZ_NO.isin(codes)].groupby(RKEY+["STRD_YYMM","TP_BUZ_NO"],as_index=False)[["amt","cnt"]].sum()
    counts=p.groupby(RKEY+["TP_BUZ_NO"]).STRD_YYMM.nunique()
    eligible=set(counts[counts==6].index)
    p=p[p.set_index(RKEY+["TP_BUZ_NO"]).index.isin(eligible)].copy()
    rows=[]
    for key,g in p.groupby(RKEY):
        if g.TP_BUZ_NO.nunique()!=len(codes): continue
        wide={m:g[g.STRD_YYMM==m].set_index("TP_BUZ_NO") for m in MONTHS}
        within=mix=total=0.0
        for a,b in zip(MONTHS[:-1],MONTHS[1:]):
            ga,gb=wide[a].loc[codes],wide[b].loc[codes]
            sa,sb=ga.cnt/ga.cnt.sum(),gb.cnt/gb.cnt.sum(); pa,pb=ga.amt/ga.cnt,gb.amt/gb.cnt
            within += (((sa+sb)/2)*(pb-pa)).sum(); mix += (((pa+pb)/2)*(sb-sa)).sum()
            total += gb.amt.sum()/gb.cnt.sum()-ga.amt.sum()/ga.cnt.sum()
        # 전반 3개월과 후반 3개월의 월평균 업종별 AMT/CNT로 별도 분해한다.
        a=g[g.STRD_YYMM.isin(MONTHS[:3])].groupby("TP_BUZ_NO")[["amt","cnt"]].mean().loc[codes]
        b=g[g.STRD_YYMM.isin(MONTHS[3:])].groupby("TP_BUZ_NO")[["amt","cnt"]].mean().loc[codes]
        sa,sb=a.cnt/a.cnt.sum(),b.cnt/b.cnt.sum(); pa,pb=a.amt/a.cnt,b.amt/b.cnt
        within3=(((sa+sb)/2)*(pb-pa)).sum(); mix3=(((pa+pb)/2)*(sb-sa)).sum()
        total3=b.amt.sum()/b.cnt.sum()-a.amt.sum()/a.cnt.sum()
        agg=g.groupby("STRD_YYMM")[["amt","cnt"]].sum(); tt=trend((agg.amt/agg.cnt).to_numpy())
        scale=abs(within)+abs(mix)
        rows.append({"coverage":label,"SIDO_NM":key[0],"CCG_NM":key[1],"industry_n":len(codes),"ticket_change_sum_5_intervals":total,"within_industry_effect":within,"between_industry_mix_effect":mix,"identity_error":total-within-mix,"within_abs_share":abs(within)/scale if scale else np.nan,"mix_abs_share":abs(mix)/scale if scale else np.nan,"first3_last3_ticket_change":total3,"first3_last3_within_effect":within3,"first3_last3_mix_effect":mix3,"first3_last3_identity_error":total3-within3-mix3,"ticket_ols_pct_m":tt["ols_pct_m"],"ticket_theil_beta":tt["theil_beta"],"ticket_robust_signs":tt["robust_signs"]})
    return pd.DataFrame(rows)


def h3_segments(bc):
    d=bc.copy(); d["segment"]=np.select([d.GENDER_CD.eq("x"),d.GENDER_CD.eq("3")],["법인","외국인"],default="국내개인")
    p=d.groupby(RKEY+["STRD_YYMM","segment"],as_index=False)[["amt","cnt"]].sum()
    totals=p.groupby(RKEY+["STRD_YYMM"])[["amt","cnt"]].transform("sum")
    p[["amt_share","cnt_share"]]=p[["amt","cnt"]].to_numpy()/totals.to_numpy()
    rows=[]
    for key,g in p.groupby(RKEY+["segment"]):
        if g.STRD_YYMM.nunique()!=6: continue
        g=g.sort_values("STRD_YYMM"); row=dict(zip(RKEY+["segment"],key))
        for col in ["amt","cnt"]:
            tr=trend(g[col]); sh=linear_trend(g[col+"_share"])
            row.update({f"{col}_growth_pct_m":tr["ols_pct_m"],f"{col}_robust":tr["robust_signs"],f"{col}_share_slope_pp_m":sh["slope"]*100})
        rows.append(row)
    return pd.DataFrame(rows),p


def h4_h6_population(region_month,bc,pop):
    x=region_month.merge(pop,on=RKEY+["STRD_YYMM"],validate="one_to_one")
    x["amt_pc"]=x.amt/x.population; x["cnt_pc"]=x.cnt/x.population
    human=bc[bc.AGE_CD.isin(AGE_NAMES)].groupby(RKEY+["STRD_YYMM","AGE_CD"],as_index=False).cnt.sum()
    human["share"]=human.cnt/human.groupby(RKEY+["STRD_YYMM"]).cnt.transform("sum")
    old=human[human.AGE_CD.eq("6")][RKEY+["STRD_YYMM","share"]].rename(columns={"share":"bc_60plus_share"})
    # AGE_CD 1=‘20대 이하’와 2=‘20대’가 겹치므로 중점값은 보조적 근사치다.
    age_midpoints={"1":15.0,"2":25.0,"3":35.0,"4":45.0,"5":55.0,"6":70.0}
    weighted=human.assign(age_midpoint=lambda q:q.AGE_CD.map(age_midpoints),weighted=lambda q:q.share*q.age_midpoint).groupby(RKEY+["STRD_YYMM"],as_index=False).weighted.sum().rename(columns={"weighted":"bc_weighted_age"})
    cons=human.assign(hhi=lambda q:q.share**2,ent=lambda q:-q.share*np.log(q.share)).groupby(RKEY+["STRD_YYMM"],as_index=False).agg(bc_age_hhi=("hhi","sum"),bc_age_entropy=("ent","sum"))
    cons.bc_age_entropy/=math.log(6)
    x=x.merge(old,on=RKEY+["STRD_YYMM"]).merge(weighted,on=RKEY+["STRD_YYMM"]).merge(cons,on=RKEY+["STRD_YYMM"])
    rows=[]
    for key,g in x.groupby(RKEY):
        g=g.sort_values("STRD_YYMM"); row=dict(zip(RKEY,key))
        for col in ["population","amt","cnt","amt_pc","cnt_pc"]:
            tr=trend(g[col]); row[f"{col}_growth_pct_m"]=tr["ols_pct_m"]; row[f"{col}_robust"]=tr["robust_signs"]
        for col in ["bc_60plus_share","pop_60plus_share","bc_weighted_age","pop_weighted_age","bc_age_hhi","pop_age_hhi","bc_age_entropy","pop_age_entropy"]:
            tr=linear_trend(g[col]); row[f"{col}_slope"]=tr["slope"]
        row["commercial_aging_gap_pp_m"]=(row["bc_60plus_share_slope"]-row["pop_60plus_share_slope"])*100
        row["weighted_age_gap_year_m"]=row["bc_weighted_age_slope"]-row["pop_weighted_age_slope"]
        row["hhi_gap_slope"]=row["bc_age_hhi_slope"]-row["pop_age_hhi_slope"]
        row["entropy_gap_slope"]=row["bc_age_entropy_slope"]-row["pop_age_entropy_slope"]
        row["amt_total_pc_quadrant"]=("총액+" if row["amt_growth_pct_m"]>=0 else "총액-")+("/1인당+" if row["amt_pc_growth_pct_m"]>=0 else "/1인당-")
        row["cnt_total_pc_quadrant"]=("총건수+" if row["cnt_growth_pct_m"]>=0 else "총건수-")+("/1인당+" if row["cnt_pc_growth_pct_m"]>=0 else "/1인당-")
        rows.append(row)
    return pd.DataFrame(rows),x


def h5_concentration_volatility(region_month,age_monthly):
    a=age_monthly.groupby(RKEY,as_index=False).agg(age_hhi_mean=("age_hhi","mean"),age_entropy_mean=("age_entropy_normalized","mean"))
    rows=[]
    for key,g in region_month.groupby(RKEY):
        g=g.sort_values("STRD_YYMM"); row=dict(zip(RKEY,key))
        for col in ["amt","cnt"]:
            ly=np.log(g[col].to_numpy()); fit=linregress(np.arange(6),ly); resid=ly-(fit.intercept+fit.slope*np.arange(6))
            row[f"{col}_residual_sd"]=np.std(resid,ddof=1); row[f"{col}_cv"]=g[col].std()/g[col].mean()
        row["total_cnt_6m"]=g.cnt.sum(); rows.append(row)
    z=a.merge(pd.DataFrame(rows),on=RKEY)
    tests=[]
    for c in ["age_hhi_mean","age_entropy_mean"]:
        for v in ["amt_residual_sd","cnt_residual_sd"]:
            rho,p=spearmanr(z[c],z[v]); tests.append({"concentration":c,"volatility":v,"spearman_rho":rho,"p":p,"n":len(z)})
    tests=pd.DataFrame(tests).sort_values("p")
    tests["p_fdr"]=np.minimum.accumulate((tests.p*len(tests)/np.arange(1,len(tests)+1))[::-1])[::-1].clip(upper=1)
    return z,tests


def load_nts_mapping():
    nts=pd.read_excel(NTS_FILE,sheet_name="국세청_시군구_101업종")
    mp=pd.read_excel(NTS_FILE,sheet_name="BC업종_매핑")
    mp=mp[mp["국세청업종"].notna()].copy(); mp["BC업종코드"]=mp.BC업종코드.astype(int)
    return nts,mp


def h7_h9_business(bc):
    nts,mp=load_nts_mapping()
    # BC 한식 3종은 국세청 한식음식점 하나와 비교하므로 먼저 합산한다.
    d=bc.merge(mp[["BC업종코드","연결업종","국세청업종","매핑수준"]],left_on="TP_BUZ_NO",right_on="BC업종코드",how="inner")
    b=d.groupby(RKEY+["STRD_YYMM","연결업종","국세청업종","매핑수준"],as_index=False)[["amt","cnt"]].sum()
    n=nts[nts.국세청업종.isin(mp.국세청업종.unique())][["기준년월","시도","시군구","국세청업종","가동_당월","신규_당월","폐업_당월","폐업강도_pct","가동_당월_마스킹","폐업_당월_마스킹"]].rename(columns={"기준년월":"STRD_YYMM","시도":"SIDO_NM","시군구":"CCG_NM"})
    z=b.merge(n,on=RKEY+["STRD_YYMM","국세청업종"],how="left",validate="one_to_one")
    z["amt_per_active_business"]=z.amt/z.가동_당월; z["cnt_per_active_business"]=z.cnt/z.가동_당월
    rows=[]
    for key,g in z.groupby(RKEY+["연결업종"]):
        g=g.sort_values("STRD_YYMM"); row=dict(zip(RKEY+["연결업종"],key)); row["매핑수준"]=g.매핑수준.iloc[0]
        g["avg_ticket"]=g.amt/g.cnt
        for col in ["amt","cnt","avg_ticket","가동_당월","amt_per_active_business","cnt_per_active_business"]:
            tr=trend(g[col]); row[f"{col}_growth_pct_m"]=tr["ols_pct_m"]; row[f"{col}_robust"]=tr["robust_signs"]
        row["initial_amt"]=g.amt.iloc[:3].mean(); row["initial_cnt"]=g.cnt.iloc[:3].mean()
        row["closure_observed_months"]=g.폐업_당월.notna().sum(); row["new_observed_months"]=g.신규_당월.notna().sum()
        rows.append(row)
    opp=pd.DataFrame(rows)
    opp["demand_supply_gap_cnt_pct_m"]=opp.cnt_growth_pct_m-opp["가동_당월_growth_pct_m"]
    opp["opportunity_quadrant"]=np.select([
        (opp.cnt_growth_pct_m>=0)&(opp["가동_당월_growth_pct_m"]<0),
        (opp.cnt_growth_pct_m>=0)&(opp["가동_당월_growth_pct_m"]>=0),
        (opp.cnt_growth_pct_m<0)&(opp["가동_당월_growth_pct_m"]>=0)],
        ["수요증가·공급감소","수요·공급동반증가","수요감소·공급증가"],default="수요·공급동반감소")
    # H9: 마스킹이 없는 가동사업자 순변화를 설명대상으로 사용. 폐업은 별도 커버리지 보고.
    model=opp.dropna(subset=["가동_당월_growth_pct_m"]).copy()
    # 사업체당 지표는 가동사업자 수를 분모로 가져 target leakage가 되므로 설명모형에서 제외한다.
    feats=["amt_growth_pct_m","cnt_growth_pct_m","avg_ticket_growth_pct_m","initial_amt","initial_cnt"]
    X=model[feats].replace([np.inf,-np.inf],np.nan).copy(); X[["initial_amt","initial_cnt"]]=np.log1p(X[["initial_amt","initial_cnt"]]); X=X.fillna(X.median())
    y=model["가동_당월_growth_pct_m"]; groups=model.SIDO_NM+"|"+model.CCG_NM
    cv=GroupKFold(5); rf=RandomForestRegressor(n_estimators=400,min_samples_leaf=10,random_state=42,n_jobs=-1)
    pred=cross_val_predict(rf,X,y,cv=cv,groups=groups); rf.fit(X,y); pi=permutation_importance(rf,X,y,n_repeats=20,random_state=42)
    imp=pd.DataFrame({"feature":feats,"rf_impurity_importance":rf.feature_importances_,"permutation_importance":pi.importances_mean}).sort_values("permutation_importance",ascending=False)
    quality={"rows":len(opp),"matched_month_rate":z.가동_당월.notna().mean(),"closure_observed_rate":z.폐업_당월.notna().mean(),"rf_cv_r2":r2_score(y,pred)}
    return opp,z,imp,quality,mp


def h10_base_effect(region_month):
    rows=[]
    for key,g in region_month.groupby(RKEY):
        g=g.sort_values("STRD_YYMM"); tr=trend(g.amt)
        rows.append({"SIDO_NM":key[0],"CCG_NM":key[1],"is_gun":key[1].endswith("군"),"initial_amt":g.amt.iloc[0],"first3_amt_mean":g.amt.iloc[:3].mean(),"amt_growth_pct_m":tr["ols_pct_m"],"robust":tr["robust_signs"]})
    d=pd.DataFrame(rows); tests=[]
    for group,q in [("전체",d),("군",d[d.is_gun]),("비군",d[~d.is_gun])]:
        for base in ["initial_amt","first3_amt_mean"]:
            rho,p=spearmanr(np.log(q[base]),q.amt_growth_pct_m); tests.append({"group":group,"base":base,"n":len(q),"spearman_rho":rho,"p":p})
    return d,pd.DataFrame(tests)


def h12_h13_portfolio(bc,codes,label):
    p=bc[bc.TP_BUZ_NO.isin(codes)].groupby(RKEY+["STRD_YYMM","TP_BUZ_NO"],as_index=False)[["amt","cnt"]].sum()
    rows=[]; monthly=[]
    for key,g in p.groupby(RKEY):
        if g.TP_BUZ_NO.nunique()!=len(codes) or len(g)!=6*len(codes): continue
        mats={}
        for metric in ["amt","cnt"]:
            mat=g.pivot(index="STRD_YYMM",columns="TP_BUZ_NO",values=metric).loc[MONTHS,codes]
            share=mat.div(mat.sum(axis=1),axis=0); mats[metric]=share
            for m,s in share.iterrows():
                monthly.append({"coverage":label,"SIDO_NM":key[0],"CCG_NM":key[1],"STRD_YYMM":m,"metric":metric,"hhi":sum(s*s),"entropy":-sum(s*np.log(s))/math.log(len(codes)),"top1":s.max(),"top3":s.nlargest(3).sum()})
        row={"coverage":label,"SIDO_NM":key[0],"CCG_NM":key[1]}
        for metric,s in mats.items():
            js=[jensenshannon(s.iloc[i-1],s.iloc[i]) for i in range(1,6)]
            cs=[cosine(s.iloc[i-1],s.iloc[i]) for i in range(1,6)]
            a=s.iloc[:3].mean(); b=s.iloc[3:].mean()
            row.update({f"{metric}_js_mean_adjacent":np.mean(js),f"{metric}_js_first3_last3":jensenshannon(a,b),f"{metric}_cosine_mean_adjacent":np.mean(cs),f"{metric}_cosine_first3_last3":cosine(a,b)})
        rows.append(row)
    mon=pd.DataFrame(monthly); shifts=pd.DataFrame(rows)
    tr=[]
    for key,g in mon.groupby(["coverage"]+RKEY+["metric"]):
        g=g.sort_values("STRD_YYMM"); row=dict(zip(["coverage"]+RKEY+["metric"],key))
        for c in ["hhi","entropy","top1","top3"]: row[c+"_slope"]=linear_trend(g[c])["slope"]
        tr.append(row)
    return pd.DataFrame(tr),mon,shifts


def h15_quality(h1,pop_metrics,opp,segments,portfolio):
    d=h1.merge(pop_metrics,on=RKEY,suffixes=("","_pop"))
    store=opp.groupby(RKEY,as_index=False).agg(per_store_cnt_growth=("cnt_per_active_business_growth_pct_m","median"),demand_supply_gap=("demand_supply_gap_cnt_pct_m","median"))
    foreign=segments[segments.segment.eq("외국인")][RKEY+["amt_share_slope_pp_m"]].rename(columns={"amt_share_slope_pp_m":"foreign_amt_share_slope_pp_m"})
    port=portfolio[portfolio.coverage.eq("7개전국공통") & portfolio.metric.eq("cnt")][RKEY+["hhi_slope","entropy_slope"]]
    d=d.merge(store,on=RKEY,how="left").merge(foreign,on=RKEY,how="left").merge(port,on=RKEY,how="left")
    d["real_per_day_growth"]=d.real_amt_per_day_ols_pct_m
    conditions=[
        (d.real_per_day_growth>0)&(d.cnt_per_day_ols_pct_m>0)&(d.per_store_cnt_growth>0)&(d.commercial_aging_gap_pp_m<=0),
        (d.real_per_day_growth>0)&(d.cnt_per_day_ols_pct_m>0),
        (d.real_per_day_growth<=0)&(d.cnt_per_day_ols_pct_m>0),
        (d.real_per_day_growth>0)&(d.cnt_per_day_ols_pct_m<=0)]
    d["growth_quality_type"]=np.select(conditions,["다축 건강성장 후보","거래량 기반 성장","실질금액 약화·건수 증가","금액 중심 성장"],default="동반 약화/정체")
    return d


def save_plot(h1,mix,popm,opp,base,portfolio):
    configure_plot(); FIG.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(8,6)); ax.scatter(h1.cnt_per_day_ols_pct_m,h1.real_amt_per_day_ols_pct_m,s=18,alpha=.65); ax.axhline(0,c="grey",lw=.8); ax.axvline(0,c="grey",lw=.8); ax.set(xlabel="일평균 CNT 성장률(%/월)",ylabel="일평균 실질 AMT 성장률(%/월)",title="달력일수·CPI 조정 후 지역 성장"); fig.tight_layout(); fig.savefig(FIG/"01_real_daily_growth.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,6)); ax.scatter(mix.within_industry_effect,mix.between_industry_mix_effect,s=18,alpha=.65); ax.axhline(0,c="grey",lw=.8); ax.axvline(0,c="grey",lw=.8); ax.set(xlabel="업종내 객단가 효과(원)",ylabel="업종구성 효과(원)",title="객단가 변화의 업종내·업종간 분해"); fig.tight_layout(); fig.savefig(FIG/"02_ticket_mix_decomposition.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,6)); ax.scatter(popm.pop_60plus_share_slope*100,popm.bc_60plus_share_slope*100,s=18,alpha=.65); lo=min(ax.get_xlim()[0],ax.get_ylim()[0]); hi=max(ax.get_xlim()[1],ax.get_ylim()[1]); ax.plot([lo,hi],[lo,hi],ls="--",c="grey"); ax.set(xlabel="주민 60+ 점유율 변화(%p/월)",ylabel="소비 CNT 60+ 점유율 변화(%p/월)",title="주민 고령화와 소비 고령화"); fig.tight_layout(); fig.savefig(FIG/"03_commercial_aging_gap.png",dpi=180); plt.close(fig)
    q=opp.opportunity_quadrant.value_counts(); fig,ax=plt.subplots(figsize=(8,5)); q.plot.bar(ax=ax,color="#4C78A8"); ax.set(title="수요(CNT)·공급(가동사업자) 변화 유형",ylabel="지역×매핑업종 수",xlabel=""); ax.tick_params(axis="x",rotation=20); fig.tight_layout(); fig.savefig(FIG/"04_demand_supply_quadrants.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,6)); colors=np.where(base.is_gun,"#E45756","#4C78A8"); ax.scatter(np.log10(base.first3_amt_mean),base.amt_growth_pct_m,c=colors,s=20,alpha=.7); ax.set(xlabel="초기 3개월 평균 AMT(log10)",ylabel="AMT 성장률(%/월)",title="초기 규모와 성장률: 군/비군"); fig.tight_layout(); fig.savefig(FIG/"05_base_effect.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,6)); ax.scatter(portfolio.amt_js_first3_last3,portfolio.cnt_js_first3_last3,s=18,alpha=.65); ax.set(xlabel="AMT 포트폴리오 JS 거리",ylabel="CNT 포트폴리오 JS 거리",title="업종 포트폴리오 이동 거리"); fig.tight_layout(); fig.savefig(FIG/"06_portfolio_shift.png",dpi=180); plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    bc=load_bc(); region_month=bc.groupby(RKEY+["STRD_YYMM"],as_index=False)[["amt","cnt"]].sum(); region_month["avg_ticket"]=region_month.amt/region_month.cnt
    pop=parse_population(bc[RKEY].drop_duplicates())
    h1,h1monthly=h1_calendar_cpi(region_month)
    mix7=h2_ticket_mix(bc,CORE7,"7개전국공통"); mix8=h2_ticket_mix(bc,FULL_COVERAGE,"8개대형할인포함"); mix9=h2_ticket_mix(bc,HIGH_COVERAGE,"9개일식포함"); mix11=h2_ticket_mix(bc,ALL11,"11개완전"); mix=pd.concat([mix7,mix8,mix9,mix11],ignore_index=True)
    seg,segmonthly=h3_segments(bc)
    popm,popmonthly=h4_h6_population(region_month,bc,pop)
    age_metrics,age_monthly,age_shares=age_concentration(bc)
    vol,voltests=h5_concentration_volatility(region_month,age_monthly)
    opp,oppmonthly,importance,business_quality,mapping=h7_h9_business(bc)
    base,basetests=h10_base_effect(region_month)
    div7,divmon7,shift7=h12_h13_portfolio(bc,CORE7,"7개전국공통"); div8,divmon8,shift8=h12_h13_portfolio(bc,FULL_COVERAGE,"8개대형할인포함"); div9,divmon9,shift9=h12_h13_portfolio(bc,HIGH_COVERAGE,"9개일식포함"); div11,divmon11,shift11=h12_h13_portfolio(bc,ALL11,"11개완전")
    diversity=pd.concat([div7,div8,div9,div11],ignore_index=True); divmonthly=pd.concat([divmon7,divmon8,divmon9,divmon11],ignore_index=True); shifts=pd.concat([shift7,shift8,shift9,shift11],ignore_index=True)
    quality=h15_quality(h1,popm,opp,seg,diversity)
    tables={
      "01_h1_calendar_cpi_region.csv":h1,"02_h1_adjusted_monthly.csv":h1monthly,"03_h2_ticket_mix_decomposition.csv":mix,
      "04_h3_segment_region.csv":seg,"05_h3_segment_monthly.csv":segmonthly,"06_h4_h6_population_consumption.csv":popm,
      "07_h4_h6_monthly_panel.csv":popmonthly,"08_h5_concentration_volatility.csv":vol,"09_h5_correlation_tests.csv":voltests,
      "10_h7_h8_business_opportunity.csv":opp,"11_h7_h8_monthly_matched.csv":oppmonthly,"12_h9_feature_importance.csv":importance,
      "13_h10_base_effect_regions.csv":base,"14_h10_base_effect_tests.csv":basetests,"15_h12_diversity_trends.csv":diversity,
      "16_h12_diversity_monthly.csv":divmonthly,"17_h13_portfolio_shift.csv":shifts,"18_h15_growth_quality.csv":quality,
      "19_external_population_monthly.csv":pop,"20_bc_nts_industry_mapping.csv":mapping}
    # 사람이 해석·평가한 점수표가 있으면 통합 Excel에도 함께 수록한다.
    for name in ["21_hypothesis_scorecard.csv", "22_top5_patterns.csv"]:
        path=OUT/name
        if path.exists(): tables[name]=pd.read_csv(path)
    for name,df in tables.items(): df.to_csv(OUT/name,index=False,encoding="utf-8-sig")
    with pd.ExcelWriter(OUT/"third_exploration_tables.xlsx",engine="openpyxl") as w:
        for name,df in tables.items(): df.to_excel(w,sheet_name=name[:27],index=False)
    save_plot(h1,mix7,popm,opp,base,shift7)
    summary={
      "rows_bc":len(bc),"regions":region_month[RKEY].drop_duplicates().shape[0],
      "H1_nominal_positive":int((h1.amt_ols_pct_m>0).sum()),"H1_real_daily_positive":int((h1.real_amt_per_day_ols_pct_m>0).sum()),
      "H1_sign_changed_nominal_to_real_daily":int((np.sign(h1.amt_ols_pct_m)!=np.sign(h1.real_amt_per_day_ols_pct_m)).sum()),
      "H2_within_negative":int((mix7.within_industry_effect<0).sum()),"H2_mix_negative":int((mix7.between_industry_mix_effect<0).sum()),
      "H2_within_dominant":int((mix7.within_abs_share>.5).sum()),"H2_regions":len(mix7),
      "H3_foreign_share_rising":int((seg[seg.segment.eq('외국인')].amt_share_slope_pp_m>0).sum()),
      "H3_corporate_share_rising":int((seg[seg.segment.eq('법인')].amt_share_slope_pp_m>0).sum()),
      "H4_positive_commercial_aging_gap":int((popm.commercial_aging_gap_pp_m>0).sum()),
      "H6_amt_quadrants":popm.amt_total_pc_quadrant.value_counts().to_dict(),"H6_cnt_quadrants":popm.cnt_total_pc_quadrant.value_counts().to_dict(),
      "H7_H8_quadrants":opp.opportunity_quadrant.value_counts().to_dict(),"H9_quality":business_quality,
      "H10":basetests.to_dict('records'),"H12_concentration_rising":int(((div7.metric=='cnt')&(div7.hhi_slope>0)&(div7.entropy_slope<0)).sum()),
      "H13_top10":shift7.nlargest(10,'cnt_js_first3_last3')[RKEY+['cnt_js_first3_last3']].to_dict('records'),
      "H15_types":quality.growth_quality_type.value_counts().to_dict(),
      "limitations":{"H11":"외국인 소비는 분석했으나 국내 관광객/거주민 분리는 관광 데이터랩 원자료 미확보로 불가","H14":"공식 시군구 경계 파일이 별도 로그인 다운로드라 Moran's I 미검증","closure":"시군구 폐업건수 마스킹률이 매우 높아 폐업 예측 대신 가동사업자 변화 설명"}}
    (OUT/"analysis_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
