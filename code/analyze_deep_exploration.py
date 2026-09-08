"""BC카드 3차 심층 탐색: H1~H18 재현 파이프라인.

실행: python3 scripts/analyze_deep_exploration.py
출력: analysis/deep_exploration/
"""
from __future__ import annotations

import json, math, os, tempfile, warnings
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TMP=Path(tempfile.gettempdir())/"bc_deep_exploration"; TMP.mkdir(parents=True,exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR",str(TMP/"mpl")); os.environ.setdefault("XDG_CACHE_HOME",str(TMP/"xdg"))
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import linregress, spearmanr, mannwhitneyu
from sklearn.cluster import AgglomerativeClustering
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score, r2_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analyze_growth_illusion import AGE_NAMES, configure_plot, bh_adjust, build_slope_table
from analyze_consumption_patterns import age_concentration
from analyze_third_exploration import (
    RKEY, MONTHS, CORE7, ALL11, trend, linear_trend, load_bc, parse_population,
    h2_ticket_mix, h3_segments, h4_h6_population, h5_concentration_volatility,
    h10_base_effect, h12_h13_portfolio,
)
from build_localdata_closure_panel import normalize_file, region_lookup

OUT=ROOT/"analysis"/"deep_exploration"; FIG=OUT/"figures"
DETAILED_CPI=ROOT/"dataset"/"external_raw"/"kosis"/"202601_202606_detailed_cpi.csv"
CPI_MAP=ROOT/"dataset"/"external_raw"/"kosis"/"bc_cpi_mapping.csv"
LOCALDATA=ROOT/"dataset"/"processed"/"LOCALDATA_식품업종_개폐업_202601_202606.xlsx"
LOCAL_MAP=ROOT/"dataset"/"external_raw"/"localdata"/"bc_industry_mapping.csv"


def fdr_table(df,pcol="p"):
    z=df.copy(); z["p_fdr"]=bh_adjust(z[pcol]); return z


def h1_real_industry(bc):
    cpi=pd.read_csv(DETAILED_CPI).rename(columns={"year_month":"STRD_YYMM"})
    mp=pd.read_csv(CPI_MAP)
    p=bc.groupby(RKEY+["STRD_YYMM","TP_BUZ_NO","TP_BUZ_NM"],as_index=False)[["amt","cnt"]].sum()
    p=p.merge(mp[["BC_TP_BUZ_NO","cpi_series","confidence"]],left_on="TP_BUZ_NO",right_on="BC_TP_BUZ_NO",validate="many_to_one").merge(cpi,on="STRD_YYMM",validate="many_to_one")
    p["cpi_value"]=p.apply(lambda r:r[r.cpi_series],axis=1); p["real_amt"]=p.amt/(p.cpi_value/100)
    p["ticket"]=p.amt/p.cnt; p["days"]=pd.to_datetime(p.STRD_YYMM.astype(str)+"01").dt.days_in_month
    for c in ["amt","real_amt","cnt"]: p[c+"_daily"]=p[c]/p.days
    rows=[]
    for key,g in p.groupby(RKEY+["TP_BUZ_NO","TP_BUZ_NM"]):
        if g.STRD_YYMM.nunique()!=6: continue
        g=g.sort_values("STRD_YYMM"); row=dict(zip(RKEY+["TP_BUZ_NO","TP_BUZ_NM"],key)); row["cpi_mapping_confidence"]=g.confidence.iloc[0]
        for c in ["amt","real_amt","cnt","ticket","amt_daily","real_amt_daily","cnt_daily"]:
            tr=trend(g[c]); row[f"{c}_growth_pct_m"]=tr["ols_pct_m"]; row[f"{c}_r2"]=tr["r2"]; row[f"{c}_robust_signs"]=tr["robust_signs"]
        row["nominal_real_type"]=("명목+" if row["amt_growth_pct_m"]>=0 else "명목-")+("/실질+" if row["real_amt_growth_pct_m"]>=0 else "/실질-")
        row["total_cnt_6m"]=g.cnt.sum(); rows.append(row)
    return pd.DataFrame(rows),p


def h4_growth_contribution(bc):
    d=bc.copy(); d["segment"]=np.select([d.GENDER_CD.eq("x"),d.GENDER_CD.eq("3")],["법인","외국인"],default="국내개인")
    p=d.groupby(RKEY+["STRD_YYMM","TP_BUZ_NO","TP_BUZ_NM","segment"],as_index=False)[["amt","cnt"]].sum()
    rows=[]
    for key,g in p.groupby(RKEY+["TP_BUZ_NO","TP_BUZ_NM"]):
        pivot={metric:g.pivot_table(index="STRD_YYMM",columns="segment",values=metric,aggfunc="sum").reindex(index=MONTHS,columns=["국내개인","외국인","법인"]) for metric in ["amt","cnt"]}
        if any(w.isna().any().any() for w in pivot.values()):
            continue
        row=dict(zip(RKEY+["TP_BUZ_NO","TP_BUZ_NM"],key))
        for metric,w in pivot.items():
            total=w.sum(axis=1); total_slope=linregress(range(6),total).slope; row[f"{metric}_raw_slope_total"]=total_slope
            for s in ["국내개인","외국인","법인"]:
                y=w[s].to_numpy() if s in w else np.zeros(6); sl=linregress(range(6),y).slope
                row[f"{metric}_{s}_raw_slope"]=sl; row[f"{metric}_{s}_signed_contribution_pct"]=sl/total_slope*100 if abs(total_slope)>1e-9 else np.nan
                row[f"{metric}_{s}_growth_pct_m"]=trend(y)["ols_pct_m"] if np.all(y>0) else np.nan
            row[f"{metric}_identity_error"]=total_slope-sum(row[f"{metric}_{s}_raw_slope"] for s in ["국내개인","외국인","법인"])
        rows.append(row)
    return pd.DataFrame(rows),p


def h7_age_industry_volatility(bc):
    h=bc[bc.AGE_CD.isin(AGE_NAMES)].groupby(RKEY+["STRD_YYMM","TP_BUZ_NO","AGE_CD"],as_index=False)[["amt","cnt"]].sum()
    total=h.groupby(RKEY+["STRD_YYMM","TP_BUZ_NO"])[["amt","cnt"]].transform("sum")
    h["share"]=h.cnt/total.cnt
    conc=h.assign(hhi=lambda q:q.share**2,ent=lambda q:-q.share*np.log(q.share)).groupby(RKEY+["STRD_YYMM","TP_BUZ_NO"],as_index=False).agg(age_hhi=("hhi","sum"),age_entropy=("ent","sum")); conc.age_entropy/=math.log(6)
    monthly=bc.groupby(RKEY+["STRD_YYMM","TP_BUZ_NO"],as_index=False)[["amt","cnt"]].sum()
    rows=[]
    for key,g in monthly.groupby(RKEY+["TP_BUZ_NO"]):
        if g.STRD_YYMM.nunique()!=6: continue
        g=g.sort_values("STRD_YYMM"); row=dict(zip(RKEY+["TP_BUZ_NO"],key))
        for c in ["amt","cnt"]:
            ly=np.log(g[c]); o=linregress(range(6),ly); row[c+"_residual_sd"]=np.std(ly-(o.intercept+o.slope*np.arange(6)),ddof=1)
        rows.append(row)
    z=pd.DataFrame(rows).merge(conc.groupby(RKEY+["TP_BUZ_NO"],as_index=False).agg(age_hhi_mean=("age_hhi","mean"),age_entropy_mean=("age_entropy","mean")),on=RKEY+["TP_BUZ_NO"])
    tests=[]
    for industry,g in list(z.groupby("TP_BUZ_NO"))+[("전체",z)]:
        for a in ["age_hhi_mean","age_entropy_mean"]:
            for v in ["amt_residual_sd","cnt_residual_sd"]:
                rho,p=spearmanr(g[a],g[v]); tests.append({"industry":industry,"concentration":a,"volatility":v,"n":len(g),"rho":rho,"p":p})
    return z,fdr_table(pd.DataFrame(tests))


def localdata_integrate(bc,pop):
    ld=pd.read_excel(LOCALDATA,sheet_name="시군구_월별_개폐업").rename(columns={"기준년월":"STRD_YYMM","시도":"SIDO_NM","시군구":"CCG_NM"})
    # BC 업종을 LOCALDATA 연결업종으로 묶는다.
    map_code={8001:"한식통합",8002:"한식통합",8003:"한식통합",8004:"일식회집",8005:"중국음식",8006:"서양음식",8021:"스넥",8301:"제과점"}
    conf={"한식통합":"MEDIUM","일식회집":"MEDIUM","중국음식":"HIGH","서양음식":"MEDIUM","스넥":"MEDIUM","제과점":"HIGH"}
    b=bc[bc.TP_BUZ_NO.isin(map_code)].assign(연결업종=lambda q:q.TP_BUZ_NO.map(map_code)).groupby(RKEY+["STRD_YYMM","연결업종"],as_index=False)[["amt","cnt"]].sum()
    z=b.merge(ld,on=RKEY+["STRD_YYMM","연결업종"],how="inner",validate="one_to_one").merge(pop[RKEY+["STRD_YYMM","population"]],on=RKEY+["STRD_YYMM"],validate="many_to_one")
    cpi=pd.read_csv(DETAILED_CPI).rename(columns={"year_month":"STRD_YYMM"}); z=z.merge(cpi,on="STRD_YYMM",validate="many_to_one")
    z["cpi_value"]=np.where(z.연결업종.eq("제과점"),z.bread_cereals_cpi,z.restaurants_accommodation_cpi)
    z["real_amt"]=z.amt/(z.cpi_value/100)
    z["confidence"]=z.연결업종.map(conf); z["ticket"]=z.amt/z.cnt; z["amt_pc"]=z.amt/z.population; z["cnt_pc"]=z.cnt/z.population
    z["real_amt_pc"]=z.real_amt/z.population; z["amt_per_store"]=z.amt/z.가동_당월말.replace(0,np.nan); z["real_amt_per_store"]=z.real_amt/z.가동_당월말.replace(0,np.nan); z["cnt_per_store"]=z.cnt/z.가동_당월말.replace(0,np.nan)
    rows=[]
    for key,g in z.groupby(RKEY+["연결업종"]):
        if g.STRD_YYMM.nunique()!=6 or (g.가동_당월말<=0).any(): continue
        g=g.sort_values("STRD_YYMM"); row=dict(zip(RKEY+["연결업종"],key)); row["confidence"]=g.confidence.iloc[0]
        for c in ["amt","real_amt","cnt","ticket","amt_pc","real_amt_pc","cnt_pc","가동_당월말","amt_per_store","real_amt_per_store","cnt_per_store"]:
            tr=trend(g[c]); row[c+"_growth_pct_m"]=tr["ols_pct_m"]; row[c+"_robust_signs"]=tr["robust_signs"]
        first=g.가동_전월말.iloc[0]; row["initial_stores"]=first; row["openings_6m"]=g.신규_당월.sum(); row["closures_6m"]=g.폐업_당월.sum()
        row["total_cnt_6m"]=g.cnt.sum(); row["total_amt_6m"]=g.amt.sum()
        row["mean_amt_per_initial_store"]=g.amt.mean()/first if first else np.nan; row["mean_cnt_per_initial_store"]=g.cnt.mean()/first if first else np.nan
        row["opening_rate_6m_pct"]=row["openings_6m"]/first*100 if first else np.nan; row["closure_rate_6m_pct"]=row["closures_6m"]/first*100 if first else np.nan
        row["net_store_change_pct"]=(g.가동_당월말.iloc[-1]-first)/first*100 if first else np.nan
        row["demand_supply_gap_cnt"]=row["cnt_growth_pct_m"]-row["가동_당월말_growth_pct_m"]
        row["demand_supply_type"]=("수요+" if row["cnt_growth_pct_m"]>=0 else "수요-")+("/공급+" if row["가동_당월말_growth_pct_m"]>=0 else "/공급-")
        if row["amt_growth_pct_m"]>=0 and row["가동_당월말_growth_pct_m"]>=0 and row["amt_per_store_growth_pct_m"]>=0:
            row["market_store_type"]="A 총시장·점포·점포당 증가"
        elif row["amt_growth_pct_m"]>=0 and row["가동_당월말_growth_pct_m"]>=0:
            row["market_store_type"]="B 총시장·점포 증가/점포당 감소"
        elif row["amt_growth_pct_m"]<0 and row["가동_당월말_growth_pct_m"]>=0:
            row["market_store_type"]="C 총시장 감소/점포 증가"
        else:
            row["market_store_type"]="D 기타·동반감소"
        row["market_store_type_robust"] = row["market_store_type"] if (
            row["amt_robust_signs"] and row["가동_당월말_robust_signs"] and row["amt_per_store_robust_signs"]
        ) else "부호 강건성 미충족"
        row["demand_supply_type_robust"] = row["demand_supply_type"] if (
            row["cnt_robust_signs"] and row["가동_당월말_robust_signs"]
        ) else "부호 강건성 미충족"
        rows.append(row)
    return pd.DataFrame(rows),z


def h14_admin_base_effect(base):
    """최종 행정구역명 기준 시·구·군을 분리한 초기규모 효과."""
    d=base.copy()
    d["admin_type"]=np.select(
        [d.CCG_NM.str.endswith("군"),d.CCG_NM.str.endswith("구")],
        ["군","구"],default="시"
    )
    rows=[]
    for typ,g in [("전체",d),*list(d.groupby("admin_type"))]:
        for base_col in ["initial_amt","first3_amt_mean"]:
            rho,p=spearmanr(g[base_col],g.amt_growth_pct_m)
            rows.append({"admin_type":typ,"base":base_col,"n":len(g),"spearman_rho":rho,"p":p})
    return d,fdr_table(pd.DataFrame(rows))


def store_sensitivity(stores):
    """거래량과 추세 부호 강건성에 따른 H8/H9 유형별 민감도."""
    rows=[]
    for conf in ["HIGH","MEDIUM","ALL"]:
        q=stores if conf=="ALL" else stores[stores.confidence.eq(conf)]
        for threshold in [0,1_000,10_000,100_000,500_000]:
            z=q[q.total_cnt_6m>=threshold]
            for dimension,col in [("H8_market_store","market_store_type"),("H9_demand_supply","demand_supply_type")]:
                for typ,n in z[col].value_counts().items():
                    rows.append({"confidence":conf,"cnt_threshold":threshold,"dimension":dimension,"type":typ,"n":int(n),"denominator":len(z),"share_pct":n/len(z)*100 if len(z) else np.nan})
    return pd.DataFrame(rows)


def breadth(bc):
    slopes,_=build_slope_table(bc)
    q=slopes[slopes.complete_6m].copy()
    q["amt_strengthened"]=q.relative_amt_slope_pct_month>0; q["cnt_strengthened"]=q.relative_cnt_slope_pct_month>0
    out=q.groupby(RKEY,as_index=False).agg(observed_industries=("TP_BUZ_NO","nunique"),amt_strengthened_count=("amt_strengthened","sum"),cnt_strengthened_count=("cnt_strengthened","sum"))
    for x in ["amt","cnt"]: out[x+"_strengthened_fraction"]=out[x+"_strengthened_count"]/out.observed_industries; out[x+"_weakened_fraction"]=1-out[x+"_strengthened_fraction"]
    return out,q


def model_comparison(features,outcome,groups):
    specs={
      "A_AMT":["amt_growth_pct_m"],
      "B_AMT_CNT":["amt_growth_pct_m","cnt_growth_pct_m"],
      "C_BC_structure":["amt_growth_pct_m","cnt_growth_pct_m","ticket_growth_pct_m","foreign_share_slope","corp_share_slope","bc_age_hhi_slope","industry_hhi_slope","portfolio_js","cnt_weakened_fraction"],
      "D_BC_external":["amt_growth_pct_m","cnt_growth_pct_m","ticket_growth_pct_m","foreign_share_slope","corp_share_slope","bc_age_hhi_slope","industry_hhi_slope","portfolio_js","cnt_weakened_fraction","real_amt_growth_pct_m","amt_pc_growth_pct_m","cnt_pc_growth_pct_m","commercial_aging_gap_pp_m","population_growth_pct_m","initial_stores","mean_amt_per_initial_store","mean_cnt_per_initial_store"]}
    rows=[]; importances=[]; cv=GroupKFold(5)
    for name,cols in specs.items():
        dat=features[cols+[outcome]].replace([np.inf,-np.inf],np.nan).dropna(); idx=dat.index; X=dat[cols]; y=dat[outcome]; gr=groups.loc[idx]
        ridge=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),Ridge(alpha=10))
        rf=make_pipeline(SimpleImputer(strategy="median"),RandomForestRegressor(n_estimators=500,min_samples_leaf=10,max_features=.8,random_state=42,n_jobs=-1))
        for alg,model in [("Ridge",ridge),("RF",rf)]:
            pred=cross_val_predict(model,X,y,cv=cv,groups=gr); rows.append({"outcome":outcome,"model":name,"algorithm":alg,"n":len(y),"cv_r2":r2_score(y,pred),"cv_mae":np.mean(abs(y-pred))})
        if name=="D_BC_external":
            rf.fit(X,y); pi=permutation_importance(rf,X,y,n_repeats=20,random_state=42)
            for c,v in zip(cols,pi.importances_mean): importances.append({"outcome":outcome,"feature":c,"permutation_importance_in_sample":v})
    return pd.DataFrame(rows),pd.DataFrame(importances).sort_values(["outcome","permutation_importance_in_sample"],ascending=[True,False])


def residual_growth(base,popm,segments,stores,diversity,shift,bread):
    d=base.copy(); x=np.log(d.first3_amt_mean); isgun=d.is_gun.astype(int)
    X=np.c_[np.ones(len(d)),x,isgun,x*isgun]; beta=np.linalg.lstsq(X,d.amt_growth_pct_m,rcond=None)[0]; d["residual_growth"]=d.amt_growth_pct_m-X@beta
    foreign=segments[segments.segment.eq("외국인")][RKEY+["amt_share_slope_pp_m"]].rename(columns={"amt_share_slope_pp_m":"foreign_share_slope"})
    corp=segments[segments.segment.eq("법인")][RKEY+["amt_share_slope_pp_m"]].rename(columns={"amt_share_slope_pp_m":"corp_share_slope"})
    st=stores.groupby(RKEY,as_index=False).agg(store_growth=("가동_당월말_growth_pct_m","median"),perstore_growth=("cnt_per_store_growth_pct_m","median"),closure_rate=("closure_rate_6m_pct","median"))
    dv=diversity[(diversity.coverage.eq("7개전국공통"))&(diversity.metric.eq("cnt"))][RKEY+["hhi_slope","entropy_slope"]]
    sh=shift[shift.coverage.eq("7개전국공통")][RKEY+["cnt_js_first3_last3"]]
    d=d.merge(popm[RKEY+["population_growth_pct_m","commercial_aging_gap_pp_m"]],on=RKEY).merge(foreign,on=RKEY).merge(corp,on=RKEY).merge(st,on=RKEY,how="left").merge(dv,on=RKEY,how="left").merge(sh,on=RKEY,how="left").merge(bread,on=RKEY,how="left")
    specs={
      "A_external_context":["population_growth_pct_m","commercial_aging_gap_pp_m","foreign_share_slope","corp_share_slope","store_growth","closure_rate"],
      "B_plus_structure":["population_growth_pct_m","commercial_aging_gap_pp_m","foreign_share_slope","corp_share_slope","store_growth","closure_rate","hhi_slope","cnt_js_first3_last3","cnt_weakened_fraction"],
      "C_plus_perstore_demand":["population_growth_pct_m","commercial_aging_gap_pp_m","foreign_share_slope","corp_share_slope","store_growth","closure_rate","hhi_slope","cnt_js_first3_last3","cnt_weakened_fraction","perstore_growth"]}
    g=d[d.is_gun].copy(); y=g.residual_growth; cv=GroupKFold(5); groups=g.SIDO_NM; comp=[]
    for name,cols in specs.items():
        X=g[cols].replace([np.inf,-np.inf],np.nan)
        rf=make_pipeline(SimpleImputer(strategy="median"),RandomForestRegressor(n_estimators=500,min_samples_leaf=5,random_state=42,n_jobs=-1,max_features=.8))
        pred=cross_val_predict(rf,X,y,cv=cv,groups=groups)
        comp.append({"model":name,"n":len(g),"cv_r2":r2_score(y,pred),"cv_mae":np.mean(abs(y-pred))})
    cols=specs["C_plus_perstore_demand"]; X=g[cols].replace([np.inf,-np.inf],np.nan)
    rf.fit(X,y); pi=permutation_importance(rf,X,y,n_repeats=30,random_state=42)
    imp=pd.DataFrame({"feature":cols,"importance":pi.importances_mean}).sort_values("importance",ascending=False)
    return d,imp,pd.DataFrame(comp),{"n_gun":len(g),"rf_cv_r2_full":comp[-1]["cv_r2"],"rf_cv_r2_without_perstore":comp[1]["cv_r2"],"base_coefficients":beta.tolist()}


def structure_associations(diversity,shift,bread,popm,age_vol,stores):
    div=diversity[(diversity.coverage.eq("7개전국공통"))&(diversity.metric.eq("cnt"))][RKEY+["hhi_slope","entropy_slope","top1_slope","top3_slope"]]
    sh=shift[shift.coverage.eq("7개전국공통")][RKEY+["cnt_js_first3_last3"]]
    st=stores.groupby(RKEY,as_index=False).agg(closure_rate=("closure_rate_6m_pct","median"),store_growth=("가동_당월말_growth_pct_m","median"),perstore_growth=("cnt_per_store_growth_pct_m","median"))
    d=div.merge(sh,on=RKEY).merge(bread,on=RKEY).merge(popm[RKEY+["amt_growth_pct_m","cnt_growth_pct_m","population_growth_pct_m","commercial_aging_gap_pp_m"]],on=RKEY).merge(age_vol[RKEY+["amt_residual_sd","cnt_residual_sd"]],on=RKEY).merge(st,on=RKEY,how="left")
    tests=[]
    outcomes=["amt_growth_pct_m","cnt_growth_pct_m","population_growth_pct_m","commercial_aging_gap_pp_m","amt_residual_sd","cnt_residual_sd","closure_rate","store_growth","perstore_growth"]
    for structural in ["hhi_slope","entropy_slope","cnt_js_first3_last3","cnt_weakened_fraction"]:
        for y in outcomes:
            q=d[[structural,y]].dropna(); rho,p=spearmanr(q[structural],q[y]); tests.append({"structural_metric":structural,"outcome":y,"n":len(q),"spearman_rho":rho,"p":p})
    tests=fdr_table(pd.DataFrame(tests))
    # 포트폴리오 이동 상위 10%와 나머지의 특성 차이.
    cutoff=d.cnt_js_first3_last3.quantile(.9); d["portfolio_top10pct"]=d.cnt_js_first3_last3>=cutoff; comps=[]
    for y in outcomes+["hhi_slope","cnt_weakened_fraction"]:
        a=d.loc[d.portfolio_top10pct,y].dropna(); b=d.loc[~d.portfolio_top10pct,y].dropna(); u=mannwhitneyu(a,b)
        comps.append({"feature":y,"top_n":len(a),"rest_n":len(b),"top_median":a.median(),"rest_median":b.median(),"median_difference":a.median()-b.median(),"rank_biserial":2*u.statistic/(len(a)*len(b))-1,"p":u.pvalue})
    return d,tests,fdr_table(pd.DataFrame(comps))


def region_centroids():
    prefixes,lookup=region_lookup(); path=ROOT/"dataset"/"localdata_raw"/"제과점.csv"; parts=[]
    for z in normalize_file(path,prefixes,lookup):
        q=z.dropna(subset=["시도","시군구","좌표X","좌표Y"])
        q=q[q.좌표X.between(50000,400000)&q.좌표Y.between(50000,700000)]
        parts.append(q.groupby(["시도","시군구"],as_index=False).agg(xsum=("좌표X","sum"),ysum=("좌표Y","sum"),n=("좌표X","size")))
    p=pd.concat(parts).groupby(["시도","시군구"],as_index=False).sum(); p["x"]=p.xsum/p.n; p["y"]=p.ysum/p.n
    return p.rename(columns={"시도":"SIDO_NM","시군구":"CCG_NM"})


def moran_knn(metrics,centroids,columns,permutations=499):
    d=metrics.merge(centroids,on=RKEY).drop_duplicates(RKEY).reset_index(drop=True); xy=d[["x","y"]].to_numpy(); dist=cdist(xy,xy); np.fill_diagonal(dist,np.inf)
    rng=np.random.default_rng(42); rows=[]
    for k in [4,6,8]:
        W=np.zeros((len(d),len(d))); nei=np.argpartition(dist,k,axis=1)[:,:k]
        for i,js in enumerate(nei): W[i,js]=1
        W=np.maximum(W,W.T); S0=W.sum()
        for col in columns:
            ok=d[col].notna().to_numpy(); sub=W[np.ix_(ok,ok)]; y=d.loc[ok,col].to_numpy(); n=len(y); z=y-y.mean(); den=(z*z).sum(); obs=n/sub.sum()*(sub*np.outer(z,z)).sum()/den
            sims=[]
            for _ in range(permutations):
                zp=rng.permutation(z); sims.append(n/sub.sum()*(sub*np.outer(zp,zp)).sum()/den)
            p=(1+sum(abs(v)>=abs(obs) for v in sims))/(permutations+1)
            rows.append({"metric":col,"k":k,"n":n,"moran_i":obs,"permutation_p":p,"null_mean":np.mean(sims)})
    return fdr_table(pd.DataFrame(rows),"permutation_p"),d


def alternative_clustering(features):
    cols=["real_amt_daily_growth_pct_m","cnt_daily_growth_pct_m","ticket_growth_pct_m","amt_pc_growth_pct_m","cnt_pc_growth_pct_m","commercial_aging_gap_pp_m","bc_age_hhi_slope","hhi_slope","foreign_share_slope","corp_share_slope","store_growth","perstore_growth","closure_rate","cnt_js_first3_last3","cnt_weakened_fraction"]
    cols=[c for c in cols if c in features]
    X=features[cols].replace([np.inf,-np.inf],np.nan); X=SimpleImputer(strategy="median").fit_transform(X); X=StandardScaler().fit_transform(np.clip(X,np.quantile(X,.01,axis=0),np.quantile(X,.99,axis=0)))
    rows=[]; models={}; rng=np.random.default_rng(42)
    for k in range(2,7):
        ag=AgglomerativeClustering(n_clusters=k,linkage="ward"); la=ag.fit_predict(X); rows.append({"method":"hierarchical_ward","k":k,"silhouette":silhouette_score(X,la),"bic":np.nan,"seed_ari_mean":1.0}); models[("hierarchical_ward",k)]=la
        labs=[]; bics=[]
        for seed in range(10):
            gm=GaussianMixture(n_components=k,covariance_type="diag",reg_covar=1e-4,n_init=5,random_state=seed).fit(X); labs.append(gm.predict(X)); bics.append(gm.bic(X))
        aris=[adjusted_rand_score(labs[0],l) for l in labs[1:]]
        rows.append({"method":"gmm_diag","k":k,"silhouette":silhouette_score(X,labs[0]),"bic":np.mean(bics),"seed_ari_mean":np.mean(aris)}); models[("gmm_diag",k)]=labs[0]
    sel=pd.DataFrame(rows)
    # 결정적 계층군집을 단순히 '안정적'이라 보지 않고, 표준화 feature에
    # 작은 교란(0.05 SD)을 주었을 때의 ARI로 경계 강건성을 따로 평가한다.
    perturb=[]
    for _,r in sel.iterrows():
        method,k=r.method,int(r.k); base_labels=models[(method,k)]; aris=[]
        for seed in range(20):
            Xp=X+rng.normal(0,.05,size=X.shape)
            if method=="hierarchical_ward": lab=AgglomerativeClustering(n_clusters=k,linkage="ward").fit_predict(Xp)
            else: lab=GaussianMixture(n_components=k,covariance_type="diag",reg_covar=1e-4,n_init=5,random_state=seed).fit_predict(Xp)
            aris.append(adjusted_rand_score(base_labels,lab))
        perturb.append(np.mean(aris))
    sel["perturbation_ari_mean"]=perturb
    eligible=sel[(sel.seed_ari_mean>=.75)&(sel.perturbation_ari_mean>=.75)&(sel.silhouette>0)]
    best=eligible.sort_values(["silhouette","perturbation_ari_mean","seed_ari_mean"],ascending=False).iloc[0]
    labels=models[(best.method,int(best.k))]; out=features.copy(); out["cluster_method"]=best.method; out["cluster_k"]=int(best.k); out["cluster"]=labels+1
    profile=out.groupby("cluster")[cols].median().reset_index(); sizes=out.groupby("cluster").size().rename("n").reset_index(); profile=sizes.merge(profile,on="cluster")
    return out,sel,profile,cols


def plots(real,stores,aging,base,moran,clustered):
    configure_plot(); FIG.mkdir(parents=True,exist_ok=True)
    specs=[
      (real,"amt_growth_pct_m","real_amt_growth_pct_m","01_nominal_real_industry.png","명목 AMT 성장","실질 AMT 성장"),
      (stores,"가동_당월말_growth_pct_m","cnt_growth_pct_m","02_demand_supply.png","점포수 성장","CNT 성장"),
      (aging,"pop_60plus_share_slope","bc_60plus_share_slope","03_aging_gap.png","주민 60+ 점유율 slope","소비 60+ 점유율 slope"),
      (base.assign(log_initial=np.log10(base.first3_amt_mean)),"log_initial","amt_growth_pct_m","04_base_effect.png","초기 AMT(log10)","AMT 성장")]
    for d,x,y,name,xlab,ylab in specs:
        fig,ax=plt.subplots(figsize=(8,6)); ax.scatter(d[x],d[y],s=14,alpha=.55); ax.axhline(0,c="grey",lw=.8); ax.axvline(0,c="grey",lw=.8); ax.set(xlabel=xlab,ylabel=ylab); fig.tight_layout(); fig.savefig(FIG/name,dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5)); q=moran[moran.k.eq(6)].sort_values("moran_i"); ax.barh(q.metric,q.moran_i,color=np.where(q.p_fdr<.05,"#E45756","#9ecae9")); ax.set(xlabel="Moran's I",title="거리기반 6-nearest 공간 자기상관"); fig.tight_layout(); fig.savefig(FIG/"05_moran_i.png",dpi=180); plt.close(fig)
    num=[c for c in clustered if c.endswith("growth_pct_m")][:2]
    if len(num)>=2:
        fig,ax=plt.subplots(figsize=(8,6));
        for c,g in clustered.groupby("cluster"): ax.scatter(g[num[0]],g[num[1]],s=20,alpha=.65,label=f"C{c}")
        ax.set(xlabel=num[0],ylabel=num[1]); ax.legend(); fig.tight_layout(); fig.savefig(FIG/"06_alternative_clusters.png",dpi=180); plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    bc=load_bc(); pop=parse_population(bc[RKEY].drop_duplicates()); region_month=bc.groupby(RKEY+["STRD_YYMM"],as_index=False)[["amt","cnt"]].sum(); region_month["avg_ticket"]=region_month.amt/region_month.cnt
    real,real_month=h1_real_industry(bc)
    popm,pop_month=h4_h6_population(region_month,bc,pop)
    # 2026년 하반기 화성시 분구 4개는 1~6월 공식 인구가 비어 있으므로
    # 임의 배분하지 않고 1인당 판정을 결측으로 둔다.
    invalid_population=popm.population_growth_pct_m.isna()
    popm.loc[invalid_population,["amt_total_pc_quadrant","cnt_total_pc_quadrant"]]=np.nan
    mix7=h2_ticket_mix(bc,CORE7,"7개전국공통"); mix11=h2_ticket_mix(bc,ALL11,"11개완전"); mix=pd.concat([mix7,mix11])
    segments,segment_month=h3_segments(bc); contrib,contrib_month=h4_growth_contribution(bc)
    age_metrics,age_month,age_share=age_concentration(bc); age_vol,age_vol_tests=h5_concentration_volatility(region_month,age_month); age_ind,age_ind_tests=h7_age_industry_volatility(bc)
    stores,store_month=localdata_integrate(bc,pop)
    div7,divm7,shift7=h12_h13_portfolio(bc,CORE7,"7개전국공통"); div11,divm11,shift11=h12_h13_portfolio(bc,ALL11,"11개완전"); diversity=pd.concat([div7,div11]); div_month=pd.concat([divm7,divm11]); shift=pd.concat([shift7,shift11])
    broad,broad_detail=breadth(bc); base,_=h10_base_effect(region_month); base,base_tests=h14_admin_base_effect(base)
    # H10/H18 단계별 모형: LOCALDATA 실제 폐업률·순점포변화.
    foreign=segments[segments.segment.eq("외국인")][RKEY+["amt_share_slope_pp_m"]].rename(columns={"amt_share_slope_pp_m":"foreign_share_slope"}); corp=segments[segments.segment.eq("법인")][RKEY+["amt_share_slope_pp_m"]].rename(columns={"amt_share_slope_pp_m":"corp_share_slope"})
    divreg=diversity[(diversity.coverage.eq("7개전국공통"))&(diversity.metric.eq("cnt"))][RKEY+["hhi_slope","entropy_slope"]].rename(columns={"hhi_slope":"industry_hhi_slope"})
    shiftreg=shift[shift.coverage.eq("7개전국공통")][RKEY+["cnt_js_first3_last3"]].rename(columns={"cnt_js_first3_last3":"portfolio_js"})
    context=popm[RKEY+["population_growth_pct_m","commercial_aging_gap_pp_m","bc_age_hhi_slope"]].merge(foreign,on=RKEY).merge(corp,on=RKEY).merge(divreg,on=RKEY,how="left").merge(shiftreg,on=RKEY,how="left").merge(broad[RKEY+["cnt_weakened_fraction"]],on=RKEY)
    model_data=stores.merge(context,on=RKEY,how="left")
    store_sens=store_sensitivity(stores)
    groups=model_data.SIDO_NM+"|"+model_data.CCG_NM; model_frames=[]; imp_frames=[]
    for outcome in ["closure_rate_6m_pct","net_store_change_pct"]:
        a,b=model_comparison(model_data,outcome,groups); model_frames.append(a); imp_frames.append(b)
    models=pd.concat(model_frames); importances=pd.concat(imp_frames)
    residual,resid_imp,resid_models,resid_quality=residual_growth(base,popm,segments,stores,diversity,shift,broad)
    structure_data,structure_tests,portfolio_compare=structure_associations(diversity,shift,broad,popm,age_vol,stores)
    # 공간·군집용 지역 feature.
    real_region=real.groupby(RKEY,as_index=False).agg(real_amt_daily_growth_pct_m=("real_amt_daily_growth_pct_m","median"),cnt_daily_growth_pct_m=("cnt_daily_growth_pct_m","median"),ticket_growth_pct_m=("ticket_growth_pct_m","median"))
    store_region=stores.groupby(RKEY,as_index=False).agg(store_growth=("가동_당월말_growth_pct_m","median"),perstore_growth=("cnt_per_store_growth_pct_m","median"),closure_rate=("closure_rate_6m_pct","median"),demand_supply_gap=("demand_supply_gap_cnt","median"))
    divreg_feat=divreg.rename(columns={"industry_hhi_slope":"hhi_slope"}); shiftreg_feat=shiftreg.rename(columns={"portfolio_js":"cnt_js_first3_last3"})
    feat=real_region.merge(popm,on=RKEY).merge(store_region,on=RKEY,how="left").merge(divreg_feat,on=RKEY,how="left").merge(shiftreg_feat,on=RKEY,how="left").merge(foreign,on=RKEY).merge(corp,on=RKEY).merge(broad,on=RKEY)
    cent=region_centroids(); moran,moran_data=moran_knn(feat,cent,["real_amt_daily_growth_pct_m","cnt_daily_growth_pct_m","amt_pc_growth_pct_m","commercial_aging_gap_pp_m","hhi_slope","cnt_js_first3_last3","demand_supply_gap"])
    clustered,cluster_selection,cluster_profile,cluster_features=alternative_clustering(feat)
    tables={
      "01_h1_real_industry.csv":real,"02_h1_real_monthly.csv":real_month,"03_h2_percapita_aging.csv":popm,"04_h3_ticket_decomposition.csv":mix,
      "05_h4_segment_contribution.csv":contrib,"06_h4_segment_monthly.csv":contrib_month,"07_h4_segment_region.csv":segments,
      "08_h6_age_metrics.csv":age_metrics,"09_h7_age_volatility_region.csv":age_vol,"10_h7_age_volatility_tests.csv":age_vol_tests,
      "11_h7_age_industry_metrics.csv":age_ind,"12_h7_age_industry_tests.csv":age_ind_tests,"13_h8_h9_store_metrics.csv":stores,
      "14_h8_h9_store_monthly.csv":store_month,"15_h10_h18_model_comparison.csv":models,"16_h10_h18_importance.csv":importances,
      "17_h11_diversity.csv":diversity,"18_h11_diversity_monthly.csv":div_month,"19_h12_portfolio_shift.csv":shift,
      "20_h13_breadth.csv":broad,"21_h13_breadth_detail.csv":broad_detail,"22_h14_base_effect.csv":base,"23_h14_base_tests.csv":base_tests,
      "24_h15_residual_growth.csv":residual,"25_h15_importance.csv":resid_imp,"26_h16_moran.csv":moran,"27_h16_centroids.csv":cent,
      "28_h17_clustered.csv":clustered,"29_h17_cluster_selection.csv":cluster_selection,"30_h17_cluster_profile.csv":cluster_profile,
      "31_bc_localdata_mapping.csv":pd.read_csv(LOCAL_MAP),"32_bc_cpi_mapping.csv":pd.read_csv(CPI_MAP),"33_h10_h18_model_dataset.csv":model_data,
      "34_h15_residual_model_comparison.csv":resid_models,"35_h11_h13_structure_dataset.csv":structure_data,"36_h11_h13_association_tests.csv":structure_tests,"37_h12_top10pct_comparison.csv":portfolio_compare}
    tables["38_h8_h9_volume_sensitivity.csv"]=store_sens
    # 보고서용 판정·점수표가 존재하면 통합 Excel에도 함께 싣는다.
    for extra in ["39_hypothesis_scorecard.csv","40_top5_scoring.csv"]:
        path=OUT/extra
        if path.exists(): tables[extra]=pd.read_csv(path)
    for n,d in tables.items(): d.to_csv(OUT/n,index=False,encoding="utf-8-sig")
    with pd.ExcelWriter(OUT/"deep_exploration_tables.xlsx",engine="openpyxl") as w:
        for n,d in tables.items(): d.to_excel(w,sheet_name=n[:29],index=False)
    plots(real,stores,popm,base,moran,clustered)
    summary={
      "regions":255,"H1_types":real.nominal_real_type.value_counts().to_dict(),"H1_nominal_plus_real_minus":int((real.nominal_real_type=="명목+/실질-").sum()),
      "H2_valid_regions":int((~invalid_population).sum()),"H2_excluded_regions":popm.loc[invalid_population,RKEY].to_dict("records"),"H2_amt_quadrants":popm.amt_total_pc_quadrant.value_counts().to_dict(),"H3_7industry":mix7[["within_industry_effect","between_industry_mix_effect","within_abs_share"]].median().to_dict(),
      "H4_max_identity_error":float(max(contrib.amt_identity_error.abs().max(),contrib.cnt_identity_error.abs().max())),
      "H6_positive_aging_gap":int((popm.commercial_aging_gap_pp_m>0).sum()),"H7_region_tests":age_vol_tests.to_dict("records"),
      "H8_market_store_types":stores.groupby(["confidence","market_store_type"]).size().rename("n").reset_index().to_dict("records"),"H9_demand_supply":stores.groupby(["confidence","demand_supply_type"]).size().rename("n").reset_index().to_dict("records"),
      "H8_robust_market_store_types":stores.groupby(["confidence","market_store_type_robust"]).size().rename("n").reset_index().to_dict("records"),"H9_robust_demand_supply":stores.groupby(["confidence","demand_supply_type_robust"]).size().rename("n").reset_index().to_dict("records"),
      "H10_H18_models":models.to_dict("records"),"H11_concentration_up":int(((div7.metric=="cnt")&(div7.hhi_slope>0)&(div7.entropy_slope<0)).sum()),
      "H12_js_median":float(shift7.cnt_js_first3_last3.median()),"H13_breadth_median":float(broad.cnt_weakened_fraction.median()),
      "H15_quality":resid_quality,"H16_significant":moran[moran.p_fdr<.05].to_dict("records"),
      "H17_selected":cluster_selection.sort_values(["silhouette"],ascending=False).iloc[0].to_dict(),"H17_features":cluster_features,
      "limitations":{"H5":"관광 데이터랩 원자료 미확보로 국내 관광객/거주민 분리 불가","H16":"공식 경계 인접성이 아닌 공식 시설좌표 기반 KNN","age":"BC 1·2 연령 라벨 중복","localdata":"2026 하반기 개편주소를 상반기 경계로 역매핑"}}
    (OUT/"analysis_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2,default=str))

if __name__=="__main__": main()
