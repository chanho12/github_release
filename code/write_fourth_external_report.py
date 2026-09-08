#!/usr/bin/env python3
"""Build the fourth-exploration scorecards and Markdown report from result tables."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/fourth_external_integration"


def fmt_table(df, digits=3):
    q = df.copy()
    for c in q.select_dtypes(include="number"):
        q[c] = q[c].map(lambda x: f"{x:.{digits}f}" if pd.notna(x) else "—")
    q = q.fillna("—").astype(str).map(lambda x: x.replace("|", "\\|"))
    header = "| " + " | ".join(q.columns.astype(str)) + " |"
    rule = "|" + "|".join(["---"] * len(q.columns)) + "|"
    rows = ["| " + " | ".join(row) + " |" for row in q.to_numpy().tolist()]
    return "\n".join([header, rule, *rows])


def main():
    h1 = pd.read_csv(OUT / "05_h1_market_opportunity_summary.csv")
    h1_main = h1[(h1.confidence == "HIGH") & (h1.cnt_threshold == 0) & (~h1.robust_only)].copy()
    h1_rob = h1[(h1.confidence == "HIGH") & (h1.cnt_threshold == 0) & h1.robust_only].copy()
    h1_vol = h1[(h1.confidence == "HIGH") & (~h1.robust_only) & h1.h1_type.str.startswith(("B", "C"))]
    h2 = pd.read_csv(OUT / "07_h2_gap_group_summary.csv")
    h2test = pd.read_csv(OUT / "08_h2_gap_extreme_tests.csv")
    visitor = pd.read_csv(OUT / "03_visitor_demand_region.csv")
    vassoc = pd.read_csv(OUT / "12_h4_visitor_association_tests.csv")
    h5 = pd.read_csv(OUT / "13_h5_visitor_growth_comparison.csv")
    reb = pd.read_csv(OUT / "14_h6_h7_reb_province.csv")
    rebtest = pd.read_csv(OUT / "15_h6_h7_reb_tests.csv")
    h8 = pd.read_csv(OUT / "16_h8_similar_amt_growth_regions.csv")
    load = pd.read_csv(OUT / "17_h9_pca_loadings.csv")
    clusters = pd.read_csv(OUT / "20_h9_cluster_profile.csv")
    neg = pd.read_csv(OUT / "21_negative_tests.csv")
    avail = pd.read_csv(OUT / "00_external_data_availability.csv")
    mapping = pd.read_csv(OUT / "01_bc_store_mapping.csv")

    scorecard = pd.DataFrame([
        ["H1", "총시장 성장과 점포당 기회 차이", "B 31/955(3.25%), C 176/955(18.43%), E 272/955(28.48%)", "B는 강건 표본 8/490; C·E는 거래량 필터에도 잔존", "MODERATE"],
        ["H2", "수요-공급 속도차", "하위10% CNT Gap -1.38%p/월, 상위10% +9.62%p/월", "하위군 신규율 5.42%, 폐업률 4.43%; 극단군 차이 FDR<.05", "MODERATE"],
        ["H3", "동일업종 경쟁압력 차이", "업종별 Store growth-CNT growth percentile 제공", "HIGH 4개 연결업종 중심; 점포밀도는 LOCALDATA만", "MODERATE"],
        ["H4", "주민수요형과 방문수요형 구분", "체류 상대성장과 BC 상대 AMT/CNT ρ=.688/.630", "89개·1~3월; k=2 silhouette=.385", "MODERATE"],
        ["H5", "체류수요 성장군의 소비구조 차이", "상위25% AMT/CNT 효과 .444/.418", "FDR=.055로 경계적; 고령·HHI·JS 차이 없음", "MODERATE"],
        ["H6", "소비와 임대료 방향 괴리", "소비+·임대료- 40/51, 소비-·임대료+ 1/51", "17개 시도×3 유형; 시군구 직접 검증 아님", "WEAK"],
        ["H7", "소비성장과 공실 증가 공존", "중대형 11/17, 소규모 9/17, 집합 8/17", "36개 상관검정 모두 FDR>.05", "WEAK"],
        ["H8", "같은 AMT 성장의 환경 다양성", "6개월 +5~10% 지역 48개; 고령화 Gap -0.274~+0.370", "수요-공급 Gap은 모두 양수라 해당 축 차이는 제한", "MODERATE"],
        ["H9", "성장의 질 잠재차원", "3PC 71.25%; Ward k=2 silhouette=.562", "교란 ARI 평균 .693, 최저 .440", "MODERATE"],
        ["N1", "AMT와 점포당 AMT 방향은 거의 같다", "HIGH 94.03%, HIGH+MEDIUM 95.62% 동일", "점포당 기회 단독 주제를 약화", "SUPPORTED"],
        ["N2", "체류자료가 소비성장 설명을 더하지 못한다", "상대 AMT/CNT CV R² 개선 +.645/+.662", "3개월 공통계절성 위험은 남음", "REJECTED"],
        ["N3", "부동산과 BC 구조는 거의 무관하다", "최대 |ρ|=.701, FDR=.062; 나머지는 약함", "전국 주근거가 아닌 보조 사례로 제한", "PARTIAL"],
        ["N4", "소비고령화 Gap은 통제 후 사라진다", "89개 조정평균 +.572%p/월, p<1e-40", "인구감소 여부는 표본 내 고정이라 완전 통제 불가", "REJECTED"],
    ], columns=["가설", "질문", "핵심 결과", "강건성·한계", "판정"])
    scorecard.to_csv(OUT / "23_hypothesis_scorecard.csv", index=False, encoding="utf-8-sig")

    top5 = pd.DataFrame([
        [1, "체류인구 변화와 BC 상대소비 변화", "89개", "ρ=.688(AMT), .630(CNT)", "주민등록인구만으로 설명되지 않는 비거주 수요 맥락"],
        [2, "인구보다 빠른 소비고령화의 잔존", "89개 인구감소지역", "조정평균 +.572%p/월", "체류·군·인구변화를 넣어도 평균 Gap이 남음"],
        [3, "수요 감소·점포 증가", "176/955", "18.43%; 강건 86/490", "총시장과 공급이 반대 방향인 경쟁환경 후보"],
        [4, "시장 성장·점포 감소", "272/955", "28.48%; 강건 176/490", "남은 점포로 소비가 집중될 수 있는 공급축소형"],
        [5, "소비성장과 부동산 방향 괴리", "시도×상가유형 51개", "소비+·임대료- 40개; 소비+·공실+ 28개", "매출과 비용·공간환경은 같은 방향이 아님"],
    ], columns=["순위", "괴리", "규모", "효과", "외부데이터가 추가한 의미"])
    top5.to_csv(OUT / "24_divergence_top5.csv", index=False, encoding="utf-8-sig")

    topics = pd.DataFrame([
        [1, "거주인구만으로 보이지 않는 지역 소비기반 전환", "주민·체류 수요와 소비고령화를 분리해 같은 소비성장의 기반 차이를 설명", "체류-BC 상대성장 ρ=.688/.630; 조정 소비고령화 Gap +.572", "BC AMT·CNT·연령·업종구조가 결과변수", "생활인구·주민등록인구가 수요 기반을 구분", "HIGH"],
        [2, "총시장과 점포기회의 수요-공급 속도차", "소비수요와 점포공급 변화속도가 다른 지역·업종을 비교", "C형 18.43%, 하위 Gap군 신규·폐업률 상승", "BC 수요를 점포수 분모에 결합", "LOCALDATA·국세청·향후 SEMAS 점포", "MEDIUM"],
        [3, "같은 매출성장 속 객단가·고객·포트폴리오의 다른 경로", "AMT 하나가 숨기는 CNT·객단가·고령집중·업종이동을 분해", "3개 잠재차원 71.25%; 2개 국면", "BC 세부 차원이 핵심", "인구·체류·점포는 해석과 검증", "MEDIUM-HIGH"],
    ], columns=["순위", "주제", "Problem", "Evidence", "BC가 필요한 이유", "외부데이터 역할", "추천도"])
    topics.to_csv(OUT / "25_topic_candidates.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUT / "fourth_external_integration_tables.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        scorecard.to_excel(writer, sheet_name="23_scorecard", index=False)
        top5.to_excel(writer, sheet_name="24_divergence_top5", index=False)
        topics.to_excel(writer, sheet_name="25_topic_candidates", index=False)

    h1_show = h1_main[["h1_type", "n", "denominator", "share_pct", "region_n", "industry_n"]]
    h1r_show = h1_rob[["h1_type", "n", "denominator", "share_pct"]]
    vol = h1_vol[["cnt_threshold", "h1_type", "n", "denominator", "share_pct"]]
    h2_show = h2[["gap_group", "demand_supply_gap_cnt_median", "amt_per_store_growth_pct_m_median",
                  "cnt_per_store_growth_pct_m_median", "opening_rate_6m_pct_median",
                  "closure_rate_6m_pct_median", "net_store_change_pct_median"]]
    va = vassoc.head(4)[["visitor_metric", "bc_metric", "n", "spearman_rho", "p_fdr"]]
    h5show = h5.head(6)[["metric", "high_median", "low_median", "rank_biserial", "p_fdr"]]
    h8range = h8[["regional_demand_supply_gap", "commercial_aging_gap_pp_m", "cnt_js_first3_last3",
                  "regional_perstore_amt_growth", "amt_residual_sd"]].agg(["min", "median", "max"]).T.reset_index(names="metric")
    loading_show = load.set_index("feature").abs().apply(lambda s: s.nlargest(5).index.tolist()).to_dict()

    report = f"""# BC카드 소비데이터 4차 외부데이터 결합 심층분석

## 1. 결론과 주제 재선정

이번 분석의 최종 결론은 **‘점포당 소비기회’를 단독 주제로 정하는 것보다, 지역 소비성장의 수요 기반을 주민·체류·소비연령 구조로 분해하는 주제가 데이터에 더 잘 맞는다**는 것이다.

> **최우선 주제: 거주인구만으로 보이지 않는 지역 소비기반 전환 — 주민수요, 체류수요, 소비고령화를 결합해 같은 매출성장의 서로 다른 기반을 설명한다.**

가장 중요한 새 증거는 인구감소지역 89곳에서 2026년 1~3월 체류인구의 전국 대비 상대성장과 BC 상대 AMT·CNT 성장의 Spearman 상관이 각각 **0.688, 0.630**이었다는 점이다. 주민등록인구만 쓴 지역 그룹 교차검증은 상대소비 성장 설명에 실패했지만, 체류인구 변화와 체류/주민 비율을 넣으면 AMT와 CNT의 검증 `R²`가 각각 **0.645, 0.662만큼 개선**됐다.

동시에 소비고령화 Gap은 체류변화·체류의존도·군 여부·주민인구 변화와 규모를 통제한 89개 표본에서도 평균 **+0.572%p/월**로 남았다. 이는 방문수요와 고령 소비층 집중이 같은 현상의 대체설명이 아니라, 지역 소비기반을 구분하는 서로 다른 축일 가능성을 보여준다.

반면 점포당 기회 가설은 중요한 반증을 받았다. HIGH 매핑 955개 유효 지역×연결업종에서 AMT 성장과 점포당 AMT 성장의 부호는 **94.03%**가 같았다. 시장은 성장하지만 공급이 더 빨라 점포당 AMT가 줄어드는 B형은 **31개(3.25%)**, 세 추세방법이 모두 일치한 사례는 **8개(1.63%)**였다. 따라서 이 현상은 유용한 보조 진단이지만 전국을 관통하는 대표 주제로는 약하다.

## 2. 외부데이터 통합 결과

{fmt_table(avail, 0)}

공식 출처는 [행정안전부 2026년 1분기 생활인구](https://mois.go.kr/frt/bbs/type010/commonSelectBoardArticle.do?bbsId=BBSMSTR_000000000008&nttId=128294), [소상공인시장진흥공단 상가정보 API](https://www.data.go.kr/data/15012005/openapi.do), [한국관광 데이터랩](https://datalab.visitkorea.or.kr/datalab/portal/main/getMainForm.do), [한국부동산원 상업용부동산 임대동향조사](https://www.reb.or.kr/reb/cm/cntnts/cntntsView.do?cntntsId=1049&mi=10335&statId=S237220284)다.

소상공인 상가정보 API는 무료·자동승인이지만 서비스키 없이 호출되지 않았고, 현재 스냅샷만으로는 1~6월 점포 slope를 만들 수 없다. 따라서 11개 업종 매핑 설계는 보존하되 이번 계량분석은 LOCALDATA와 국세청 월간 가동사업자로 실행했다. 한국관광 데이터랩 값은 로그인·사전설문 제한 때문에 추정하지 않았으며, 대신 다운로드 가능한 행정안전부 체류인구를 **관광객이 아닌 비거주 방문수요 대리 맥락**으로 사용했다.

## 3. BC × 외부업종 매핑

11개 BC 업종의 SEMAS 연결 설계는 [01_bc_store_mapping.csv](01_bc_store_mapping.csv)에 있다. HIGH는 명칭·범위가 직접 대응하는 경우, MEDIUM은 근접 범위, LOW는 다른 BC 업종과 중복될 가능성이 큰 경우다.

- HIGH 설계: 대형할인점, 편의점, 슈퍼마켓, 일반한식, 중국음식, 서양음식, 제과점
- MEDIUM 설계: 갈비전문점, 일식회집, 스넥
- LOW 설계: 한정식
- 실제 월별 주 분석 HIGH: 편의점·슈퍼마켓(국세청), 중국음식·제과점(LOCALDATA)
- 실제 HIGH+MEDIUM: LOCALDATA 한식통합·일식회집·서양음식·스넥 추가

API의 최신 247개 소분류 코드는 인증키로 코드목록을 조회한 뒤 확정해야 하므로 임의 코드를 기입하지 않았다. 이는 잘못된 코드로 매칭률을 부풀리는 것보다 안전하다.

## 4. H1 — 총시장 성장과 점포당 소비기회

{fmt_table(h1_show, 2)}

세 방법(로그 OLS, Theil–Sen, 전반3개월/후반3개월)이 모두 같은 부호인 490개만 보면 다음과 같다.

{fmt_table(h1r_show, 2)}

B형은 작고, C형과 E형이 더 크다. C형은 소비수요 감소와 공급 증가가 동시에 나타나는 후보이고, E형은 시장이 성장하면서 점포가 줄어 소비가 남은 점포에 집중될 수 있는 구조다. 어느 쪽도 폐업위험이나 기회 개선을 직접 뜻하지 않는다.

거래량 민감도에서 B·C는 다음처럼 변했다.

{fmt_table(vol, 2)}

CNT 50만 이상에서는 B가 0.77%로 더 작아졌지만 C는 11.20% 남았다. **H1은 부분적으로 성립하지만 중심은 B보다 C·E다. 판정은 MODERATE다.**

![시장 성장과 점포공급](figures/01_market_supply_types.png)

## 5. H2·H3 — 수요-공급 속도차와 동일업종 경쟁압력

HIGH 955개 조합의 `CNT growth - Store growth` 분포를 전국 percentile로 나눴다.

{fmt_table(h2_show, 2)}

하위 10%는 점포당 CNT가 월 -1.37%였고, 상위 10%는 +9.68%였다. 이 차이는 지표의 산식상 상당 부분 기계적이므로 독립 검증으로 과대해석하지 않았다. 더 의미 있는 차이는 하위군에서 6개월 신규율 중앙값이 5.42%, 폐업률이 4.43%로 상위군의 0%보다 높았다는 점이다. 하위·상위 10%의 신규율 rank-biserial은 0.442(`p_FDR<0.001`), 폐업률은 0.276(`p_FDR=0.016`)이었다. 공급과 수요의 속도차가 실제 진입·이탈이 많은 시장구조와 함께 관찰됐다.

H3는 업종별로 `Store growth-CNT growth`, `Store growth-real AMT growth`, 점포 1만명당 밀도의 percentile과 업종 중앙값 대비 차이를 만들었다. 결과는 [09_h3_competition_pressure.csv](09_h3_competition_pressure.csv)에 있으며, 서로 다른 업종의 절대 점포수를 직접 비교하지 않았다. **H2·H3는 MODERATE**지만 실제 HIGH 범위가 4개 연결업종이라는 제약이 있다.

## 6. H4·H5 — 주민수요와 체류수요

{fmt_table(va, 3)}

체류의존도 수준 자체보다 **체류인구의 상대 변화**가 BC 상대소비 변화와 강하게 연결됐다. 이 관계는 3개월 slope 하나에만 의존하지 않았다. 1→2월 상대변화의 상관은 AMT 0.534·CNT 0.608, 2→3월은 AMT 0.529·CNT 0.453으로 두 월간 구간에서 모두 같은 방향이었다. 표준화 feature를 강제 라벨 없이 K-means로 나누면 `k=2`가 가장 높았지만 silhouette는 0.385로 중간 수준이었다.

- Cluster 1: 12개 지역, 체류인구 -17.97%/월, BC AMT -10.27%/월, CNT -6.10%/월
- Cluster 2: 77개 지역, 체류인구 +8.48%/월, BC AMT +3.02%/월, CNT +4.73%/월

이는 1~3월 체류수요의 회복·위축과 BC 소비가 함께 움직였다는 뜻이지 관광이 소비를 발생시켰다는 인과증거가 아니다. 두 지표 모두 겨울에서 봄으로 넘어가는 계절성을 공유할 수 있다.

체류 상대성장 상·하위 25% 비교는 다음과 같다.

{fmt_table(h5show, 3)}

AMT·CNT·점포당 AMT/CNT 효과는 중간 이상이었지만 FDR이 0.055로 5% 기준을 조금 넘었다. 연령 HHI, 소비고령화 Gap, 업종 HHI, 포트폴리오 JS, 변동성은 유의한 차이가 없었다. 즉 체류 증가는 **소비 성장 수준**과 연결됐지만 소비구조 전체를 설명하지는 못했다. **H4·H5는 MODERATE**다.

![체류인구와 BC 상대성장](figures/02_visitor_bc_growth.png)

## 7. H6·H7 — 임대료와 공실

한국부동산원 Q1·Q2를 17개 시도에서 중대형·소규모·집합상가로 나눴다. 시군구에 억지로 배분하지 않았기 때문에 51개는 독립된 시군구 표본이 아니라 `17개 시도 × 3개 상가유형`의 맥락 관측이다.

| 결과 | 중대형 | 소규모 | 집합 | 합계 |
|---|---:|---:|---:|---:|
| 소비+·임대료- | 14 | 12 | 14 | 40 |
| 소비·점포당소비·임대료+ | 2 | 4 | 2 | 8 |
| 소비-·임대료+ | 0 | 0 | 1 | 1 |
| 소비·임대료- | 1 | 1 | 0 | 2 |
| 소비+·공실률+ | 11 | 9 | 8 | 28 |

36개 상관검정 중 FDR 5%를 통과한 것은 없었다. 가장 큰 관계는 집합상가 임대료 변화와 점포당 AMT 성장의 `ρ=-0.701`, `p_FDR=0.062`로 경계적이었다. **H6·H7은 WEAK**이며 부동산 자료는 전국 핵심근거보다 시도·대표상권 사례 검증에 적합하다.

![소비와 임대료·공실](figures/03_reb_context.png)

## 8. H8 — 같은 AMT +5~10%의 다른 환경

월 slope를 1~6월 적합변화로 환산해 AMT가 +5~10%인 지역은 48개였다.

{fmt_table(h8range, 3)}

48개 모두 지역 중앙 수요-공급 Gap은 양수였기 때문에 이 좁은 성장구간에서 공급과속형이 크게 갈리지는 않았다. 반면 소비고령화 Gap은 -0.274~+0.370%p/월, 포트폴리오 JS는 0.005~0.038, AMT 잔차변동성은 0.039~0.167로 달랐다. 같은 AMT 성장률 안에서도 고객구조와 안정성은 분명 달랐다. **H8은 부분 채택, MODERATE**다.

![같은 성장의 다른 환경](figures/04_same_growth_different_environment.png)

## 9. H9 — 성장의 질을 구분한 잠재차원

전국 255개 지역에서 결측 coverage가 높은 17개 변수를 사용했다. 임대료·공실은 시도 단위이고 체류인구는 89개만 있어 전국 PCA에 강제로 반복·대치하지 않았다. 3개 주성분이 전체 분산의 71.25%를 설명했다.

- PC1: 성장·점포기회 축 — AMT, CNT, 1인당 소비, 점포당 소비, 수요-공급 Gap
- PC2: 소비층 집중·구조변화 축 — 소비고령화 Gap, 연령 HHI, 객단가, 포트폴리오 JS
- PC3: 인구·공급·업종집중 축 — 인구성장, 업종 HHI, 점포성장, 객단가 mix

절대 loading 상위 feature는 `{loading_show}`다.

Ward 군집은 `k=2`가 silhouette 0.562로 가장 높았다.

{fmt_table(clusters[["cluster", "n", "amt_growth_pct_m", "cnt_growth_pct_m", "ticket_growth_pct_m", "commercial_aging_gap_pp_m", "regional_perstore_amt_growth", "cnt_js_first3_last3", "amt_residual_sd"]], 3)}

38개 고성장 국면은 AMT·CNT·점포당 기회와 소비고령화·변동성이 함께 높았고, 217개 일반 국면은 CNT 중심 성장과 객단가 압축이 두드러졌다. 0.05 SD 교란의 군집 ARI는 평균 0.693, 최저 0.440으로 완전히 고정된 자연군집은 아니다. **H9는 MODERATE**다.

![통합 PCA 군집](figures/05_integrated_pca_clusters.png)

## 10. 반증 분석 N1~N4

{fmt_table(scorecard[scorecard.가설.str.startswith("N")], 3)}

N1이 지지됐다는 점이 중요하다. 점포당 AMT는 대부분 총 AMT와 같은 결론을 주므로 공모전 전체를 ‘숨은 점포당 위기’로 포장하면 데이터보다 주장이 앞선다. 반대로 N2와 N4는 기각되어 체류수요와 소비고령화가 최종 주제에서 남을 근거가 생겼다. N3는 시도 17개·분기 2개라 부동산을 중심축으로 올릴 만큼 강하지 않다.

## 11. 새롭게 발견된 괴리 TOP 5

{fmt_table(top5, 3)}

TOP 5는 위험순위가 아니다. 특히 시장성장·점포감소형은 경쟁 완화일 수도 있고, 점포 접근성 악화나 특정 사업체 집중일 수도 있다. 점포 단위 매출·생존자료 없이는 방향을 확정하지 않는다.

## 12. 공모전 주제 후보 3개와 최종 선택

{fmt_table(topics, 3)}

### 최종 선택: 거주인구만으로 보이지 않는 지역 소비기반 전환

**Problem.** 동일한 지역 AMT 성장이라도 주민수요가 받치는 성장, 체류수요가 받치는 성장, 특정 고령 소비층에 집중된 성장은 지속성과 소상공인 수요 기반이 다를 수 있다. 기존 상권분석은 총매출·유동인구·주민인구를 따로 보여주어 이 차이를 한 소비행동 체계 안에서 설명하기 어렵다.

**BC가 필요한 이유.** 체류인구만으로는 실제 결제금액·빈도·객단가·연령집중·업종 포트폴리오를 알 수 없다. 주민등록인구만으로도 비거주자가 만든 실제 소비 변화를 알 수 없다. BC AMT/CNT/연령/업종이 핵심 결과를 만들고 외부 인구·체류자료가 수요 기반을 분리한다.

**Novelty.** ‘관광 활성화’나 ‘고령상권’ 하나를 주장하는 대신 `소비성장 × 체류수요 × 소비고령화 × 포트폴리오 안정성`이 서로 독립적인지를 먼저 검증한다. 이번 결과에서 체류성장은 AMT·CNT와 강하게 연결됐지만 연령집중·업종구조를 설명하지 못했고, 소비고령화 Gap도 통제 후 남았다. 이 비동일성이 주제의 핵심이다.

**공모전 적합성.** BC 데이터 없이는 문제를 정의할 수 없고, 특정 업종·지역·폐업예측에 묶이지 않는다. 단, 본선 분석에서는 관광데이터랩 또는 전국 생활인구의 1~6월 자료를 합법적으로 확보해 89개·3개월이라는 현재 한계를 넓혀야 한다.

두 번째 후보인 수요-공급 속도차는 C형 발견에는 강하지만 N1 때문에 단독 대표성이 낮다. 세 번째 후보는 BC 데이터 적합성이 가장 높지만 기존 3차 분석의 확장에 가까워 외부데이터 결합의 새로움은 첫 번째보다 약하다.

## 13. 해석 한계

1. 2026년 6개월은 구조변화와 계절성을 분리하기 짧고, 체류인구는 1~3월뿐이다.
2. 행정안전부 체류인구는 관광객만이 아니라 통근·통학·업무·방문을 포함하는 더 넓은 개념이다.
3. 체류인구와 BC 소비의 높은 Q1 동행은 공통 계절성의 영향을 받을 수 있다. 12개월·전년동월 자료가 필요하다.
4. SEMAS는 인증키와 과거 스냅샷이 없어 매핑 설계만 했고 실제 slope는 LOCALDATA·국세청으로 계산했다.
5. HIGH 계량 매핑은 편의점·슈퍼마켓·중국음식·제과점 4개 연결업종이다. 11개 BC 업종 전체로 일반화할 수 없다.
6. 국세청 가동사업자 수는 물리적 점포 수와 같지 않다. LOCALDATA와 함께 source_scope를 보존했다.
7. 임대료·공실은 시도·대표상권 조사라 시군구 BC 값에 직접 귀속할 수 없다.
8. Gap과 점포당 성장률은 같은 수요·공급 변수에서 계산되므로 둘의 높은 연관은 독립 검증이 아니다.
9. 어떤 결과도 폐업, 과잉경쟁, 관광의 인과효과를 확정하지 않는다.

## 14. 재현 산출물

- 외부자료 가용성: [00_external_data_availability.csv](00_external_data_availability.csv)
- 11개 업종 매핑: [01_bc_store_mapping.csv](01_bc_store_mapping.csv)
- 생활인구 월별·지역지표: [02_living_population_monthly.csv](02_living_population_monthly.csv), [03_visitor_demand_region.csv](03_visitor_demand_region.csv)
- 점포 결합 지표: [04_market_store_metrics_combined.csv](04_market_store_metrics_combined.csv)
- H1 요약: [05_h1_market_opportunity_summary.csv](05_h1_market_opportunity_summary.csv)
- H2 상세·요약·검정: [06_h2_gap_group_detail.csv](06_h2_gap_group_detail.csv), [07_h2_gap_group_summary.csv](07_h2_gap_group_summary.csv), [08_h2_gap_extreme_tests.csv](08_h2_gap_extreme_tests.csv)
- H3 동일업종 percentile: [09_h3_competition_pressure.csv](09_h3_competition_pressure.csv)
- H4·H5: [12_h4_visitor_association_tests.csv](12_h4_visitor_association_tests.csv), [13_h5_visitor_growth_comparison.csv](13_h5_visitor_growth_comparison.csv)
- H6·H7: [14_h6_h7_reb_province.csv](14_h6_h7_reb_province.csv), [15_h6_h7_reb_tests.csv](15_h6_h7_reb_tests.csv)
- H8 동일성장 표본: [16_h8_similar_amt_growth_regions.csv](16_h8_similar_amt_growth_regions.csv)
- H9 PCA·군집: [17_h9_pca_loadings.csv](17_h9_pca_loadings.csv), [18_h9_pca_scores_clusters.csv](18_h9_pca_scores_clusters.csv), [20_h9_cluster_profile.csv](20_h9_cluster_profile.csv)
- 가설·TOP5·주제: [23_hypothesis_scorecard.csv](23_hypothesis_scorecard.csv), [24_divergence_top5.csv](24_divergence_top5.csv), [25_topic_candidates.csv](25_topic_candidates.csv)
- 통합 Excel: [fourth_external_integration_tables.xlsx](fourth_external_integration_tables.xlsx)
- 분석 코드: [../../scripts/analyze_fourth_external_integration.py](../../scripts/analyze_fourth_external_integration.py)
- PDF 파서: [../../scripts/parse_mois_living_population.py](../../scripts/parse_mois_living_population.py)
- 부동산 파서: [../../scripts/parse_reb_commercial_rent.py](../../scripts/parse_reb_commercial_rent.py)
- 보고서 생성 코드: [../../scripts/write_fourth_external_report.py](../../scripts/write_fourth_external_report.py)

---

분석 기준일: 2026-09-04  
분석 성격: 공모전 주제 선정 전 탐색적 외부데이터 결합 분석. 인과관계·폐업위험·과잉경쟁 확정 판정이 아님.
"""
    (OUT / "fourth_external_integration_report.md").write_text(report, encoding="utf-8")
    print(f"wrote report and three final tables to {OUT}")


if __name__ == "__main__":
    main()
