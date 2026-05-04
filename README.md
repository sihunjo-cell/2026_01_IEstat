# Bitcoin Effect on Hardware Pricing

이 프로젝트는 암호화폐 시장 변동이 한국 GPU 가격에 영향을 주는지 검정하기 위해, 원시 제품 가격표를 바로 평균하지 않고 `data_preprocessing → EDA_process → Hypo_1 → Hypo_2` 순서로 분석하였다. 핵심은 제품 구성 변화와 크롤링 노이즈를 먼저 줄인 뒤, 가격 수준이 아니라 주간 로그수익률로 단기 관계를 검정하고, 가격 수준의 장기 관계는 별도 공적분 검정으로 확인하는 것이다.

---

## 0. Key Methodologies Reference

본 프로젝트에서 활용된 주요 통계 및 시계열 분석 방법론의 도입 목적과 해석 기준이다. 

* <strong>Chain Matched Jevons Index</strong>: 매주 관측되는 제품 구성이 달라져서 생기는 평균 가격의 왜곡을 막기 위해, <strong>전주와 이번 주에 모두 존재하는 동일 제품의 가격 변화율만 연결</strong>하여 지수화할 때 사용한다. 신제품 출시나 단종 노이즈가 제거된 순수한 시장 가격 변동 흐름으로 해석한다.
* <strong>로그 수익률 (Log Return)</strong>: 지속적으로 상승하거나 하락하는 추세(Level)를 가진 데이터를 회귀분석할 때 생기는 <strong>가짜 상관관계(Spurious Correlation)를 방지</strong>하기 위해 사용한다. 시계열 데이터의 안정성을 확보하며, 주간 단위의 실질적인 가격 변동률로 해석한다.
* <strong>HAC / Newey-West Standard Error</strong>: 시계열 데이터의 특성상 과거의 충격이 현재에 영향을 주거나 변동폭이 일정하지 않을 때 <strong>표준오차를 보정</strong>하기 위해 사용한다. 이를 통해 도출된 p-value가 통계적으로 신뢰할 수 있음을 보증한다.
* <strong>VAR Granger Causality Test</strong>: 한 변수의 과거 값이 다른 변수의 현재를 예측하는 데 통계적으로 기여하는지 확인할 때 쓰인다. 본 분석에서는 비트코인 변동이 GPU 가격 변동을 이끄는 <strong>선행 지표 역할을 하는지</strong> 판단하는 기준이 된다.
* <strong>Engle-Granger Cointegration & ECM</strong>: Cointegration(공적분)은 개별 데이터가 불안정하더라도 두 지표 사이에 <strong>장기적인 균형 궤적</strong>이 존재하는지 검증할 때 쓰인다. ECM(오차수정모형)은 그 균형이 단기적으로 깨졌을 때 <strong>다시 원래의 장기 균형으로 회귀하는 속도와 방향</strong>이 있는지를 해석할 때 사용한다.
* <strong>Welch ANOVA & 상호작용 회귀 (Interaction Regression)</strong>: Welch ANOVA는 그룹 간 데이터의 분산이 다를 때 평균 차이를 보다 안전하게 검정하기 위해 사용한다. 상호작용 회귀는 비트코인 수익률이라는 요인이 GPU의 성능군(High/Mid/Low)에 따라 <strong>서로 다른 민감도로 영향을 미치는지</strong>를 분리해서 해석할 때 활용한다.

## 1. Data Processing and Index Construction

원시 데이터에서 가격이 `0`인 값은 실제 거래가격이 아니라 출시 전, 단종, 품절, 수집 실패가 섞인 관측 불가능 상태로 보았기 때문에 가격 계산에서 제외하였다. 동일 제품이 여러 행에 중복된 경우에는 날짜별 `median` 가격으로 병합하였다. 평균 대신 median을 쓴 이유는 판매처별 극단값과 크롤링 오류에 덜 민감하기 때문이다. GPU 제품명에서는 chip, VRAM, 성능군 정보를 추출했고, Hypo 2의 본분석은 `chip_perf_group`, 보조분석은 VRAM proxy를 사용하였다.

대표가격은 단순 평균이 아니라 Chain Matched Jevons Index로 만들었다. 인접한 두 주에 모두 관측된 동일 제품만 매칭하고, 같은 제품의 로그 가격변화율을 평균하여 주간 대표 수익률로 사용하였다. 이후 이 수익률을 누적해 기준값 100의 가격지수를 만들었다. 이 방식은 “어떤 제품이 새로 들어왔는가”가 아니라 “같은 제품의 가격이 얼마나 변했는가”를 추적하므로, 제품 구성 변화로 인한 대표가격 왜곡을 줄인다.

```text
group_return(g,t)
= average log price change of products observed in both week t-1 and week t

chain_index(g,t)
= chain_index(g,t-1) × exp(group_return(g,t)), with base index = 100
```

---

## 2. EDA: Data Quality and Preliminary Signals

EDA의 목적은 지수가 실제 검정에 쓸 수 있을 만큼 안정적인지 확인하는 것이다. 전체 지수는 109개 주간 step 중 106개가 valid였고, return 기준으로는 105개 주가 분석 가능했다. median matched product 수는 GPU 711개, CPU 209개, RAM 1156개였다. chip 성능군 기준으로도 high, mid, low 모두 105개 valid return을 확보했지만, median matched product 수는 high 269개, mid 316개, low 43개로 low 그룹의 표본이 작았다. 따라서 low는 해석상 주의가 필요한 그룹으로 보았다.

기초 상관에서는 GPU와 BTC 수익률의 상관이 약 `-0.246`, CPU는 `0.094`, RAM은 `-0.086`이었다. 단순 BTC beta도 GPU `-0.0399`, CPU `0.0096`, RAM `-0.0374`로 나타났다. 즉 EDA 단계에서 이미 “BTC 상승 → GPU 가격 상승”이라는 단순 양의 관계는 보이지 않았다. 따라서 본검정에서는 단순 상관이 아니라 BTC 시차, 환율, NVDA, 자기시차를 통제한 회귀로 넘어갔다.

### EDA Plot Interpretation

| Plot | What was checked | Interpretation |
|---|---|---|
| `Chain_drift_check_*` | fixed basket index와 Chain Matched Jevons Index 비교 | fixed basket은 high/low에서 크게 벌어졌고, chain Jevons는 더 안정적이었다. 제품 구성 변화가 큰 데이터에서는 chain matched 방식이 더 적절하다는 근거다. |
| `H2_*_weekly_median_vs_last` | 주간 median 가격과 last 가격 기준 지수 비교 | high/mid/low 모두 두 지수가 비슷하게 움직였다. 주간 대표값으로 median을 사용한 것이 결론을 임의로 만든 것은 아니라고 볼 수 있다. |
| `H2_chip_groups_valid_adjacent_matched_products` | 성능군별 매칭 제품 수 | high/mid는 충분하지만 low는 훨씬 작다. 그래서 Hypo 2에서 low 기준 해석은 조심하고, mid 기준항 및 high-vs-low contrast를 함께 확인했다. |
| `Trim_sensitivity_*` | 5% trim index와 no-trim index 비교 | no-trim 지수는 2025년과 2026년 초에 훨씬 크게 튄다. main index는 크롤링 오류와 극단 충격을 줄이지만, 실제 충격까지 완화할 수 있어 no-trim robustness도 수행했다. |
| `Weekly_GPU_CPU_RAM_Overall_Chain_Matched_Jevons_Index` | GPU/CPU/RAM 전체 지수 비교 | RAM이 2025년 말~2026년 초 GPU/CPU보다 훨씬 크게 상승했다. 하드웨어 가격 충격이 GPU-BTC만의 현상이라고 보기 어렵다는 근거다. |
| `Monthly/Quarterly distribution` | 월별·분기별 수익률 분포 | 특정 월·분기에 outlier와 계절성이 존재한다. 제품 출시·시즌성 효과를 완전히 제거하지 못했다는 한계를 인정해야 한다. |

---

## 3. Controls and Domain Interpretation

본분석에서는 `fx_ret`, `nvda_ret`, `gpu_ret_l1`을 공통 통제변수로 사용하였다. `fx_ret`는 원화 GPU 가격에서 USD/KRW 환율 효과를 분리하기 위한 변수이고, `nvda_ret`는 AI 및 반도체 수요 proxy이다. `gpu_ret_l1`은 GPU 가격 변동 자체의 관성을 통제하기 위해 넣었다. Hypo 1에서는 NASDAQ과 SOX를 확장 통제 robustness로 추가했고, Hypo 2에서는 active product 수, new product share, bridge ratio를 제품 구성 변화 proxy로 사용하였다.

BTC는 2024년 이후 분석 기간에서 GPU 채굴 직접 수요라기보다 crypto sentiment 또는 위험자산 심리 proxy로 해석하였다. Ethereum이 2022년 Proof-of-Stake로 전환한 이후에는 주요 암호화폐 채굴 수요가 GPU 가격을 직접 밀어 올린다고 주장하기 어렵기 때문이다. 따라서 ETC-KRW는 mining 관련 보조 EDA로만 사용하고, 메인 회귀에는 넣지 않았다.

---

## 4. Hypothesis 1: BTC-GPU Co-movement

Hypo 1은 전체 GPU 가격과 BTC 가격 사이에 유의미한 단기 또는 장기 관계가 있는지 검정한다. 단기 모형에서는 전체 GPU 주간 로그수익률을 종속변수로 두고, BTC-KRW 수익률의 lag 0~8주, FX, NVDA, GPU 1주 자기시차를 설명변수로 넣었다. 개별 lag 하나의 p-value보다 BTC lag 전체의 공동 유의성 검정을 핵심 기준으로 삼았다. 표준오차는 시계열 자기상관과 이분산을 고려해 HAC/Newey-West 방식으로 보정하였다.

```text
gpu_return_t
= BTC lag effects from week 0 to week 8
+ FX control
+ NVDA control
+ GPU return lag 1
+ error
```

장기 관계는 log GPU index와 log BTC-KRW price를 사용해 ADF, Engle-Granger 공적분 검정, ECM 순서로 확인하였다. ADF는 가격 수준이 비정상인지 확인하기 위해, 공적분은 두 가격 수준 사이의 장기 균형관계를 보기 위해, ECM은 장기 균형에서 벗어난 오차가 다음 시점에 조정되는지를 보기 위해 사용하였다.

### H1 Numeric Results

BTC-GPU lag correlation은 lag 0에서 약 `-0.222`로 가장 음의 값이 컸고, lag 2~8에서는 약한 양의 상관이 보였다. 통제변수를 제거한 partial correlation은 lag 2에서 약 `0.186`으로 가장 컸지만, 전체적으로 일관적인 구조는 아니었다. 메인 distributed lag regression의 BTC lag 공동 F-test p-value는 `0.1017`로 5% 기준에서 유의하지 않았다. FX와 NVDA를 포함한 VAR Granger 검정도 p-value `0.1114`로 BTC가 GPU 수익률의 유의한 선행지표라고 보기 어려웠다. 장기 검정에서도 Engle-Granger p-value는 `0.3463`, ECM error correction term p-value는 `0.9220`으로 장기 균형관계와 균형 회귀 조정력이 확인되지 않았다.

BTC-ETC 수익률 상관은 `0.7457`로 높아 두 자산이 crypto sentiment를 공유한다는 점은 확인되었다. 그러나 GPU-ETC lag correlation은 lag 0에서 약 `-0.193`이고, 이후 lag에서 약한 양의 값만 나타나 ETC도 GPU 가격과 강한 양의 관계를 보이지 않았다.

### H1 Plot Interpretation

| Plot | What was checked | Interpretation |
|---|---|---|
| `BTC-KRW_and_ETC-KRW_weekly_log_returns` | BTC와 ETC가 같은 crypto cycle을 공유하는지 | 둘은 대체로 같이 움직인다. 따라서 BTC를 직접 채굴 수요가 아니라 crypto sentiment proxy로 해석하는 것이 타당하다. |
| `CCF-style_lag_correlation_BTC-KRW_return_vs_GPU_return` | BTC lag별 GPU 수익률 상관과 confidence band | 대부분 band 안에 있고 lag 0은 음수이다. BTC가 안정적으로 GPU 가격을 선행한다는 증거는 약하다. |
| `Hypo_1_distributed_lag_coefficients_*` | BTC lag별 회귀계수와 신뢰구간 | 대부분 신뢰구간이 0을 포함한다. 개별 lag 효과를 강하게 주장하기 어렵고, 공동검정이 핵심이다. |
| `Partial_correlation_*` | FX, NVDA, 자기시차 제거 후 BTC-GPU 관계 | lag 1~2에서 신호가 커지지만 이후 사라진다. 통제 후에도 관계가 안정적이지 않다. |
| `Rolling_BTC-GPU_return_correlation` / `Rolling_BTC_beta_of_GPU_return` | 기간별 상관과 beta 변화 | 시기별로 부호와 크기가 크게 바뀐다. 특히 2026년 초에는 음의 관계가 강해져, 전체 기간에 안정적인 관계가 있다고 보기 어렵다. |
| `H1_residual_plot` / `H1_residual_Q-Q_plot` / `Residual_distribution_*` | 회귀 잔차의 이상치와 정규성 | 2026년 초 큰 잔차 spike가 있고 Q-Q plot은 fat-tail을 보인다. 단순 OLS p-value만 신뢰하기 어려워 HAC 및 robust check가 필요하다. |
| `Squared_residuals_*` / `BTC_volatility_proxy_*` | 변동성 군집 여부 | 잔차 제곱이 2026년 초 집중적으로 커진다. BTC 변동성만으로 연속적으로 설명되기보다 특정 시장 충격이 존재했음을 시사한다. |

**Conclusion for H1:** 일부 약한 시차 신호는 있지만, 공동검정, Granger, 공적분, ECM, rolling 분석 모두 안정적인 BTC-GPU 관계를 강하게 지지하지 않는다.

---

## 5. Hypothesis 2: Performance-group BTC Sensitivity

Hypo 2는 고성능 GPU가 BTC 가격 변동에 더 민감하게 반응하는지 검정한다. 본분석은 chip 성능군을 기준으로 했고, VRAM 기준은 보조 robustness로만 사용하였다. 성능군은 chip tier로 정의하였다. high는 RTX/RX 80급 이상, 90급, 일부 70 Ti/Super급, mid는 RTX/RX 60급, 70급 일반형, GTX 1660 이상 및 주요 ARC 중급형, low는 GT 계열, GTX 1660 미만, RX 하위급과 하위 ARC이다.

먼저 ANOVA와 Welch ANOVA로 성능군 평균 수익률이 다른지 확인했다. 그러나 ANOVA는 평균 차이만 보는 검정이므로, BTC 민감도 차이를 직접 검정하기 위해 BTC lag와 성능군 dummy의 상호작용 회귀를 사용하였다. low 그룹은 표본이 작기 때문에 최종 모형은 mid를 기준항으로 두고 high-vs-mid를 1차 판단으로 보며, high-vs-low는 별도 contrast로 확인하였다. AI 수요가 high GPU에 더 크게 작용할 수 있으므로 NVDA와 NVDA×성능군 interaction을 통제하였다.

```text
chip_group_gpu_return
= common BTC lag effects
+ additional BTC lag effects by performance group
+ FX / NVDA / NVDA×group controls
+ product-composition controls in robustness checks
+ error
```

### H2 Numeric Results

chip 기준 ANOVA p-value는 `0.6163`, Welch ANOVA p-value는 `0.6212`로 성능군 평균 수익률 차이가 유의하지 않았다. 기존 low 기준 검정에서 high-vs-low BTC interaction p-value는 `0.9004`, mid-vs-low는 `0.4338`, 전체 interaction은 `0.6018`이었다. mid 기준으로 재정리해도 high-vs-mid p-value는 약 `0.7469`로 유의하지 않았다. high-minus-low 누적효과는 estimate `-0.0143`, p-value `0.6968`로 high 그룹이 low보다 더 민감하다는 증거가 없었다. 제품구성 통제 후 high-vs-low p-value는 `0.9136`, lag-window에서는 `0.5322`, product FE에서는 `0.9452`로 결론이 유지되었다. VIF 최대값은 약 `1.44`로 다중공선성은 심하지 않았다.

VRAM proxy에서는 high-vs-low p-value가 `0.4650`으로 유의하지 않았고, mid-vs-low는 `0.0136`으로 일부 신호가 있었다. 그러나 VRAM과 chip 성능군의 불일치율이 약 `24.86%`이고, 특히 mid chip이 high VRAM으로 분류되는 경우가 많기 때문에 이 결과는 “고성능 chip 효과”가 아니라 메모리 용량 또는 메모리 집약적 작업 수요의 보조 신호로만 해석한다.

### H2 Plot Interpretation

| Plot | What was checked | Interpretation |
|---|---|---|
| `Hypo_2_Phase_1_chip_group_weekly_returns` | 성능군별 주간 수익률 분포 | 중앙값이 모두 0 근처이고 outlier만 존재한다. ANOVA/Welch가 유의하지 않은 결과와 일치한다. |
| `BTC-KRW_return_vs_GPU_return_by_chip_performance_group` | BTC 수익률과 성능군별 GPU 수익률 산점도 | 세 그룹의 추세선이 모두 약한 음의 기울기 또는 거의 평평한 형태이다. high 그룹의 기울기가 더 크다는 시각적 증거가 없다. |
| `Hypo_2_group-specific_BTC_lag_coefficients_chip_main` | chip 성능군별 BTC lag 계수 | high/mid/low 계수가 서로 명확히 분리되지 않고 신뢰구간이 대부분 0을 포함한다. 성능군별 민감도 차이가 약하다. |
| `Hypo_2_EDA_group-specific_CCF_*` | 성능군별 BTC lag CCF | lag 0은 세 그룹 모두 음수이고 대부분 95% band 안에 있다. high 그룹만 더 빠르거나 강하게 반응한다고 보기 어렵다. |
| `Hypo_2_active_product_count_*` / `new_product_share_*` | 제품 구성 변화 | high/mid/low의 active product 수와 신규 제품 비율이 시기별로 크게 변한다. raw group 비교만 하면 제품 구성 변화가 섞일 수 있어 composition-control이 필요했다. |
| `Hypo_2_chip_performance_group_price_index` | chip 성능군별 가격지수 | 2026년 초 모든 그룹이 뛰며 low/mid가 high보다 더 크게 오르는 구간도 있다. “고성능 GPU가 BTC에 더 민감하다”는 패턴은 보이지 않는다. |
| `Hypo_2_robustness_VRAM_proxy_group_price_index` | VRAM 기준 가격지수 | VRAM mid/high가 다른 움직임을 보이지만 VRAM은 chip 성능과 다르다. 보조 신호로만 해석해야 한다. |
| `H2_chip_main_model_date-level_mean_residual` / `Q-Q plot` | 모형 잔차와 fat-tail | 2026년 초 잔차 spike와 Q-Q plot의 fat-tail이 보인다. 특정 시기 충격이 크므로 robust/clustered SE와 잔차진단이 필요하다. |
| `Hypo_2_EDA_chip_group_rolling_volatility` | 성능군별 변동성 | 2026년 초 모든 그룹의 변동성이 크게 상승한다. 개별 성능군의 BTC 민감도보다 시장 충격이 더 강하게 보인다. |

**Conclusion for H2:** chip 기준 본모형, composition-control, lag-window, Almon lag, product FE 모두 고성능 GPU의 추가 BTC 민감도를 지지하지 않는다. VRAM proxy의 일부 유의 신호는 성능 효과가 아니라 메모리 용량 관련 보조 신호로만 해석한다.

---

## 6. Final Takeaways

이 프로젝트는 제품 구성 변화가 큰 GPU 가격 데이터를 단순 평균하지 않고 Chain Matched Jevons Index로 변환했으며, 비정상 가격 수준으로 인한 허위상관을 피하기 위해 주간 로그수익률을 사용하였다. EDA에서 표본 안정성, chain drift, trim sensitivity, 계절성, 제품 구성 변화를 먼저 확인했고, 본검정에서는 FX, NVDA, 자기시차, 제품구성 proxy를 통제하였다. Hypo 1은 BTC가 전체 GPU 가격을 안정적으로 설명한다는 증거를 보이지 않았고, Hypo 2도 고성능 GPU가 BTC에 더 민감하다는 증거를 보이지 않았다. 따라서 본 데이터에서는 Ethereum PoS 전환 이후 암호화폐 시장 변동성이 한국 GPU 가격을 직접적으로 견인한다는 주장이 강하게 지지되지 않는다.
