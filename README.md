# Bitcoin Effect on Hardware Pricing

암호화폐 시장 변동이 GPU 가격에 미치는 영향을 검정하기 위한 프로젝트이다. 원시 제품 가격표를 그대로 평균하지 않고, 제품 구성 변화와 크롤링 노이즈를 줄인 주간 가격지수와 수익률을 만든 뒤 가설검정을 수행하였다.

## 0. Key Methodologies Reference

본 프로젝트에서 활용된 주요 통계 및 시계열 분석 방법론의 도입 목적과 해석 기준이다. 

* <strong>Chain Matched Jevons Index</strong>: 매주 관측되는 제품 구성이 달라져서 생기는 평균 가격의 왜곡을 막기 위해, <strong>전주와 이번 주에 모두 존재하는 동일 제품의 가격 변화율만 연결</strong>하여 지수화할 때 사용한다. 신제품 출시나 단종 노이즈가 제거된 순수한 시장 가격 변동 흐름으로 해석한다.
* <strong>로그 수익률 (Log Return)</strong>: 지속적으로 상승하거나 하락하는 추세(Level)를 가진 데이터를 회귀분석할 때 생기는 <strong>가짜 상관관계(Spurious Correlation)를 방지</strong>하기 위해 사용한다. 시계열 데이터의 안정성을 확보하며, 주간 단위의 실질적인 가격 변동률로 해석한다.
* <strong>HAC / Newey-West Standard Error</strong>: 시계열 데이터의 특성상 과거의 충격이 현재에 영향을 주거나 변동폭이 일정하지 않을 때 <strong>표준오차를 보정</strong>하기 위해 사용한다. 이를 통해 도출된 p-value가 통계적으로 신뢰할 수 있음을 보증한다.
* <strong>VAR Granger Causality Test</strong>: 한 변수의 과거 값이 다른 변수의 현재를 예측하는 데 통계적으로 기여하는지 확인할 때 쓰인다. 본 분석에서는 비트코인 변동이 GPU 가격 변동을 이끄는 <strong>선행 지표 역할을 하는지</strong> 판단하는 기준이 된다.
* <strong>Engle-Granger Cointegration & ECM</strong>: Cointegration(공적분)은 개별 데이터가 불안정하더라도 두 지표 사이에 <strong>장기적인 균형 궤적</strong>이 존재하는지 검증할 때 쓰인다. ECM(오차수정모형)은 그 균형이 단기적으로 깨졌을 때 <strong>다시 원래의 장기 균형으로 회귀하는 속도와 방향</strong>이 있는지를 해석할 때 사용한다.
* <strong>Welch ANOVA & 상호작용 회귀 (Interaction Regression)</strong>: Welch ANOVA는 그룹 간 데이터의 분산이 다를 때 평균 차이를 보다 안전하게 검정하기 위해 사용한다. 상호작용 회귀는 비트코인 수익률이라는 요인이 GPU의 성능군(High/Mid/Low)에 따라 <strong>서로 다른 민감도로 영향을 미치는지</strong>를 분리해서 해석할 때 활용한다.

## 1. Pipeline Overview

```text
data_preprocessing.ipynb
→ EDA_process.ipynb
→ Hypo_1.ipynb
→ Hypo_2.ipynb
```

| Step | Purpose | Main Output |
|---|---|---|
| Data Preprocessing | 원시 제품 가격표를 주간 가격지수와 수익률로 변환 | `idx_*`, `ret_*`, `panel_gpu`, `diag_*` |
| EDA Process | 지수 품질, 표본 안정성, 기초 상관 확인 | `eda_health`, `eda_corr_all`, `eda_beta` |
| Hypo 1 | BTC와 전체 GPU 가격 사이의 단기/장기 공변동 검정 | `h1_*` |
| Hypo 2 | 고성능 GPU가 BTC 변동에 더 민감한지 검정 | `h2_*` |

---

## 2. Data Preprocessing

### 2.1 Core Decisions

원시 데이터의 `0` 가격은 실제 거래가격이 아니라 출시 전, 단종, 품절, 크롤링 실패 등이 섞인 관측 불가능 상태로 보았다. 따라서 가격 계산에서는 0을 유효 가격으로 사용하지 않았다.

동일 제품의 중복 row는 날짜별 median 가격으로 병합하였다. 평균 대신 median을 사용한 이유는 판매처별 극단값과 크롤링 노이즈에 덜 민감하기 때문이다.

GPU 제품명에서 chip, VRAM, 성능군 정보를 추출하였다. 성능군 본분석은 `chip_perf_group`을 사용했고, VRAM 기준 분류는 보조 proxy로만 사용하였다.

### 2.2 GPU Performance Group Rule

성능군은 벤치마크 점수를 직접 사용한 것이 아니라, 제품명에서 추출한 GPU chip tier를 기준으로 high/mid/low로 나누었다.

| Group | Approximate Rule | Examples / Interpretation |
|---|---|---|
| `high` | 최상위·상위 게이밍/연산 chip | RTX/RX 80급 이상, 90급, 일부 70 Ti/Super급 등 |
| `mid` | 주류 게이밍급 chip | RTX/RX 60급, 70급 일반형, GTX 1660 이상, 주요 ARC 중급형 등 |
| `low` | 엔트리·구형·저성능 chip | GT 계열, GTX 1660 미만, RX 하위급, 하위 ARC 등 |

VRAM proxy는 다음처럼 별도 정의했다.

| VRAM Group | Rule |
|---|---|
| `high` | VRAM ≥ 12GB |
| `mid` | 6GB < VRAM < 12GB |
| `low` | VRAM ≤ 6GB |

VRAM은 실제 성능과 완전히 일치하지 않는다. 예를 들어 RTX 3060 12GB는 VRAM은 크지만 성능은 mid급에 가깝고, 일부 8GB 모델은 VRAM은 작아도 chip 성능은 high에 가까울 수 있다. 따라서 Hypo 2의 본결론은 chip 기준 분석에 두고, VRAM 분석은 robustness 또는 보조 결과로 해석하였다.

### 2.3 Chain Matched Jevons Index

단순 평균 가격은 매주 관측되는 제품 구성이 달라질 때 왜곡된다. 따라서 인접한 두 주에 모두 가격이 관측된 동일 제품만 매칭하여 가격 변화를 계산하였다.

그룹 \(g\)의 \(t\)주에 대해 \(M_{g,t}\)는 \(t-1\)주와 \(t\)주에 동시에 관측된 product의 집합이다. 그룹의 수익률(가격변동성)은

$$
\begin{aligned}
r_{g,t}
&=
\frac{1}{|M_{g,t}|}
\sum_{i \in M_{g,t}}
\left(\log p_{i,t} - \log p_{i,t-1}\right)
\end{aligned}
$$

체인 인덱스는

$$
\begin{aligned}
I_{g,t}
&=
I_{g,t-1}\exp(r_{g,t}),
\qquad I_{g,0}=100
\end{aligned}
$$

기호 의미: `g`는 분석 그룹, `t`는 주차, `i`는 제품, `p_{i,t}`는 제품 `i`의 `t`주 가격, `r_{g,t}`는 그룹 주간 로그수익률, `I_{g,t}`는 체인 가격지수이다.

이 방식은 “이번 주에 어떤 제품이 새로 들어왔는가”가 아니라 “같은 제품의 가격이 얼마나 변했는가”를 추적한다. 따라서 제품 구성 변화에 의한 대표가격 왜곡을 줄인다.

### 2.4 Main Outputs

| Output | Meaning |
|---|---|
| `idx_all_w`, `ret_all_w` | 전체 GPU/CPU/RAM 주간 가격지수 및 수익률 |
| `idx_chip_w`, `ret_chip_w` | chip 성능군별 GPU 가격지수 및 수익률 |
| `idx_vram_w`, `ret_vram_w` | VRAM proxy 기준 GPU 가격지수 및 수익률 |
| `panel_gpu` | 제품 단위 GPU 패널 데이터 |
| `diag_all`, `diag_chip`, `diag_vram` | 매칭 제품 수, valid week, gap 진단 |

---

## 3. EDA Process

EDA의 목적은 본검정 전에 지수가 분석 가능한 품질인지 확인하고, 단순 상관만으로 가설을 판단하지 않도록 하는 것이다.

### 3.1 Index Quality

전체 지수는 109개 주간 step 중 106개가 valid였고, return 기준 105개 주가 분석 가능했다. median matched product 수는 GPU 711개, CPU 209개, RAM 1156개였다.

성능군 기준으로도 high/mid/low 모두 105개 valid return을 확보했다. median matched product 수는 high 269개, mid 316개, low 43개였다. low 그룹은 표본 수가 작아 해석에 주의가 필요하지만, 성능군별 검정 자체는 가능한 수준으로 판단하였다.

### 3.2 Preliminary Correlation and Beta

가격 level이 아니라 주간 로그수익률을 사용하였다. 이는 추세가 있는 가격 수준에서 발생하는 spurious correlation을 피하기 위한 선택이다.

기초 상관에서 GPU와 BTC 수익률의 상관은 약 \(-0.246\)이었다. CPU는 약 \(0.094\), RAM은 약 \(-0.086\)이었다. 단순 회귀 beta는 GPU \(-0.0399\), CPU \(0.0096\), RAM \(-0.0374\)였다.

이 결과는 BTC 상승이 GPU 가격 상승으로 단순 연결된다는 강한 패턴을 보여주지 않았다. 따라서 본검정에서는 시차, 환율, NVDA, 자기시차를 통제하였다.

---

## 4. Controls and Limitations

### 4.1 Controls Used

| Variable | Role |
|---|---|
| `fx_ret` | 원화 GPU 가격에서 USD/KRW 환율 효과 통제 |
| `nvda_ret` | AI/반도체 수요 proxy |
| `gpu_ret_l1` | GPU 수익률의 자기시차 통제 |
| `ETC-KRW` | GPU mining 관련 도메인 보조 EDA; 메인 회귀에는 미포함 |
| `active_product_n`, `new_product_share`, `bridge_ratio` | Hypo 2에서 신제품/제품구성 변화 proxy |

BTC는 현대 기간에서 GPU 채굴 직접 수요라기보다 crypto sentiment 또는 위험자산 심리 proxy로 해석하였다. Ethereum은 2022년 Proof-of-Stake로 전환했기 때문에, 2024년 이후 분석에서 GPU 채굴 수요를 직접 주장하지 않았다.

### 4.2 Limitations

금리, 글로벌 유동성, VIX/NASDAQ/SOX, 정확한 신제품 출시 dummy, 재고/품절, 국내 소비 시즌성은 직접 통제하지 못했다. 신제품 효과는 임의 출시일 dummy를 만들기보다 제품구성 proxy로 보완했고, 남는 부분은 한계로 인정하였다.

---

## 5. Hypothesis 1: BTC-GPU Co-movement

### 5.1 Hypothesis

> 전체 GPU 가격과 BTC 가격 사이에는 유의미한 공변동 관계가 존재한다.

BTC 효과가 즉시 반영되지 않을 수 있으므로 lag 0~8주의 BTC 수익률을 사용하였다.

### 5.2 Short-run Model

주요 회귀식은 다음과 같다.

$$
\begin{aligned}
r^{GPU}_t
&=
\alpha
+
\sum_{k=0}^{8}\beta_k r^{BTC}_{t-k}
+
\gamma_1 r^{FX}_t
+
\gamma_2 r^{NVDA}_t
+
\rho r^{GPU}_{t-1}
+
\epsilon_t
\end{aligned}
$$

핵심 공동 검정은 다음과 같다.

$$
H_0:\ \beta_0=\beta_1=\cdots=\beta_8=0
$$

기호 의미: `r^{GPU}_t`는 전체 GPU 주간 로그수익률, `r^{BTC}_{t-k}`는 `k`주 전 BTC-KRW 로그수익률, `r^{FX}_t`는 USD/KRW 환율 수익률, `r^{NVDA}_t`는 NVDA 수익률, `β_k`는 BTC 시차 효과, `γ`는 통제변수 효과, `ρ`는 GPU 자기시차 효과, `ε_t`는 잔차이다.

개별 시차 계수의 p-value보다 BTC 시차 구조 전체의 공동 유의성을 핵심 판단 기준으로 사용하였다. 표준오차는 시계열 자기상관과 이분산 가능성을 고려해 HAC/Newey-West 방식으로 보정하였다.

### 5.3 Long-run Model

가격 수준 변수의 장기 관계는 ADF 단위근 검정과 Engle-Granger 공적분 검정으로 확인하였다. 공적분 가능성이 있을 경우 ECM을 사용하였다.

$$
\begin{aligned}
\Delta y_t
&=
\alpha
+
\lambda u_{t-1}
+
\sum_{k=0}^{K}\beta_k \Delta b_{t-k}
+
\gamma_1 \Delta fx_t
+
\gamma_2 \Delta nvda_t
+
\epsilon_t
\end{aligned}
$$

여기서

$$
u_{t-1}=y_{t-1}-\hat{a}-\hat{b}b_{t-1}
$$

기호 의미: `y_t`는 log GPU 가격지수, `b_t`는 log BTC 가격, `Δ`는 1차 차분, `u_{t-1}`는 전기 장기균형 오차, `λ`는 장기균형 조정 속도이다.

### 5.4 Results

| Test | Result | Interpretation |
|---|---:|---|
| Max lag correlation | lag 0, \(-0.222\) | 동시점 단순상관은 음수 |
| Max partial correlation | lag 2, \(0.186\) | 통제변수 제거 후 약한 지연 신호 존재 |
| BTC lag joint F-test | p = 0.1017 | 5% 유의수준에서 기각 실패 |
| VAR Granger with FX/NVDA | p = 0.1114 | BTC는 5% 기준에서 유의한 선행지표가 아님 |
| Engle-Granger cointegration | p = 0.3463 | 장기 균형관계 확인되지 않음 |
| ECM error correction term | p = 0.9220 | 장기 균형으로 되돌아가는 조정 효과 없음 |
| BTC-ETC return correlation | 0.7457 | 두 자산은 공통 crypto sentiment를 공유 |
| Max GPU-ETC lag correlation | lag 0, \(-0.193\) | ETC도 GPU와 강한 양의 관계를 보이지 않음 |

### 5.5 Conclusion

Hypothesis 1은 강하게 지지되지 않았다. 일부 시차별 신호는 존재하지만, BTC 시차항 공동 검정, VAR Granger 검정, 공적분 검정, ECM 모두 안정적인 BTC-GPU 관계에 대한 강한 근거를 제공하지 못했다.

---

## 6. Hypothesis 2: Performance-group BTC Sensitivity

### 6.1 Hypothesis

> 고성능 GPU일수록 BTC 가격 변동에 더 강하게 반응한다.

본분석에서는 chip 기반 성능군을 사용하였다. VRAM 기준 성능군은 보조적 강건성 검정으로만 사용하였다. VRAM 용량과 실제 연산 성능이 항상 일치하지 않기 때문이다.

### 6.2 Exploratory ANOVA

ANOVA는 성능군별 평균 수익률이 다른지 확인하는 탐색적 검정으로만 사용하였다.

$$
H_0:\ \mu_{high}=\mu_{mid}=\mu_{low}
$$

기호 의미: `μ_high`, `μ_mid`, `μ_low`는 각각 high/mid/low 성능군의 평균 주간 수익률이다.

Levene 검정으로 등분산성을 먼저 확인하였다. 등분산 가정이 약할 수 있으므로, 분산이 달라도 사용할 수 있는 Welch ANOVA도 함께 수행하였다.

### 6.3 Main Interaction Regression

ANOVA는 BTC 민감도가 성능군별로 다른지를 직접 검정하지 못한다. 따라서 본검정은 상호작용 회귀모형으로 수행하였다.

$$
\begin{aligned}
r_{g,t}
&=
\alpha_g
+
\sum_{k=0}^{8}\beta_k r^{BTC}_{t-k} \\
&\quad+
\sum_{k=0}^{8}\delta_{high,k}
\left(r^{BTC}_{t-k}\times D^{high}_g\right) \\
&\quad+
\sum_{k=0}^{8}\delta_{mid,k}
\left(r^{BTC}_{t-k}\times D^{mid}_g\right)
+
\Gamma Controls_{g,t}
+
\epsilon_{g,t}
\end{aligned}
$$

기호 의미: `r_{g,t}`는 성능군 `g`의 주간 로그수익률, `D^{high}_g`와 `D^{mid}_g`는 성능군 dummy, `δ`는 low 그룹 대비 추가 BTC 민감도, `α_g`는 그룹 고정효과, `Controls`는 FX/NVDA 및 제품구성 통제변수이다.

핵심 검정은 다음과 같다.

$$
H_0:\ \delta_{high,0}=\delta_{high,1}=\cdots=\delta_{high,8}=0
$$

그리고

$$
H_0:\ \delta_{mid,0}=\delta_{mid,1}=\cdots=\delta_{mid,8}=0
$$

그룹 \(g\)의 BTC 누적 효과는 다음과 같이 계산하였다.

$$
\begin{aligned}
C_g
&=
\sum_{k=0}^{8}
\left(\beta_k+\delta_{g,k}\right)
\end{aligned}
$$

기호 의미: `C_g`는 0~8주 BTC 시차 효과를 모두 더한 그룹 `g`의 누적 BTC 반응이다.

개별 시차 계수보다 interaction 항의 공동 유의성 검정과 누적 효과를 핵심 판단 기준으로 사용하였다.

### 6.4 Results

| Test | Result | Interpretation |
|---|---:|---|
| chip ANOVA | p = 0.6163 | 성능군 평균 수익률 차이 없음 |
| chip Welch ANOVA | p = 0.6212 | 이분산 허용 후에도 같은 결론 |
| high vs low BTC interaction | p = 0.9004 | high GPU가 low GPU보다 BTC에 더 민감하지 않음 |
| mid vs low BTC interaction | p = 0.4338 | 유의한 차이 없음 |
| all chip interactions | p = 0.6018 | 성능군별 BTC 민감도 차이 없음 |
| high - low cumulative effect | estimate = -0.0143, p = 0.6968 | high가 low보다 더 민감하다는 증거 없음 |
| composition-control high vs low | p = 0.9136 | 제품구성 변화 통제 후에도 결론 유지 |
| lag-window high vs low | p = 0.5322 | 시차를 묶어도 결론 유지 |
| product FE high vs low | p = 0.9452 | 제품 고정효과 반영 후에도 결론 유지 |
| VRAM high vs low | p = 0.4650 | high-VRAM 그룹의 추가 민감도 증거 없음 |
| VRAM mid vs low | p = 0.0136 | 보조적 신호이나 proxy mismatch 영향 가능 |
| VIF | max about 1.44 | 다중공선성은 심하지 않음 |

### 6.5 Conclusion

Hypothesis 2는 지지되지 않았다. chip 기반 본모형, 제품구성 통제 강건성 검정, lag-window 강건성 검정, 제품 고정효과 모형 모두 고성능 GPU가 BTC 수익률에 더 민감하다는 증거를 보여주지 못했다.

---

## 7. Main Takeaways

1. 제품 단위 원시 가격은 제품구성 변화에 따른 왜곡을 줄이기 위해 chain matched weekly index로 변환하였다.
2. 비정상 가격 수준에서 발생할 수 있는 허위상관을 피하기 위해 주간 로그수익률을 사용하였다.
3. 환율 효과와 AI/반도체 수요 효과를 분리하기 위해 FX와 NVDA를 통제하였다.
4. Hypothesis 1은 강하게 지지되지 않았다. BTC는 전체 GPU 가격에 대해 안정적인 단기 또는 장기 설명력을 보이지 않았다.
5. Hypothesis 2도 지지되지 않았다. 고성능 GPU가 저성능 GPU보다 BTC 수익률에 더 강하게 반응한다는 증거는 확인되지 않았다.
