# Hypo3 최종 파이프라인 요약

## 1. 최종 질문과 결론

**질문:** 한국 GPU·CPU·RAM 소매가격 수익률은 BTC, SOXX, FX, Oil, CSI 같은 외부 거시·금융 변수로 설명되는가, 아니면 제품시장 내부요인으로 더 잘 설명되는가?

**최종 결론:** broad macro 변수만으로는 가격수익률을 안정적으로 설명하기 어렵다. 반면 **GPU와 RAM은 내부 시장요인으로 설명력이 크게 개선**된다. CPU는 macro와 internal 모두 설명력이 낮아 보수적으로 해석한다.

---

## 2. 분석 파이프라인

```text
Danawa 제품별 가격 데이터
→ 주간 제품 median price 생성
→ Chain Matched Jevons 주간 로그수익률 생성
→ macro current relation 검정
→ BTC-GPU, SOXX-RAM 1~8주 lag 검정
→ 제품 패널에서 내부 시장요인 생성
→ macro-only / internal-only / combined 모델 비교
```

분석 단위는 모두 **주간 로그수익률**이다. 가격 level은 제품구성 변화와 비정상성 위험이 있어 최종 회귀의 종속변수로 쓰지 않았다.

---

## 3. 변수 선택 원칙

### 3.1 Macro 변수

| 변수 | 의미 | 사용 이유 |
|---|---|---|
| `btc_ret_z` | BTC-KRW 수익률 | crypto / risk-asset sentiment |
| `soxx_ret_z` | SOXX ETF 수익률 | NVDA보다 넓은 AI·반도체 수요 proxy |
| `fx_ret_z` | USD/KRW 수익률 | 한국 수입·소매가격의 환율 효과 |
| `oil_ret_z` | Brent oil 수익률 | 원가·물류·인플레이션 proxy |
| `csi_chg_z` | 한국 소비자심리지수(OECD Composite Consumer Confidence for Korea, FRED `CSCICP02KRM066S`, 월간) 변화 | 소비수요 환경 proxy. 원자료가 월간이라 주간 변화는 월 1회만 반영되는 한계가 있다 |

SOXX를 AI·반도체 수요 proxy로 도입했으므로, 최종 Hypo3에서는 **NVDA를 제외**했다. SOXX와 NVDA를 동시에 넣으면 동일한 반도체 수요 신호를 중복 통제하게 된다.

### 3.2 내부 시장요인

초기 내부변수는 많았지만, 최종 회귀에는 해석 가능하고 중복이 적은 변수만 남겼다.

| 부품 | 최종 내부 변수 | 의미 |
|---|---|---|
| GPU | `gpu_n_chg` | active GPU 제품 수 로그 변화 |
| GPU | `gpu_new_sh` | 신규 GPU 관측 제품 비중 |
| GPU | `gpu_riq_l1` | 전주 GPU 제품별 수익률 IQR |
| GPU | `gpu_high_chg` | high-tier GPU 비중 변화 |
| GPU | `gpu_vram_chg` | 평균 VRAM 변화 |
| CPU | `cpu_n_chg` | active CPU 제품 수 로그 변화 |
| CPU | `cpu_new_sh` | 신규 CPU 관측 제품 비중 |
| CPU | `cpu_riq_l1` | 전주 CPU 제품별 수익률 IQR |
| CPU | `cpu_amd_chg` | AMD CPU 비중 변화 |
| RAM | `ram_n_chg` | active RAM 제품 수 로그 변화 |
| RAM | `ram_new_sh` | 신규 RAM 관측 제품 비중 |
| RAM | `ram_riq_l1` | 전주 RAM 제품별 수익률 IQR |
| RAM | `ram_ddr5_chg` | DDR5 비중 변화 |
| RAM | `ram_cap_chg` | 평균 RAM 용량 변화 |

제외한 변수는 이유가 명확하다. `match_sh`는 `new_sh`와 거의 반대 관계라 제외했고, `gpu_mid_sh`, `gpu_low_sh`, `cpu_intel_sh`는 기준 share와 중복된다. `gpu_vram_hi_sh`, `ram_cap_hi_sh`는 임의 threshold가 들어가므로 제외했다. `ret_iqr`는 같은 주 수익률에서 계산되므로 동시성 위험을 줄이기 위해 **1주 lag**만 사용했다.

---

## 4. 내부요인 데이터셋의 근거

제품 패널 규모는 충분하다.

| 부품   |   제품 수 |   주차 수 |   median active |   median match share |
|:-------|----------:|----------:|----------------:|---------------------:|
| gpu    |      1979 |       110 |           727   |                0.973 |
| cpu    |       446 |       110 |           212.5 |                0.982 |
| ram    |      4288 |       110 |          1364.5 |                0.85  |

결측은 MCAR 보간이 아니라 **구조적 결측 제거**로 처리했다. 첫 주 return, 첫 차분, active product count가 0인 관측 공백, 공백 직후 신규비중 artifact, lag 변수 정의 불가능 주차를 제거했다. 최종 회귀 표본은 **110주 중 103주**, 최종 `reg.csv` 결측치는 **0개**다.

---

## 5. 현재시점 macro relation 결과

현재시점 회귀에서는 factor별 통제변수를 다르게 적용했다. 예를 들어 BTC 검정은 `FX + SOXX`, SOXX 검정은 `FX + BTC`를 통제했다.

| 관계     |    coef |     p |   partial R² |
|:---------|--------:|------:|-------------:|
| BTC-GPU  | -0.0026 | 0.094 |        0.067 |
| SOXX-RAM |  0.0034 | 0.052 |        0.018 |

GPU-BTC는 10% 수준의 약한 후보 신호이고, RAM-SOXX도 5% 기준에는 근접하지만 확정적이지 않다. 나머지 macro factor는 설명력이 더 약했다. 따라서 현재시점 분석만으로는 macro 변수가 부품 가격수익률을 안정적으로 설명한다고 보기 어렵다.

---

## 6. Time-lag 분석 결과

현재시점 결과에서 후보로 남은 **BTC-GPU**와 **SOXX-RAM**만 1~8주 lag를 검정했다.

| 관계     |   best lag by p |     p |   q(BH) | max partial R²   | 방향         |
|:---------|----------------:|------:|--------:|:-----------------|:-------------|
| BTC-GPU  |               1 | 0.344 |   0.645 | 0.013 (lag 1)    | 불안정       |
| SOXX-RAM |               3 | 0.099 |   0.437 | 0.024 (lag 4)    | 전 lag 양(+) |

BTC-GPU는 부호가 lag별로 바뀌고 유의하지 않았다. SOXX-RAM은 모든 lag에서 양(+)이지만, BH 보정 q-value가 높아 안정적인 선행효과로 보기 어렵다. 즉 “몇 주 전 macro factor만으로 가격 방향을 예측한다”는 결론은 지지되지 않는다.

---

## 7. 최종 모델 비교

최종 비교는 세 모델만 사용했다.

```text
Macro-only    : ret ~ btc + fx + soxx + oil + csi
Internal-only : ret ~ n_chg + new_sh + riq_l1 + spec-mix change
Combined      : ret ~ macro variables + internal variables
```

모든 설명변수는 최종 103주 표본에서 표준화했다. HAC 표준오차를 사용했고, VIF, Ljung-Box, Breusch-Pagan을 확인했다. 최종 최대 VIF는 약 **3.60**으로 심각한 공선성 신호는 아니었다.

| hw   | model    |   Adj.R² |    MAE | Dir.Acc.   |   VIFmax |
|:-----|:---------|---------:|-------:|:-----------|---------:|
| GPU  | macro    |    0.039 | 0.0052 | 48.5%      |     1.2  |
| GPU  | internal |    0.532 | 0.0041 | 66.0%      |     3.03 |
| GPU  | combined |    0.525 | 0.0042 | 65.0%      |     3.21 |
| CPU  | macro    |   -0.031 | 0.0039 | 55.3%      |     1.2  |
| CPU  | internal |   -0.007 | 0.0038 | 64.1%      |     3.54 |
| CPU  | combined |   -0.043 | 0.0038 | 61.2%      |     3.6  |
| RAM  | macro    |   -0.021 | 0.0138 | 55.3%      |     1.2  |
| RAM  | internal |    0.571 | 0.0098 | 57.3%      |     2.32 |
| RAM  | combined |    0.567 | 0.0102 | 56.3%      |     2.54 |

핵심은 GPU와 RAM이다. GPU의 adjusted R²는 macro-only **0.039**에서 internal-only **0.532**로 증가했고, RAM은 macro-only **-0.021**에서 internal-only **0.571**로 증가했다. CPU는 모든 모델의 adjusted R²가 0 근처 또는 음수라 강한 설명력을 주장하지 않는다.

---

## 8. 추가 설명력 검정

| hw   | test                   |        p |   ΔAdj.R² |    ΔMAE |
|:-----|:-----------------------|---------:|----------:|--------:|
| GPU  | internal_adds_to_macro | 5.32e-06 |     0.485 |  0.001  |
| GPU  | macro_adds_to_internal | 0.78     |    -0.008 | -0.0001 |
| CPU  | internal_adds_to_macro | 0.006    |    -0.012 |  0.0002 |
| CPU  | macro_adds_to_internal | 0.897    |    -0.036 |  0      |
| RAM  | internal_adds_to_macro | 7.22e-23 |     0.588 |  0.0036 |
| RAM  | macro_adds_to_internal | 0.656    |    -0.004 | -0.0004 |

GPU와 RAM에서는 내부요인을 macro 모델에 추가했을 때 설명력이 크게 증가했다. 반대로 internal 모델에 macro 변수를 추가하면 개선되지 않았다. CPU는 내부변수의 joint p-value는 작지만 adjusted R²와 AIC가 나빠져 실질 개선으로 해석하지 않았다.

---

## 9. 주요 계수 해석

| 부품   | 변수         |    coef |        p |
|:-------|:-------------|--------:|---------:|
| GPU    | gpu_riq_l1   |  0.0044 | 4.41e-06 |
| GPU    | gpu_high_chg |  0.0052 | 0.002    |
| GPU    | gpu_new_sh   |  0.0046 | 0.003    |
| GPU    | gpu_n_chg    | -0.0021 | 0.003    |
| GPU    | gpu_vram_chg | -0.0066 | 0.002    |
| CPU    | cpu_riq_l1   | -0.0009 | 0.03     |
| RAM    | ram_riq_l1   |  0.0146 | 7.43e-07 |
| RAM    | ram_new_sh   |  0.0098 | 0.055    |
| RAM    | ram_n_chg    | -0.0097 | 0.094    |
| RAM    | soxx_ret_z   |  0.0031 | 0.094    |

GPU는 전주 제품별 가격분산, 신규 유입, high-tier 비중 변화가 양(+)의 관계를 보이고, 제품 수 증가와 VRAM 평균 변화는 음(-)의 관계를 보였다. RAM은 전주 제품별 가격분산이 가장 강하고, 신규 유입과 제품 수 변화도 약한 후보 신호다. CPU는 `cpu_riq_l1`만 약하게 나타나지만 전체 모델 설명력이 낮아 보수적으로 해석한다.

---

## 10. 최종 해석

이 분석의 결론은 “컴퓨터 부품 가격은 경제적 요인과 무관하다”가 아니다. 더 정확한 결론은 다음이다.

> BTC, SOXX, FX, Oil, CSI 같은 broad macro/financial variables는 한국 GPU·CPU·RAM 가격수익률을 안정적으로 설명하지 못했다. 반면 GPU와 RAM에서는 제품 수 변화, 신규 유입, 전주 제품별 가격변동 분산, 사양 mix 변화 같은 제품시장 내부요인이 훨씬 높은 설명력을 보였다. 따라서 본 데이터에서는 컴퓨터 부품 가격, 특히 GPU와 RAM 가격이 넓은 거시 금융지표보다 제품시장 내부 구조와 구성 변화에 더 크게 좌우되는 것으로 해석된다. CPU는 두 변수군 모두 설명력이 제한적이다.

**한 줄 결론:** broad macro proxies are weak; product-market internal variables dominate for GPU and RAM.
