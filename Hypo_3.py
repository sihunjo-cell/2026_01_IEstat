import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# =========================================================
# 1. 하드웨어 주간 로그 수익률 데이터 로드 (ret_all_w.csv)
# =========================================================
print("1. 하드웨어 주간 로그 수익률 데이터를 로드합니다...")
# 팀원이 이미 Jevons Index 기반으로 로그 수익률 처리를 완료한 파일
df_ret = pd.read_csv('C:/Users/ME/Downloads/2026_01_IEstat-EDA_process/ret_all_w.csv')
df_ret['date'] = pd.to_datetime(df_ret['date'])

# LMM 모델링을 위해 Wide 포맷을 Long 포맷으로 변환
df_long = df_ret.melt(
    id_vars='date', 
    value_vars=['CPU_ALL', 'GPU_ALL', 'RAM_ALL'], 
    var_name='Category', 
    value_name='Log_Return'
)
df_long['Category'] = df_long['Category'].str.replace('_ALL', '')
df_long = df_long.dropna()

# =========================================================
# 2. 통제 변수 및 독립 변수 로드 & '로그 수익률' 변환
# =========================================================
print("2. yfinance에서 BTC, 환율, NVDA 데이터를 가져와 로그 수익률로 변환합니다...")
start_date = df_long['date'].min() - pd.Timedelta(days=14)
end_date = df_long['date'].max() + pd.Timedelta(days=7)

# 데이터 다운로드
tickers = {'BTC-KRW': 'BTC_Price', 'KRW=X': 'FX_Rate', 'NVDA': 'NVDA_Price'}
df_exog_raw = pd.DataFrame()

for ticker, col_name in tickers.items():
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)['Close']
    if isinstance(data, pd.DataFrame): # yfinance 버전에 따른 구조 차이 방어
        data = data.iloc[:, 0]
    df_exog_raw[col_name] = data

# 주간 단위(일요일 기준)로 리샘플링하여 하드웨어 데이터와 시점 동기화
df_exog_w = df_exog_raw.resample('W-SUN').last().reset_index()
df_exog_w.rename(columns={'Date': 'date'}, inplace=True)

# [핵심] 가설 1의 논리를 반영하여 모든 변수를 로그 수익률(Log Return)로 변환
df_exog_w['BTC_Return'] = np.log(df_exog_w['BTC_Price'] / df_exog_w['BTC_Price'].shift(1))
df_exog_w['FX_Return'] = np.log(df_exog_w['FX_Rate'] / df_exog_w['FX_Rate'].shift(1))
df_exog_w['NVDA_Return'] = np.log(df_exog_w['NVDA_Price'] / df_exog_w['NVDA_Price'].shift(1))

# 하드웨어 수익률 데이터와 외부 변수 병합
df_model = pd.merge(df_long, df_exog_w[['date', 'BTC_Return', 'FX_Return', 'NVDA_Return']], on='date', how='inner')
df_model = df_model.dropna()

# =========================================================
# 3. [1단계] 탐색적 분석 - One-Way ANOVA
# =========================================================
print("\n========== [1단계] 부품군별 가격 변동성(민감도) 차이 ANOVA ==========")
# 가격 변동의 '크기 자체(민감도)'를 보기 위해 로그 수익률의 절댓값을 취함
df_model['Abs_Return'] = df_model['Log_Return'].abs()

gpu_abs = df_model[df_model['Category'] == 'GPU']['Abs_Return']
cpu_abs = df_model[df_model['Category'] == 'CPU']['Abs_Return']
ram_abs = df_model[df_model['Category'] == 'RAM']['Abs_Return']

f_stat, p_val = stats.f_oneway(gpu_abs, cpu_abs, ram_abs)
print(f"F-통계량: {f_stat:.4f}, p-value: {p_val:.4e}")
if p_val < 0.05:
    print("결과: 부품군 간 절대적인 변동폭(민감도)의 크기에 유의미한 차이가 존재합니다.\n")
else:
    print("결과: 부품군 간 변동폭 크기에 유의미한 차이가 없습니다.\n")

# =========================================================
# 4. [2단계] 본 검정 - 통제 변수를 포함한 선형 혼합효과모형(LMM)
# =========================================================
print("========== [2단계] LMM: 비트코인 변동이 부품군별 가격에 미치는 영향 ==========")
# [핵심] 환율(FX_Return)과 엔비디아(NVDA_Return)를 통제 변수로 추가하여 가설 1과 방법론 통일
formula = "Log_Return ~ BTC_Return * C(Category, Treatment('CPU')) + FX_Return + NVDA_Return"

# 주차(date)를 Random Effect Group으로 설정하여 시계열 상의 주간 거시적 충격 흡수
mixed_model = smf.mixedlm(formula, data=df_model, groups=df_model['date']).fit()

print(mixed_model.summary())
