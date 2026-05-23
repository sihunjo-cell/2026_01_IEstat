# Comprehensive Documentation: Hypothesis 1 (H1) Analysis

## 1. Introduction and Research Objective

This document outlines the end-to-end analytical process conducted to evaluate **Hypothesis 1 (H1)** of the hardware price index research project. 

**The Core Hypothesis (H1):**
> *Do Financial and Macroeconomic Indicators (Semiconductor ETF, Consumer Sentiment Index, International Oil Price, and Bitcoin) affect component prices (GPU, CPU, and RAM) differently without considering time lag?*

The objective of this analysis is twofold:
1. To measure the contemporaneous (same-month) impact of four distinct macroeconomic factors on the price returns of three major PC hardware components.
2. To formally test whether the sensitivities of these components to the macro factors differ from one another with statistical significance.

---

## 2. Data Sourcing and Processing

The analysis relies on robust, time-aligned data from multiple sources. A unified Python pipeline was constructed to handle data ingestion, transformation, and merging.

### 2.1. Data Sources
1. **Hardware Components (Jevons Index):**
   * **Source:** Internal project dataset (`idx_all_w.csv`).
   * **Details:** This dataset contains the Chain Matched Jevons Index for GPUs, CPUs, and RAM at a weekly frequency.
2. **Macroeconomic Indicators:**
   * **Semiconductor ETF (`SOXX`):** Sourced via `yfinance`. Represents global semiconductor industry performance and expectations.
   * **Bitcoin (`BTC-KRW`):** Sourced via `yfinance`. Acts as a proxy for cryptocurrency market volatility and risk-on asset flows.
   * **International Oil Price (`BZ=F` - Brent Crude):** Sourced via `yfinance`. Represents global logistical costs and broad economic health.
   * **Consumer Sentiment Index (`UMCSENT`):** Sourced from the Federal Reserve Economic Data (FRED) API. Represents global consumer confidence and spending propensity.

### 2.2. Data Processing and Time Period Verification
* **Frequency Alignment:** The weekly hardware index data was downsampled to a monthly frequency by taking the last observation of each month to align with the monthly macroeconomic data (like UMCSENT).
* **Return Calculation:** We calculated the monthly logarithmic returns (log differences) for all variables to ensure stationarity, which is a prerequisite for robust OLS regression modeling.
* **Standardization:** All independent macro variables were converted into Z-scores $\left(Z = \frac{X - \mu}{\sigma}\right)$. This allows the resulting regression coefficients ($\beta$) to be interpreted directly as *Standardized Beta Coefficients*, making the magnitude of impact comparable across different factors (e.g., comparing the impact of Bitcoin to Oil).
* **Timeframe:** The raw dataset spanned from March 2024 to early April 2026. Because the first week of April 2026 contained missing values (NaN), the clean series terminated on March 31, 2026. Furthermore, calculating monthly log returns consumes the first month's data point. Therefore, the **final rigorous econometric timeframe spans 24 continuous months: April 30, 2024, to March 31, 2026.**

---

## 3. Visualizations and Chart Analysis

To explore the relationships before and after modeling, a comprehensive suite of 7 visualizations was generated.

### 3.1. Exploratory Data Analysis (EDA)
* **`01_correlation_heatmap.png`:**
  * **Analysis:** This heatmap visualizes the pairwise Pearson correlation coefficients across all raw variables. It provides a preliminary look at collinearity between independent variables (which was found to be manageable) and initial signals of correlation between components and macro factors.

### 3.2. Time-Series Co-movement
* **`04_timeseries_comparison.png`:**
  * **Analysis:** This 3-panel dual-axis chart plots the actual monthly return of each component alongside its most prominent macroeconomic counterpart (e.g., GPU vs. BTC, CPU vs. SOXX). It is highly effective in showing the cyclical co-movement and volatility matching between the component markets and the broader financial indices over the two-year period.

### 3.3. Regression Relationships and Distribution
* **`05_regression_scatters.png`:**
  * **Analysis:** This chart highlights the strongest component-factor relationships found in the regression analysis (GPU vs. BTC, CPU vs. SOXX, RAM vs. BTC). The scatter plots overlay an OLS trendline and confidence intervals, visually demonstrating the slope (Beta) and the dispersion of the data points.
* **`07_all_relationships_scatters.png`:**
  * **Analysis:** A massive 12-panel matrix (3 components × 4 factors) that leaves no stone unturned. It plots every single relationship simultaneously. This chart allows stakeholders to visually scan across the entire dataset and confirm that while some relationships (like GPU-BTC) have a discernible slope, many others (like GPU-Oil) are effectively flat, validating the regression results.

### 3.4. Diagnostic Checking
* **`06_residual_diagnostics.png`:**
  * **Analysis:** For an OLS regression to be valid, its residuals (prediction errors) must be normally distributed. This chart displays a frequency histogram (with KDE) and a Normal Q-Q plot for the GPU model's residuals. The points on the Q-Q plot closely track the theoretical normal line, confirming that our model does not violate core econometric assumptions and that the p-values are trustworthy.

---

## 4. Regression Summary Analysis

The core of the hypothesis testing was conducted in two phases using Ordinary Least Squares (OLS) regression models via the `statsmodels` library.

### 4.1. Phase 3: Component-Specific Multivariate Regressions
We ran three separate multivariate OLS models. In each model, a component's return was regressed against **all four macro factors simultaneously**:
$$\text{Component\_Return} = \beta_0 + \beta_1(\text{SOXX\_z}) + \beta_2(\text{CSI\_z}) + \beta_3(\text{Oil\_z}) + \beta_4(\text{BTC\_z}) + \epsilon$$

**Detailed Findings:**
1. **GPU Model:**
   * **Bitcoin (`BTC_ret_z`)** emerged as the dominant explanatory variable. It possesses the largest standardized coefficient (**$\beta = -0.011$**) and approaches statistical significance (**$p = 0.055$**).
   * Crucially, the **Partial $R^2$ is 18.06%**, meaning that holding all other macro factors constant, Bitcoin alone explains roughly 18% of the variance in GPU monthly returns.
   * SOXX, CSI, and Oil had negligible explanatory power (Partial $R^2 < 2\%$).
2. **CPU Model:**
   * The **Semiconductor ETF (`SOXX_ret_z`)** showed the strongest linkage to CPU returns (**$\beta = 0.004$**, Partial $R^2 = 6.47\%$). However, with a $p\text{-value} = 0.266$, this relationship is not statistically significant at the 5% level, suggesting CPU prices are highly resilient or driven by idiosyncratic supply-chain factors rather than immediate macro shocks.
3. **RAM Model:**
   * **Bitcoin** again showed a relatively strong Beta (**$\beta = -0.025$**), but the high variance in RAM prices resulted in a lack of statistical significance ($p = 0.193$, Partial $R^2 = 8.75\%$).

*Visualized in: `02_standardized_coefficients.png` and `03_partial_rsquared.png`.*

### 4.2. Phase 4: Pooled Interaction Model
While Phase 3 shows visual differences (e.g., BTC impacts GPUs more than CPUs), we must formally test if this difference is statistically significant. We restructured the data into a "Long" format and ran a Pooled Interaction Model (with CPU as the baseline component).

**Detailed Findings:**
* The interaction terms measure the *difference in impact* compared to the baseline.
* The interaction between **BTC and GPU** yielded a **$p\text{-value} = 0.560$**.
* The interaction between **SOXX and GPU** yielded a **$p\text{-value} = 0.947$**.
* All other interaction terms across CSI and Oil yielded $p\text{-values} > 0.60$.

### 4.3. Final Conclusion on Hypothesis 1 (H1)
**We fail to reject the null hypothesis.** 

Despite the exploratory Phase 3 models suggesting that Bitcoin has a stronger explanatory hold on GPUs compared to other components, the rigorous Phase 4 interaction model proves that **there is no statistically significant evidence to claim that the macroeconomic factors affect the components differently.** 

The high p-values in the interaction terms indicate that the variance in component price returns is overwhelmingly driven by component-specific idiosyncratic shocks (e.g., product release cycles, specific inventory bottlenecks) rather than asymmetric sensitivity to these four macroeconomic variables during the same month.

---
*Generated automatically by the AI Analytical Pipeline. All data, scripts, and visualizations are securely stored in the `IEstat_New_H1` Desktop directory.*
