# Phân tích Giả thuyết H1 Mới: Báo cáo Toàn diện

## 1. Mục tiêu và Giả thuyết Nghiên cứu
Báo cáo này đánh giá một cách chặt chẽ **Giả thuyết H1 mới (New H1)** được đề xuất bởi nhóm nghiên cứu:
> *Liệu các Chỉ số Tài chính và Vĩ mô (ETF Bán dẫn, Chỉ số Niềm tin Tiêu dùng, Giá Dầu Quốc tế và Bitcoin) có tạo ra sự khác biệt về mức độ ảnh hưởng lên giá của từng loại linh kiện (GPU, CPU và RAM) hay không (không xét độ trễ thời gian - time lag)?*

Mục tiêu chính là đo lường mức độ tác động đồng thời (contemporaneous impact) của 4 yếu tố vĩ mô này đối với 3 linh kiện phần cứng, đồng thời kiểm định chính thức xem mức độ nhạy cảm của các linh kiện này có sự khác biệt có ý nghĩa thống kê hay không.

---

## 2. Chuẩn bị và Đồng bộ hóa Dữ liệu

### 2.1 Dữ liệu Linh kiện Phần cứng
- **Nguồn dữ liệu**: `idx_all_w.csv` (Chỉ số giá Jevons ghép chuỗi - Chain Matched Jevons Index).
- **Xử lý dữ liệu**: Chúng tôi đã tải các chỉ số giá hàng tuần của GPU, CPU và RAM. Để đồng bộ hóa với dữ liệu kinh tế vĩ mô, các chỉ số phần cứng này đã được hạ tần suất lấy mẫu (downsampled) xuống **tần suất hàng tháng** (lấy giá trị quan sát cuối cùng của mỗi tháng). Sau đó, tỷ suất sinh lợi logarit (log returns) được tính toán để đảm bảo tính dừng (stationarity) của chuỗi số liệu và phản ánh độ co giãn của giá cả qua từng tháng.

### 2.2 Dữ liệu Kinh tế Vĩ mô
Chúng tôi đã tải xuống dữ liệu kinh tế vĩ mô bên ngoài tương ứng với khung thời gian nghiên cứu chính xác (Tháng 3/2024 - Tháng 4/2026):
1. **ETF Ngành Bán dẫn (`SOXX`)**: Đại diện cho xu hướng và kỳ vọng chung của ngành công nghiệp bán dẫn toàn cầu. Dữ liệu được lấy thông qua thư viện `yfinance`.
2. **Bitcoin (`BTC-KRW`)**: Đại diện cho tâm lý thị trường tiền mã hóa và dòng vốn chảy vào các tài sản rủi ro (risk-on assets). Dữ liệu được lấy qua thư viện `yfinance`.
3. **Giá Dầu Quốc tế (`BZ=F` - Dầu thô Brent)**: Đại diện cho chi phí logistics, vận chuyển và các biến động kinh tế vĩ mô toàn cầu. Dữ liệu được lấy qua thư viện `yfinance`.
4. **Chỉ số Niềm tin Tiêu dùng (`UMCSENT`)**: Chỉ số Niềm tin Tiêu dùng của Đại học Michigan (University of Michigan Consumer Sentiment Index) được tải qua FRED API (sử dụng thư viện `pandas_datareader`), đóng vai trò là đại diện cho tâm lý tiêu dùng toàn cầu theo tần suất tháng.

### 2.4 Xác nhận Khung Thời gian Nghiên cứu (Time Period Verification)
> [!NOTE]
> **Đồng bộ hóa thời gian chính xác:**
> Khung thời gian của nghiên cứu được đồng bộ hóa chặt chẽ từ **Tháng 3/2024 đến Tháng 3/2026** (lưu ý rằng tuần đầu tiên của Tháng 4/2026 trong bộ dữ liệu gốc `idx_all_w.csv` là ngày `2026-04-05` hoàn toàn bị khuyết giá trị - NaN, do đó điểm kết thúc thực tế của chuỗi dữ liệu sạch là **31/03/2026**). 
> 
> Việc tính toán tỷ suất sinh lợi logarit hàng tháng khiến điểm dữ liệu đầu tiên (Tháng 3/2024) bị mất đi do sai phân log. Do đó, các mô hình hồi quy OLS ở cả Giai đoạn 3 và 4 được ước lượng trên chuỗi dữ liệu liên tục từ **30/04/2024 đến 31/03/2026** (gồm 24 quan sát tháng đầy đủ cho mỗi linh kiện). Điều này đảm bảo tính toàn vẹn kinh tế lượng và phản ánh đúng thực tế dữ liệu.

---

## 3. Mô hình hóa và Kết quả Thực nghiệm

### Giai đoạn 3: Mô hình Hồi quy Riêng biệt cho Từng Linh kiện
Chúng tôi đã chạy 3 mô hình OLS riêng biệt cho log returns của GPU, CPU và RAM dựa trên 4 yếu tố vĩ mô.

#### Kết quả Chính (Hệ số Chuẩn hóa & Hệ số xác định từng phần Partial $R^2$):
* **GPU**: 
  * Bitcoin (`BTC_ret`) có khả năng giải thích mạnh nhất (**Partial $R^2$ = 18.06%**), với hệ số hồi quy là -0.011 (p-value = 0.054). Kết quả này tiệm cận mức ý nghĩa thống kê 5%, cho thấy mối quan hệ đồng thời ngược chiều ở mức vừa phải trong giai đoạn lấy mẫu.
  * Các yếu tố ETF, CSI và Giá Dầu có khả năng giải thích không đáng kể (Partial $R^2$ < 2%).
* **CPU**:
  * ETF Bán dẫn (`SOXX_ret`) là yếu tố mạnh nhất (**Partial $R^2$ = 6.47%**), mặc dù không có ý nghĩa thống kê (p = 0.26).
* **RAM**:
  * Bitcoin và SOXX lần lượt giải thích khoảng 8.7% và 2.8% phương sai giá trị, nhưng cả hai hệ số đều không có ý nghĩa thống kê.

**Kết luận trung gian**: Dưới góc nhìn trực quan và trực giác toán học, Bitcoin chiếm ưu thế lớn trong việc giải thích biến động của GPU, trong khi ETF Bán dẫn có mối liên kết tương đối mạnh hơn đối với CPU.

### Giai đoạn 4: Mô hình Tương tác Pooled (Gộp)
Để kiểm định chính thức xem sự khác biệt về tác động này có thực sự vững chắc về mặt thống kê hay không, chúng tôi gộp tất cả dữ liệu linh kiện vào một tập dữ liệu định dạng dọc (Long format, gồm 72 quan sát) và chạy Mô hình Tác động Hỗn hợp / Mô hình Tương tác Tuyến tính (với CPU là nhóm cơ sở - baseline):
$$\text{Tỷ suất biến động giá} = (\text{Các yếu tố vĩ mô}) \times \text{Linh kiện}$$

#### Kết quả Kiểm định Khác biệt:
* **Tương tác BTC x GPU**: Hệ số là -0.0094 với **p-value = 0.560**.
* **Tương tác SOXX x GPU**: Hệ số là -0.0012 với **p-value = 0.947**.
* **Tương tác CSI và Giá Dầu**: Tất cả các giá trị p-value đều lớn hơn 0.60.

> [!WARNING] 
> **Kết luận Cuối cùng về Giả thuyết H1**
> Các biến tương tác hoàn toàn thất bại trong việc bác bỏ giả thuyết không ($H_0$). Điều này có nghĩa là mặc dù Giai đoạn 3 gợi ý rằng Bitcoin giải thích nhiều biến động của GPU hơn CPU, nhưng quy mô mẫu và mức độ nhiễu của dữ liệu thực tế đã ngăn cản chúng ta khẳng định một cách có ý nghĩa thống kê rằng **"các linh kiện chịu ảnh hưởng khác nhau bởi các yếu tố vĩ mô này"**. Các yếu tố vĩ mô tác động tương đối đối xứng lên các linh kiện, hoặc sự biến động giá phần lớn được thúc đẩy bởi các cú sốc đặc thù không quan sát được của từng linh kiện (component-specific shocks) hơn là từ 4 yếu tố vĩ mô này.

---

## 4. Sản phẩm Bàn giao và Vận hành
Toàn bộ quy trình phân tích đã được tự động hóa hoàn toàn và kết quả đầu ra được tổ chức ngăn nắp trên Desktop của bạn tại thư mục: `C:\Users\PC\Desktop\IEstat_New_H1`.

### Cấu trúc Thư mục Đã Tạo:
1. **`/data/`**: 
   - `processed_h1_dataset.csv`: Tập dữ liệu cuối cùng, được căn chỉnh thời gian và chuẩn hóa, sẵn sàng để tích hợp vào dashboard hoặc các mô hình học máy tiếp theo.
2. **`/results/`**: 
   - `phase3_coefficients_and_r2.csv`: Bảng tổng hợp sạch sẽ các hệ số beta, giá trị p-value và các giá trị Partial $R^2$.
   - `GPU_regression_summary.txt` (tương tự với CPU/RAM): Đầu ra OLS nguyên bản từ thư viện Statsmodels.
   - `phase4_interaction_summary.txt`: Đầu ra kiểm định mô hình tương tác chính thức.
3. **`/plots/`**: 
   - `01_correlation_heatmap.png`: Biểu đồ nhiệt thể hiện tương quan tuyến tính giữa tất cả các biến số.
   - `02_standardized_coefficients.png`: Biểu đồ cột so sánh độ nhạy cảm (Standardized Beta) của từng linh kiện đối với từng yếu tố.
   - `03_partial_rsquared.png`: Biểu đồ làm nổi bật ưu thế giải thích của Bitcoin đối với biến động giá GPU.
   - `04_timeseries_comparison.png` **[MỚI]**: Biểu đồ so sánh chuỗi thời gian (đa trục y) thể hiện sự đồng biến động giữa tỷ suất sinh lợi của từng linh kiện với nhân tố vĩ mô nổi bật nhất của nó.
   - `05_regression_scatters.png` **[MỚI]**: Biểu đồ phân tán đi kèm đường hồi quy OLS thực tế, trực quan hóa độ dốc và độ phân tán của dữ liệu cho các mối quan hệ quan trọng (GPU vs BTC, CPU vs SOXX, RAM vs BTC).
   - `06_residual_diagnostics.png` **[MỚI]**: Biểu đồ kiểm định phần dư (phân phối tần suất và Q-Q plot) của mô hình GPU nhằm kiểm tra tính chuẩn xác của các giả định kinh tế lượng (chứng minh tính phân phối chuẩn của phần dư).
   - `07_all_relationships_scatters.png` **[MỚI]**: Ma trận biểu đồ phân tán toàn diện (gồm 12 đồ thị: 3 linh kiện x 4 yếu tố vĩ mô) hiển thị mọi cặp quan hệ hồi quy kèm theo Hệ số Beta, giá trị p-value và Partial $R^2$ tương ứng cho từng cặp.

### Chi tiết Vận hành
- Toàn bộ quy trình được thực hiện thông qua một tập lệnh Python thống nhất (`run_h1_analysis.py`).
- Tập lệnh tự động xử lý giới hạn tần suất gọi API (rate limits) của `yfinance`, đồng bộ hóa dữ liệu từ FRED API, thực hiện phép nối trong (inner join) nghiêm ngặt trên các ngày trùng khớp để tránh rò rỉ dữ liệu (data leakage), và áp dụng hiệu chỉnh bậc tự do (degrees-of-freedom adjustments) chính xác khi tính toán Partial $R^2$.

