# HƯỚNG DẪN SIÊU CHI TIẾT: Hệ thống Dự đoán Giá Chứng khoán bằng AI

> Tài liệu này được viết cho người **KHÔNG CÓ nền tảng kinh tế/tài chính**.
> Mọi thuật ngữ đều được giải thích từ đầu.

---

# PHẦN 0: KIẾN THỨC NỀN — CHỨNG KHOÁN LÀ GÌ?

## 0.1. Cổ phiếu là gì?

Hãy tưởng tượng **công ty FPT** giống như một cái bánh pizza lớn. Cái bánh này được cắt thành **hàng triệu miếng nhỏ**. Mỗi miếng nhỏ gọi là **1 cổ phiếu (stock/share)**. Khi bạn mua 1 cổ phiếu FPT, bạn sở hữu một phần rất nhỏ của công ty FPT.

## 0.2. Sàn chứng khoán là gì?

Sàn chứng khoán (ví dụ: **HOSE** — Sở Giao dịch Chứng khoán TP.HCM) giống như một **chợ online** nơi người ta mua bán cổ phiếu. Giống Shopee nhưng thay vì bán quần áo thì bán "miếng bánh pizza" của các công ty.

**VN30** là danh sách **30 công ty lớn nhất** trên sàn HOSE, bao gồm FPT, Vinamilk (VNM), Vietcombank (VCB), Hòa Phát (HPG), v.v.

## 0.3. Giá cổ phiếu thay đổi như thế nào?

Mỗi ngày (thứ 2 đến thứ 6), giá cổ phiếu thay đổi liên tục dựa trên **cung và cầu**:
- Nhiều người muốn **MUA** → giá **TĂNG** (khan hiếm)
- Nhiều người muốn **BÁN** → giá **GIẢM** (dư thừa)

Cuối mỗi ngày, hệ thống ghi lại **5 con số quan trọng**:

| Thuật ngữ | Tiếng Anh | Ý nghĩa | Ví dụ |
|-----------|-----------|---------|-------|
| **Giá mở cửa** | Open | Giá giao dịch đầu tiên trong ngày | 74,200đ |
| **Giá cao nhất** | High | Giá cao nhất trong ngày | 75,100đ |
| **Giá thấp nhất** | Low | Giá thấp nhất trong ngày | 73,800đ |
| **Giá đóng cửa** | Close | Giá giao dịch cuối cùng trong ngày | 74,600đ |
| **Khối lượng** | Volume | Tổng số cổ phiếu được mua bán | 5,200,000 |

**Giá đóng cửa (Close)** là con số quan trọng nhất — nó đại diện cho giá trị cổ phiếu vào cuối ngày.

## 0.4. Nhà đầu tư kiếm tiền thế nào?

Rất đơn giản: **Mua rẻ, Bán đắt.**

```
Ngày 1: Mua FPT giá 74,600đ × 100 cổ phiếu = 7,460,000đ
Ngày 5: Bán FPT giá 78,000đ × 100 cổ phiếu = 7,800,000đ
→ Lời: 340,000đ (trừ phí giao dịch khoảng 0.2% ≈ 15,000đ)
→ Lời thực: 325,000đ
```

**Vấn đề:** Làm sao biết "ngày mai giá sẽ tăng hay giảm"? → **Đây chính là bài toán mà đồ án này giải quyết.**

## 0.5. Tại sao dự đoán giá cổ phiếu CỰC KỲ KHÓ?

Giá cổ phiếu bị ảnh hưởng bởi hàng trăm yếu tố:
- Kết quả kinh doanh công ty
- Tin tức chính trị (chiến tranh, bầu cử)
- Lãi suất ngân hàng
- Tâm lý đám đông (hoảng loạn bán tháo hoặc hưng phấn mua vào)
- Thiên tai, dịch bệnh...

**Không có mô hình nào dự đoán chính xác 100%.** Mục tiêu thực tế là: **Đúng hơn 50%** (tốt hơn tung đồng xu) và đủ để kiếm lời sau khi trừ phí.

---

# PHẦN 1: BÀI TOÁN CỦA ĐỒ ÁN

## 1.1. Đồ án này làm gì?

Xây dựng một **hệ thống AI** có khả năng:
1. **Dự đoán giá đóng cửa ngày mai** (gọi là T+1) cho 4 mã: FPT, VCB, HPG, VNM
2. **Đưa ra lời khuyên**: MUA / BÁN / GIỮ
3. **Hiển thị trên giao diện web** đẹp, dễ đọc
4. **Cung cấp API** để các ứng dụng khác có thể gọi

## 1.2. Ý tưởng cốt lõi: Kết hợp 2 chuyên gia AI

Thay vì dùng 1 mô hình AI, đồ án dùng **2 mô hình hoàn toàn khác nhau** rồi để một "trưởng phòng" tổng hợp ý kiến:

**Ví dụ thực tế dễ hiểu:**
Bạn muốn đoán ngày mai trời có mưa không. Bạn hỏi 2 người:
- **Ông A (TFT — Temporal Fusion Transformer):** Là nhà khí tượng chuyên nghiệp. Ông ấy xem bản đồ thời tiết **2 tháng qua**, phân tích mây, gió, áp suất. Ông ấy giỏi nhìn xu hướng lớn ("tuần sau sẽ mưa nhiều") nhưng đôi khi sai về chi tiết ("mưa lúc 3h" mà thực tế mưa lúc 7h).
- **Ông B (LightGBM):** Là nông dân lâu năm. Ông ấy chỉ nhìn trời **sáng nay** — thấy chuồn chuồn bay thấp, kiến bò lên cao, nên kết luận "chiều nay mưa". Ông ấy phản ứng nhanh nhưng thiếu tầm nhìn xa.
- **Trưởng phòng (Meta-Learner):** Nghe cả ông A và ông B, biết ai thường đúng hơn, rồi đưa kết luận cuối cùng.

---

# PHẦN 2: DỮ LIỆU — AI HỌC TỪ ĐÂU?

## 2.1. Lấy dữ liệu từ đâu?

**File:** `yahoo_data.py`

AI học từ **lịch sử giá cổ phiếu** — giống như bạn ôn thi bằng cách làm đề cũ. Hệ thống lấy dữ liệu từ **Yahoo Finance** (miễn phí, đáng tin cậy) gồm 5 cột OHLCV cho mỗi ngày giao dịch:

```
Ngày        | Mở cửa  | Cao nhất | Thấp nhất | Đóng cửa | Khối lượng
2026-03-20  | 74,200  | 75,100   | 73,800    | 74,600   | 5,200,000
2026-03-19  | 73,500  | 74,500   | 73,200    | 74,200   | 4,800,000
2026-03-18  | 73,000  | 73,800   | 72,500    | 73,500   | 6,100,000
... (kéo dài 1000 ngày ≈ 4 năm)
```

## 2.2. Tại sao dữ liệu thô chưa đủ?

Chỉ có 5 con số (mở, cao, thấp, đóng, khối lượng) giống như bạn chỉ cho AI xem **1 bức ảnh chụp** — nó thiếu ngữ cảnh. AI cần thêm thông tin để hiểu:
- Giá đang **tăng hay giảm** so với tuần trước?
- Thị trường đang **quá nóng** (mua quá nhiều) hay **quá lạnh** (bán quá nhiều)?
- Giá đang ở gần **trần hay sàn** của khoảng dao động bình thường?

## 2.3. Tám "ống kính" bổ sung (Technical Indicators)

**File:** `indicators.py`

Hệ thống tính thêm 8 con số bổ sung từ dữ liệu gốc, mỗi con số cho AI một "góc nhìn" khác:

### a) SMA — Đường trung bình (2 cái: SMA_10 và SMA_20)
**Ý nghĩa đời thường:** Tính điểm trung bình của bạn trong 10 bài kiểm tra gần nhất.
```
Giá 10 ngày: 73, 74, 72, 75, 76, 74, 73, 75, 76, 74
SMA_10 = Tổng / 10 = 74.2 (nghìn đồng)
```
- Giá hôm nay (74.6) **cao hơn** SMA_10 (74.2) → Giá đang ở trên mức trung bình → Xu hướng tốt
- SMA_10 > SMA_20 → Xu hướng ngắn hạn mạnh hơn trung hạn → Đang tăng

### b) RSI — Đo "nhiệt độ" thị trường
**Ý nghĩa đời thường:** Giống nhiệt kế đo xem cổ phiếu đang "sốt" (quá nóng) hay "lạnh" (đang rẻ).
```
Trong 14 ngày: tăng 9 ngày, giảm 5 ngày
→ Lực mua mạnh hơn lực bán
→ RSI = 62.5 (thang 0-100)
```
- RSI > 70: **"Sốt cao"** — Mọi người đã mua quá nhiều, giá có thể sắp giảm (giống bong bóng)
- RSI < 30: **"Hạ thân nhiệt"** — Mọi người bán tháo quá, giá có thể sắp tăng lại
- RSI ≈ 50: Bình thường

### c) MACD — Phát hiện "đảo chiều"
**Ý nghĩa đời thường:** So sánh tốc độ chạy ngắn (sprint 12 ngày) và tốc độ chạy dài (marathon 26 ngày).
- Nếu sprint **nhanh hơn** marathon → Đang tăng tốc → MACD dương
- Nếu sprint **chậm hơn** marathon → Đang giảm tốc → MACD âm
- Thời điểm MACD **đổi dấu** = thời điểm xu hướng thay đổi

### d) Bollinger Bands — "Đường ray" của giá
**Ý nghĩa đời thường:** Vẽ 2 đường biên trên và dưới, giá thường nằm bên trong. Giống đường ray xe lửa — nếu tàu chệch ra ngoài, nó sẽ bị kéo trở lại.
- Giá chạm **biên trên** → Có thể giảm lại
- Giá chạm **biên dưới** → Có thể tăng lại

### e) Log Return — Tốc độ thay đổi giá
**Ý nghĩa đời thường:** Hôm nay giá thay đổi bao nhiêu % so với hôm qua. Giá tăng 1% ghi là +0.01, giảm 2% ghi là -0.02.

**→ Kết quả cuối:** Mỗi ngày được mô tả bằng **13 con số** (5 gốc + 8 chỉ báo), thay vì chỉ 5.

## 2.4. Biến mục tiêu — AI cần đoán cái gì?

```python
df['target'] = df['close'].shift(-1)  # Giá đóng cửa NGÀY MAI
```

Hiểu đơn giản: Mỗi dòng dữ liệu là một "câu hỏi thi" kèm "đáp án":
```
Câu hỏi: Hôm nay FPT đóng cửa 74,600đ, RSI=62, MACD=+0.3...  
Đáp án:  Ngày mai FPT đóng cửa 75,200đ

AI cần học: Nhìn 13 con số hôm nay → đoán ra con số ngày mai
```

## 2.5. Chuẩn hóa — Tại sao BẮT BUỘC phải làm?

**Vấn đề:** Các con số có khoảng giá trị rất khác nhau:
- Khối lượng giao dịch: **5,000,000** (hàng triệu)
- Giá cổ phiếu: **74,600** (hàng chục nghìn)
- RSI: **62** (hàng chục)
- Log Return: **0.005** (rất nhỏ)

Nếu đưa thẳng vào AI, nó sẽ nghĩ "Volume quan trọng nhất" (vì số to nhất) → **SAI hoàn toàn.**

**Giải pháp — StandardScaler:** Đưa tất cả về cùng khoảng (trung bình = 0, dao động ≈ -3 đến +3):
```
Volume 5,000,000   → sau chuẩn hóa: -0.12
Giá    74,600      → sau chuẩn hóa:  0.35
RSI    62          → sau chuẩn hóa:  0.78
Log Return 0.005   → sau chuẩn hóa:  0.45
```

Giờ AI thấy tất cả ở cùng "thang đo" → học công bằng, không thiên vị.

## 2.6. Chia dữ liệu — Quy tắc vàng KHÔNG ĐƯỢC VI PHẠM

```
1000 ngày dữ liệu:
|<-------- 800 ngày TRAIN -------->|<---- 200 ngày VALIDATION ---->|
   AI được HỌC từ đây                 AI bị KIỂM TRA ở đây
   (giống ôn bài)                     (giống thi thật)
```

**Tại sao phải chia theo THỜI GIAN (không phải ngẫu nhiên)?**

Giống việc ôn thi:
- **Cách ĐÚNG:** Ôn bài từ chương 1-8 (quá khứ), rồi thi chương 9-10 (tương lai) → Biết thực lực thật
- **Cách SAI:** Ôn bài lẫn lộn cả chương 9-10 vào → Khi thi "quen đề" → Điểm cao ảo → Ra đời không biết gì

Trong chứng khoán, nếu AI "nhìn trước" dữ liệu tương lai, nó sẽ cho kết quả đẹp trên giấy nhưng hoàn toàn vô giá trị khi dự đoán thật. Lỗi này gọi là **Data Leakage (rò rỉ dữ liệu tương lai)** — một trong những lỗi nghiêm trọng nhất trong Machine Learning.

---

# PHẦN 3: BỘ NÃO AI — 3 TẦNG XỬ LÝ

## 3.1. Sơ đồ tổng quát

```
    Dữ liệu 60 ngày × 13 features (đã chuẩn hóa)
                         │
               ┌─────────┴─────────┐
               ▼                   ▼
       ┌──────────────┐   ┌──────────────┐
       │     TFT      │   │   LightGBM   │
       │  (Chuyên gia │   │  (Chuyên gia │
       │   dài hạn)   │   │   ngắn hạn)  │
       └──────┬───────┘   └──────┬───────┘
              │                   │
         Dự đoán A           Dự đoán B
              │                   │
              └─────────┬─────────┘
                        ▼
               ┌──────────────┐
               │ Meta-Learner │
               │ (Trưởng ban) │
               └──────┬───────┘
                      ▼
             Dự đoán cuối cùng
                      │
                      ▼
               ┌──────────────┐
               │   Decision   │
               │    Policy    │
               │ (Bộ quy tắc │
               │  ra quyết    │
               │  định)       │
               └──────┬───────┘
                      ▼
               MUA / BÁN / GIỮ
```

## 3.2. Tầng 1: TFT (Temporal Fusion Transformer) — Chuyên gia dài hạn

**File:** `tft_model.py` | **Đầu vào:** 60 ngày × 13 features | **Đầu ra:** 1 con số (giá dự đoán)

TFT là mô hình Deep Learning (AI sâu) gồm 4 bộ phận, mỗi bộ phận giải quyết 1 vấn đề cụ thể:

### Bộ phận A: Variable Selection Network (VSN) — "Bộ lọc thông minh"

**Vấn đề cần giải:** Trong 13 con số, không phải lúc nào tất cả đều hữu ích. Có ngày RSI quan trọng, có ngày Volume quan trọng hơn.

**Cách hoạt động (ví dụ dễ hiểu):**
Giống như bạn đọc 13 tờ báo mỗi sáng, nhưng không có thời gian đọc kỹ hết. VSN giúp bạn:
1. Lướt qua 13 tờ → Gán điểm quan trọng (tổng = 100%)
   - Hôm nay: RSI = 25%, Volume = 20%, MACD = 15%, Close = 12%... (tổng 100%)
2. Đọc kỹ tờ quan trọng nhất, lướt nhanh tờ ít quan trọng
3. Tổng hợp lại thành 1 bản tóm tắt (vector 64 chiều)

**Điểm đặc biệt:** VSN thay đổi trọng số HẰNG NGÀY. Ngày thị trường bình thường thì Close quan trọng nhất. Ngày có tin sốc thì Volume vọt lên hàng đầu.

### Bộ phận B: Gated Residual Network (GRN) — "Cổng lọc bụi"

**Vấn đề cần giải:** Khi dữ liệu đi qua nhiều lớp xử lý, thông tin hữu ích có thể bị mất (giống photocopy nhiều lần thì mờ dần).

**Cách hoạt động (ví dụ dễ hiểu):**
Giống máy lọc nước có 2 đường:
- **Đường chính:** Nước đi qua bộ lọc (biến đổi phức tạp) → loại bỏ tạp chất
- **Đường tắt (skip connection):** Nước đi thẳng, không lọc → giữ nguyên khoáng chất tốt
- **Cổng (gate):** Quyết định lấy bao nhiêu % từ đường chính, bao nhiêu % từ đường tắt

```
Đầu vào → Biến đổi → Cổng (0 đến 1) → Kết quả
    └───────────────────────────────────→ Cộng lại (đường tắt)
```

Nếu cổng = 0: Giữ nguyên đầu vào (không đổi gì)
Nếu cổng = 1: Dùng hoàn toàn kết quả biến đổi
Thực tế: Cổng ở giữa (ví dụ 0.6) → Lấy 60% biến đổi + 40% gốc

### Bộ phận C: LSTM — "Bộ nhớ tuần tự"

**Vấn đề cần giải:** Cần nhớ chuyện xảy ra những ngày trước để hiểu ngày hôm nay.

**Cách hoạt động (ví dụ dễ hiểu):**
Giống như bạn đọc một cuốn tiểu thuyết 60 trang (60 ngày), LSTM đọc từng trang một:
- Trang 1: "FPT tăng mạnh" → nhớ lại: "Ồ, đang tăng"
- Trang 2: "FPT tiếp tục tăng" → nhớ lại: "Tăng 2 ngày liên tiếp"
- Trang 3: "FPT giảm nhẹ" → nhớ lại: "Tăng 2 ngày rồi giảm → có thể đang điều chỉnh"
- ...
- Trang 60: Tổng hợp ký ức 59 trang trước → hiểu bức tranh toàn cảnh

### Bộ phận D: Multi-Head Attention — "Nhìn tổng thể"

**Vấn đề cần giải:** LSTM đọc từng trang nên có thể quên trang đầu khi đọc đến trang 60. Attention cho phép nhìn TẤT CẢ 60 ngày cùng lúc.

**Cách hoạt động (ví dụ dễ hiểu):**
Giống bạn trải 60 bức ảnh chụp biểu đồ giá ra bàn. Thay vì xem từng ảnh, bạn nhìn toàn bộ cùng lúc và phát hiện:
- "Ngày 5 và ngày 55 có mẫu hình giống nhau!" (mối tương quan xa)
- "Mỗi 10 ngày lại có 1 đợt tăng mạnh!" (chu kỳ)

**Multi-Head (4 heads):** 4 người cùng nhìn 60 bức ảnh nhưng mỗi người tìm kiếm thứ khác nhau:
- Người 1: Tìm mẫu hình giá (price patterns)
- Người 2: Tìm mẫu hình khối lượng (volume spikes)
- Người 3: Tìm chu kỳ tuần (weekly seasonality)
- Người 4: Tìm sự tương đồng RSI

### Kết hợp 4 bộ phận:
```
13 features → VSN (chọn feature) → vector 64 chiều cho mỗi ngày
→ LSTM (đọc tuần tự 60 ngày) → chuỗi 60 vector
→ Attention (nhìn tổng thể) → chuỗi 60 vector đã "hiểu sâu"
→ Lấy vector ngày cuối → Linear → 1 con số = giá dự đoán T+1
```

## 3.3. Tầng 2: LightGBM — Chuyên gia ngắn hạn

**File:** `lgbm_model.py` | **Đầu vào:** 13 features (CHỈ ngày cuối) | **Đầu ra:** 1 con số

### LightGBM hoạt động như thế nào? (Cực kỳ khác TFT)

LightGBM xây hàng trăm **cây quyết định** — giống trò chơi "20 câu hỏi":
```
Câu hỏi 1: RSI > 65 không?
  Có → Câu hỏi 2a: MACD > 0 không?
        Có → "Giá sẽ tăng 200đ"
        Không → "Giá sẽ tăng 50đ"
  Không → Câu hỏi 2b: Volume > 5 triệu không?
        Có → "Giá sẽ giảm 100đ"  
        Không → "Giá sẽ tăng 30đ"
```

Nhưng 1 cây không đủ chính xác. LightGBM xây **200+ cây**, mỗi cây sửa lỗi của cây trước:
```
Cây 1: Đoán 74,000 (sai +600 so với thực tế 74,600)
Cây 2: Học từ sai 600 → sửa +400 → Tổng: 74,400 (sai +200)
Cây 3: Học từ sai 200 → sửa +150 → Tổng: 74,550 (sai +50)
... cứ thế cho đến khi sai số gần 0 hoặc bắt đầu tệ hơn (Early Stopping)
```

### Tại sao cần LightGBM khi đã có TFT?
| | TFT | LightGBM |
|---|---|---|
| Nhìn bao nhiêu ngày? | 60 ngày | 1 ngày |
| Giỏi gì? | Xu hướng dài hạn | Phản ứng tức thì |
| Yếu gì? | Chậm phản ứng đột ngột | Không hiểu ngữ cảnh |
| Ví dụ | "FPT đang trong trend tăng 3 tháng" | "Hôm nay RSI=85, quá mua, nên bán" |

→ Kết hợp cả hai = bù đắp điểm yếu cho nhau.

## 3.4. Tầng 3: Meta-Learner — Trưởng ban tổng hợp

**Bản chất:** Chỉ là một phương trình cộng trọng số:
```
Giá cuối = w₁ × (dự đoán TFT) + w₂ × (dự đoán LightGBM) + b
```

**Ví dụ thực tế:**
```
TFT dự đoán: 76,000đ
LightGBM dự đoán: 73,500đ
Meta-Learner (đã học w₁=0.6, w₂=0.4, b=200):
  → Kết quả = 0.6 × 76,000 + 0.4 × 73,500 + 200
             = 45,600 + 29,400 + 200
             = 75,200đ
```

**Tại sao Meta-Learner chỉ là phép cộng đơn giản?**
Vì đầu vào chỉ có **2 con số**. Nếu dùng mô hình phức tạp (như Neural Network) cho 2 con số, nó sẽ "thuộc bài" ngay lập tức (overfitting) mà không khái quát được gì.

---

# PHẦN 4: TỪ DỰ ĐOÁN ĐẾN QUYẾT ĐỊNH — DECISION POLICY

**File:** `decision_policy.py`

## 4.1. Tại sao không đơn giản "dự đoán tăng = mua"?

3 lý do thực tế:
1. **Chi phí giao dịch:** Mỗi lần mua/bán mất 0.15-0.25% phí. Nếu dự đoán chỉ tăng 0.1%, bạn sẽ **LỖ** sau khi trừ phí.
2. **AI không chắc chắn 100%:** Nếu TFT nói "tăng" nhưng LightGBM nói "giảm", bạn có nên tin không?
3. **Thị trường biến động tự nhiên:** Giá tăng/giảm 0.3% trong 1 ngày là bình thường, không phải tín hiệu gì.

## 4.2. Ví dụ tính toán đầy đủ cho FPT

```
Cho:
  Giá hiện tại: 74,600đ
  TFT dự đoán: 86,000đ  
  LightGBM dự đoán: 82,000đ
  Meta-Learner kết hợp: 84,149đ
  Biến động 20 ngày gần nhất: 1.2%/ngày
```

**Bước 1: Lợi nhuận gộp** (chưa trừ gì)
```
gross = (84,149 - 74,600) / 74,600 × 100% = +12.80%
→ Dự đoán giá tăng 12.80%
```

**Bước 2: Trừ chi phí giao dịch**
```
net = 12.80% - 0.2% (phí) = +12.60%
```

**Bước 3: Đo mức độ "bất đồng" giữa 2 chuyên gia**
```
uncertainty = |86,000 - 82,000| / 74,600 × 100% = 5.36%
→ TFT và LightGBM chênh nhau 4,000đ → khá bất đồng
```

**Bước 4: Phạt cho sự bất đồng**
```
penalty = 0.7 × 5.36% = 3.75%
edge = 12.60% - 3.75% = +8.85%
→ Sau khi tính tất cả, vẫn còn biên lợi nhuận 8.85%
```

**Bước 5: So với ngưỡng**
```
Ngưỡng = max(0.3%, 0.5 × 1.2%) = 0.6%  
(Thị trường biến động 1.2%/ngày → ngưỡng 0.6%)

8.85% > 0.6% → Biên lợi nhuận lớn hơn ngưỡng → MUA ✓
```

**Bước 6: Tính Confidence (Độ tự tin)**
```
agreement = 1 / (1 + 5.36) = 15.7%      ← 2 chuyên gia chưa thống nhất lắm
edge_score = min(8.85/0.6, 2) / 2 = 100% ← Nhưng biên lợi nhuận rất lớn
confidence = 55% × 15.7% + 45% × 100% = 53.6%
```

**→ Kết luận: MUA FPT, Confidence 53.6%, Expected Return 8.85%**

---

# PHẦN 5: TRAINING — AI HỌC NHƯ THẾ NÀO?

## 5.1. Toàn bộ quy trình (File `ensemble_trainer.py`)

```
python final_ensemble_train.py
```

Cho MỖI mã (ví dụ FPT), hệ thống thực hiện 10 bước:

| Bước | Hành động | File |
|------|-----------|------|
| 1 | Tải 1000 ngày dữ liệu FPT từ Yahoo | yahoo_data.py |
| 2 | Tính 8 chỉ báo kỹ thuật | indicators.py |
| 3 | Chia 800 ngày train + 200 ngày val | ensemble_trainer.py |
| 4 | Chuẩn hóa (fit CHỈ trên train) | ensemble_trainer.py |
| 5 | Train TFT (30 epochs, dừng sớm nếu tệ) | tft_model.py |
| 6 | Train LightGBM (200+ cây, dừng sớm) | lgbm_model.py |
| 7 | Cho cả 2 dự đoán trên validation | ensemble_trainer.py |
| 8 | Train Meta-Learner trên 2 dự đoán | ensemble_trainer.py |
| 9 | Đánh giá MAE, RMSE, Directional Acc | ensemble_trainer.py |
| 10 | Lưu 6 file vào ./models/ | ensemble_trainer.py |

## 5.2. Early Stopping — Dừng đúng lúc

**Vấn đề:** Nếu train quá lâu, AI sẽ "thuộc bài" (Overfitting):
```
Epoch 1:  Sai trên train: 500đ   Sai trên val: 480đ  ← Đang tiến bộ
Epoch 5:  Sai trên train: 300đ   Sai trên val: 320đ  ← Vẫn OK
Epoch 10: Sai trên train: 100đ   Sai trên val: 350đ  ← CHÚ Ý! Val tệ hơn!
Epoch 15: Sai trên train:  20đ   Sai trên val: 500đ  ← THẢM HỌA!
          Train giỏi nhưng val kém = "thuộc bài" nhưng không hiểu bản chất

→ Early Stopping quay lại Epoch 5 (tốt nhất trên val) và DỪNG luôn.
```

Giống sinh viên: Ôn 5 tiếng → hiểu bài. Ôn 15 tiếng → thuộc lòng đề cũ nhưng gặp đề mới là trượt.

## 5.3. Sáu file "não bộ" — Giải thích bằng ví dụ

| File | Chứa gì | Ví dụ thực tế |
|------|---------|--------------|
| `FPT_tft_model.pt` | Hàng triệu "trọng số" (weights) của mạng Neural | Giống **bộ nhớ dài hạn** của chuyên gia, lưu tất cả quy luật đã học |
| `FPT_lgbm_model.pkl` | 200+ cây quyết định đã xây xong | Giống **cuốn sổ tay** ghi 200 quy tắc: "Nếu RSI>70 VÀ Volume giảm → Giá giảm" |
| `FPT_meta_learner.pkl` | 2 con số (w₁, w₂) + 1 bias | Giống **công thức pha trà sữa**: "60% trà + 40% sữa + 1 muỗng đường" |
| `FPT_scaler_x.pkl` | 13 cặp (mean, std) | Giống **thước đo chuẩn**: "Volume trung bình là 5 triệu, std là 2 triệu" |
| `FPT_scaler_y.pkl` | 1 cặp (mean, std) | Giống **bảng quy đổi**: "Số 0.35 = giá 74,600đ" |
| `FPT_artifact_manifest.json` | Danh sách 5 file trên | Giống **checklist**: "Đủ 5 linh kiện → Sẵn sàng hoạt động" |

---

# PHẦN 6: ĐÁNH GIÁ — AI ĐOÁN ĐÚNG BAO NHIÊU?

## 6.1. MAE — Sai trung bình bao nhiêu tiền?
```
Ngày 1: Dự đoán 74,500, Thực tế 74,600 → Sai 100đ
Ngày 2: Dự đoán 75,200, Thực tế 74,800 → Sai 400đ  
Ngày 3: Dự đoán 73,900, Thực tế 74,000 → Sai 100đ
MAE = (100 + 400 + 100) / 3 = 200đ
→ "Trung bình mỗi ngày, AI đoán sai khoảng 200đ"
```

## 6.2. Directional Accuracy — Đoán đúng hướng bao nhiêu %?
```
Ngày 1: AI nói "TĂNG", Thực tế TĂNG → ✓ Đúng
Ngày 2: AI nói "TĂNG", Thực tế GIẢM → ✗ Sai
Ngày 3: AI nói "GIẢM", Thực tế GIẢM → ✓ Đúng
→ Accuracy = 2/3 = 66.7%
```
- **50%** = Tung đồng xu (vô dụng)
- **55%+** = Có giá trị thống kê (bắt đầu hữu ích)
- **60%+** = Rất tốt cho thị trường chứng khoán

**Đây là chỉ số quan trọng nhất.** Dù AI sai 500đ nhưng đoán đúng HƯỚNG (tăng/giảm), nhà đầu tư vẫn có thể kiếm lời.

---

# PHẦN 7: GIAO DIỆN HIỂN THỊ

## 7.1. Dashboard — Giao diện web
**File:** `dashboard.py` | **Chạy:** `python dashboard.py` | **Xem:** `http://localhost:8081`

Hiển thị 4 "thẻ" cho 4 mã cổ phiếu, mỗi thẻ gồm:
- Giá hiện tại
- Giá dự đoán T+1 (↑ tăng hoặc ↓ giảm)
- Quyết định: MUA / BÁN / GIỮ
- Expected Return (lợi nhuận kỳ vọng)
- Confidence (độ tự tin)

## 7.2. REST API — Cho lập trình viên
**File:** `app.py` | **Chạy:** `python app.py` | **Gọi:** `http://localhost:8080/predict/FPT`

Trả về dữ liệu JSON:
```json
{
  "symbol": "FPT",
  "current_price": 74600.0,
  "predicted_t1": 84149.02,
  "expected_return_pct": 4.3343,
  "confidence": 0.4929,
  "action": "BUY",
  "trend": "UP"
}
```

---

# PHẦN 8: TỔN TẮT TOÀN BỘ ĐỒ ÁN

## Một câu: 
**"Dùng 2 loại AI khác nhau (Deep Learning + Tree-based) để dự đoán giá cổ phiếu ngày mai, rồi hiển thị kết quả trên web."**

## Mười vấn đề đã giải quyết:

| # | Vấn đề | Giải pháp |
|---|--------|-----------|
| 1 | 1 mô hình không đủ chính xác | Kết hợp TFT + LightGBM bằng Stacking |
| 2 | AI "thuộc bài" (overfitting) | Early Stopping + Dropout |
| 3 | Dữ liệu thô thiếu thông tin | 8 chỉ báo kỹ thuật bổ sung |
| 4 | Rò rỉ dữ liệu tương lai | Chia theo thời gian, Scaler fit chỉ trên train |
| 5 | "Tăng = mua" quá đơn giản | Decision Policy tính phí, uncertainty |
| 6 | Không biết feature nào quan trọng | VSN tự động chọn |
| 7 | Thông tin mất qua nhiều lớp | GRN với skip connection |
| 8 | Cần nhớ xu hướng xa | Multi-Head Attention |
| 9 | 2 mô hình đóng góp khác nhau | Meta-Learner tự học trọng số |
| 10 | Kết quả khó đọc | Dashboard web + REST API |
