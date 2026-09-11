# Metrics đánh giá các task Understanding

Tài liệu này mô tả bộ metric dùng để đánh giá quá trình predict trên các mẫu có `task_family = "understanding"` trong `data_train/base_legal_7b_internal_test.jsonl`.

Tập Understanding gồm bốn task, mỗi task có 180 mẫu:

| Task | Nội dung đánh giá |
|---|---|
| `intent_understanding` | Nhận diện ý định của câu hỏi |
| `legal_concept_understanding` | Trích xuất và chuẩn hóa khái niệm pháp lý |
| `scope_user_facts_understanding` | Trích xuất phạm vi tìm kiếm, tình tiết người dùng và tiền đề pháp lý |
| `question_rewrite_compatibility` | Viết lại câu hỏi độc lập dựa trên ngữ cảnh hội thoại |

## 1. Quy ước chung

Trước khi tính metric:

- Parse prediction thành JSON.
- Chuẩn hóa Unicode, loại bỏ khoảng trắng thừa ở đầu/cuối và gộp các khoảng trắng liên tiếp.
- Không xét thứ tự phần tử trong `concepts`, `scope` và `user_facts`.
- Không tự sửa nội dung hoặc tự ánh xạ từ đồng nghĩa khi tính Exact Match.
- Prediction không parse được JSON được tính là sai đối với metric của task tương ứng.

### 1.1. Xử lý trường hợp rỗng - rỗng

Các metric Precision, Recall và F1 cho field dạng danh sách được tính theo **micro-average trên toàn bộ tập**: cộng `TP`, `FP`, `FN` của tất cả mẫu trước, sau đó mới tính metric.

```text
Micro Precision = tổng TP / (tổng TP + tổng FP)
Micro Recall    = tổng TP / (tổng TP + tổng FN)
Micro F1        = 2 * Micro Precision * Micro Recall
                  / (Micro Precision + Micro Recall)
```

Quy ước cho các trường hợp rỗng:

- `gold = []`, `prediction = []`: không đóng góp vào `TP`, `FP` hoặc `FN`, do đó không làm tăng Micro-F1.
- `gold = []`, `prediction != []`: mỗi phần tử dự đoán thừa được tính là `FP`.
- `gold != []`, `prediction = []`: mỗi phần tử bị bỏ sót được tính là `FN`.

Ngay cả khi tính theo micro-average, mẫu số vẫn có thể bằng `0` ở mức toàn tập. Hai trường hợp này phải xử lý riêng, không gộp chung:

**Trường hợp 1 — `TP + FN = 0` (ground truth không có positive nào trên toàn tập).**
Recall không xác định, do đó **không báo F1**. Thay vào đó báo số `FP` tuyệt đối kèm Restraint Rate, và ghi rõ tập này không có positive nên không đánh giá được năng lực trích xuất.

Trường hợp này có thật trong dữ liệu: toàn bộ 180 mẫu `legal_concept_understanding` đều có `scope` ground truth rỗng, nhưng model vẫn sinh scope ở một số mẫu. Cách báo đúng:

```text
scope: không có positive trong ground truth (TP + FN = 0)
       -> không báo F1
       FP = 3
       Restraint Rate = 177/180 = 98.3%
```

Nếu vẫn cố tính F1 ở đây sẽ ra `Precision = 0/3 = 0%` nhưng `Recall = 0/0` không xác định, tức con số F1 thu được là giả.

**Trường hợp 2 — `TP + FP = 0` (model không sinh positive nào trên toàn tập).**
Precision quy ước bằng `0`, F1 bằng `0`, và phải ghi rõ model không sinh phần tử nào cho field đó.

Năng lực trả về danh sách rỗng đúng lúc được báo riêng bằng **Restraint Rate**:

```text
Restraint Rate
    = số mẫu có ground truth = [] và prediction cũng = []
      / số mẫu có ground truth = []
```

Không tính F1 riêng từng mẫu bằng cách gán `F1 = 1` cho trường hợp rỗng - rỗng rồi lấy trung bình, vì cách này có thể làm điểm bị thổi phồng khi phần lớn ground truth là danh sách rỗng.

### 1.2. Baseline thoái hóa

Mọi metric chính phải được báo cáo kèm điểm của một baseline thoái hóa — model luôn trả về giá trị đa số hoặc rỗng, không đọc input. Baseline này cho biết mốc sàn của metric.

Một metric mà model thật chỉ hơn baseline thoái hóa vài điểm phần trăm thì không đủ khả năng phân biệt, cần tách nhỏ hoặc thay thế.

### 1.3. Breakdown theo `slice`

Mỗi mẫu trong tập test đều có sẵn field `slice` mô tả khía cạnh được kiểm tra. Mọi metric chính nên được báo cáo thêm ở mức từng `slice`, vì việc này không tốn thêm lần chạy inference nào mà cho biết model yếu ở đúng khía cạnh nào.

### 1.4. Tính chất của micro-average cần lưu ý khi đọc số

Micro-average đánh trọng số theo số lượng phần tử, không theo số mẫu: một mẫu có 5 concept ảnh hưởng tới điểm gấp 5 lần một mẫu có 1 concept. Đây là cách tính chuẩn cho bài toán trích xuất, nhưng cần biết để giải thích khi con số không khớp với cảm nhận "trung bình từng câu".

Vì lý do này, luôn báo `TP`, `FP`, `FN` tuyệt đối kèm Precision/Recall/F1. Ví dụ đo trên model chưa fine-tune, field `concepts` của `legal_concept_understanding`: `TP = 49`, `FP = 332`, `FN = 125`, tức Precision chỉ 12.9% — cứ khoảng 8 concept sinh ra mới có 1 cái đúng. Con số `FP` tuyệt đối cho thấy mức độ bịa đặt rõ hơn nhiều so với chỉ nhìn F1 = 17.7%.

## 2. `intent_understanding`

Field được đánh giá chính là `intent`.

### 2.1. Accuracy

So sánh trực tiếp `intent` dự đoán với ground truth. Một mẫu đúng khi hai giá trị giống nhau.

```text
Accuracy = số mẫu có intent đúng / tổng số mẫu intent_understanding
```

Ví dụ:

```text
Gold:       provision_content
Prediction: provision_content
Kết quả:    đúng
```

```text
Gold:       provision_content
Prediction: rule_detail
Kết quả:    sai
```

### 2.2. Macro-F1

Tính F1 riêng cho từng intent rồi lấy trung bình trên tất cả intent:

```text
Precision_i = TP_i / (TP_i + FP_i)
Recall_i    = TP_i / (TP_i + FN_i)
F1_i        = 2 * Precision_i * Recall_i / (Precision_i + Recall_i)
Macro-F1    = trung bình F1 của tất cả intent
```

Macro-F1 giúp phát hiện trường hợp model chỉ dự đoán tốt các intent xuất hiện nhiều nhưng yếu ở các intent ít gặp.

Phân bố intent trong tập test lệch khoảng 5 lần giữa nhãn nhiều nhất và ít nhất (`rule_detail` 42 mẫu so với `ambiguous` 8 mẫu), nên khoảng cách giữa hai metric là đáng kể. Đo trên model chưa fine-tune: Accuracy 76.7% nhưng Macro-F1 chỉ 65.9%, do `ambiguous` đạt F1 = 0.0% và `out_of_scope` đạt F1 = 11.1%. Hai nhãn này quan trọng về mặt an toàn hệ thống, nếu chỉ báo Accuracy thì điểm yếu bị che mất.

### 2.3. Breakdown theo `slice`

Tập test chia làm hai loại: `clean` (124 mẫu, câu hỏi rõ ràng) và `boundary:*` (56 mẫu, được thiết kế riêng để test ranh giới giữa các cặp intent dễ nhầm như `rule_detail` với `applicability`). Hai nhóm này phải được báo riêng vì độ khó chênh nhau rõ rệt — đo trên model chưa fine-tune: `clean` 79.8% so với `boundary` 69.6%.

### Metric báo cáo

- Metric chính: **Intent Accuracy**, kèm baseline thoái hóa (luôn dự đoán nhãn đa số).
- Metric bổ sung: **Intent Macro-F1**, F1 của từng nhãn, và Accuracy tách theo `clean` / `boundary`.

## 3. `legal_concept_understanding`

Field được đánh giá chính là `concepts`.

### 3.1. Concept Exact Match Accuracy

Một mẫu chỉ được tính đúng nếu toàn bộ tập `concepts` dự đoán khớp ground truth. Không xét thứ tự phần tử.

```text
Concept Exact Match Accuracy
    = số mẫu có tập concepts khớp hoàn toàn / tổng số mẫu legal_concept_understanding
```

Ví dụ đúng:

```json
Gold:       ["khái niệm A", "khái niệm B"]
Prediction: ["khái niệm B", "khái niệm A"]
```

Ví dụ sai do thiếu một concept:

```json
Gold:       ["khái niệm A", "khái niệm B"]
Prediction: ["khái niệm A"]
```

### 3.2. Precision, Recall và F1

Xem mỗi concept là một phần tử cần trích xuất:

- `TP`: concept dự đoán đúng.
- `FP`: concept dự đoán thừa hoặc sai.
- `FN`: concept trong ground truth bị bỏ sót.

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

Ví dụ:

```json
Gold:       ["A", "B"]
Prediction: ["A", "C"]
```

Kết quả:

```text
TP = 1
FP = 1
FN = 1
Precision = 0.5
Recall    = 0.5
F1        = 0.5
Exact Match = 0
```

Theo quy ước tại mục 1.1, Concept Precision/Recall/F1 được tính theo micro-average trên toàn bộ 180 mẫu. Prediction sinh concept trong 17 mẫu có ground truth rỗng vẫn bị tính là `FP`.

### 3.3. Concept Restraint Rate

17 mẫu có `concepts` ground truth là rỗng, trong đó 6 mẫu thuộc slice `no_concept` được thiết kế riêng để kiểm tra việc model có biết khi nào không nên trích khái niệm nào hay không.

```text
Concept Restraint Rate
    = số mẫu có concepts gold = [] và prediction cũng = []
      / số mẫu có concepts gold = []
```

Đây là điểm yếu rõ rệt cần theo dõi: model chưa fine-tune chỉ đạt 35% ở metric này, tức phần lớn trường hợp vẫn bịa ra khái niệm dù đáng lẽ phải để rỗng.

### 3.4. Lưu ý về chất lượng ground truth

Một số mẫu ground truth gộp nhiều khái niệm vào cùng một chuỗi, ví dụ `"thoả thuận khung, quy trình mua sắm tập trung"` được ghi thành một phần tử thay vì hai. Khi model tách đúng thành hai khái niệm riêng, Exact Match trả về 0 và F1 bị giảm mạnh dù nội dung hoàn toàn chính xác.

Khi báo cáo Exact Match thấp, phải kèm F1 và ghi rõ phần chênh lệch có thể đến từ cách tách/gộp cụm từ của ground truth, không hẳn do model sai.

### Metric báo cáo

- Metric chính: **Concept Exact Match Accuracy**, kèm baseline thoái hóa (luôn trả `[]`).
- Metric bổ sung: **Concept Micro Precision, Micro Recall và Micro-F1** (toàn bộ 180 mẫu), **Concept Restraint Rate**, và breakdown theo `slice`.

## 4. `scope_user_facts_understanding`

Task này đánh giá ba field:

```text
scope
user_facts
asserts_premise
```

### 4.1. Structured Exact Match Accuracy

Một mẫu chỉ được tính đúng khi cả ba field đều khớp ground truth:

```text
scope đúng AND user_facts đúng AND asserts_premise đúng
```

Đối với `scope`, mỗi phần tử được xem là một cặp `(kind, value)`. Hai phần tử chỉ giống nhau khi cả `kind` và `value` đều giống nhau.

Ví dụ đúng, do không xét thứ tự:

```json
Gold: [
  {"kind": "ARTICLE", "value": "18"},
  {"kind": "DOCUMENT", "value": "23/2024/NĐ-CP"}
]

Prediction: [
  {"kind": "DOCUMENT", "value": "23/2024/NĐ-CP"},
  {"kind": "ARTICLE", "value": "18"}
]
```

Structured Exact Match trên toàn bộ 180 mẫu vẫn được báo như metric end-to-end. Tuy nhiên, 62/180 mẫu thuộc slice `empty_state` có cả ba field đều rỗng hoặc `false`, nên một model chỉ cần luôn trả về `scope=[]`, `user_facts=[]`, `asserts_premise=false` đã đạt 34.4% mà không cần đọc input. Model chưa fine-tune đạt 35.6%, tức chỉ hơn baseline thoái hóa **1.2 điểm phần trăm**.

Vì vậy, Structured Exact Match toàn tập phải đi kèm hai metric phân nhóm:

```text
Structured EM (overall)
    = số mẫu đúng cả ba field / 180

Structured EM (extraction)
    = số mẫu đúng cả ba field, trong nhóm có ít nhất một field khác rỗng
      / số mẫu có ít nhất một field khác rỗng

Structured Restraint Rate
    = số mẫu đúng cả ba field, trong nhóm empty_state
      / số mẫu thuộc nhóm empty_state
```

Nhóm extraction đo năng lực trích xuất (118/180 mẫu), còn nhóm `empty_state` đo năng lực kiềm chế (62/180 mẫu). Hai metric phân nhóm phải được báo cùng Structured EM toàn tập để kết quả vừa có tính tổng quan, vừa không che giấu ảnh hưởng của lớp đa số.

### 4.2. Scope F1

So sánh các cặp `(kind, value)`:

- `TP`: scope đúng cả `kind` và `value`.
- `FP`: scope dự đoán thừa hoặc sai.
- `FN`: scope trong ground truth bị bỏ sót.

```text
Scope Precision = TP / (TP + FP)
Scope Recall    = TP / (TP + FN)
Scope F1        = 2 * Scope Precision * Scope Recall
                  / (Scope Precision + Scope Recall)
```

Theo quy ước tại mục 1.1, Scope F1 được tính theo micro-average trên toàn bộ 180 mẫu. Các mẫu rỗng - rỗng không làm tăng điểm, nhưng scope sinh thừa trong 102 mẫu có ground truth rỗng vẫn được tính là `FP`.

Năng lực kiềm chế của riêng field `scope` báo bằng Scope Restraint Rate trên 102 mẫu có `scope` gold rỗng.

### 4.3. User Facts F1

So sánh chính xác các giá trị `text` trong danh sách `user_facts` sau bước chuẩn hóa Unicode và khoảng trắng ở mục 1. Không tự động coi hai cách diễn đạt đồng nghĩa là một match.

- `TP`: tình tiết dự đoán đúng.
- `FP`: tình tiết được sinh thừa hoặc sai.
- `FN`: tình tiết trong ground truth bị bỏ sót.

Precision, Recall và F1 được tính theo micro-average trên toàn bộ 180 mẫu, tương tự `Scope F1`. Nếu cần chấp nhận các cách diễn đạt khác nhau nhưng cùng nghĩa, phải báo thêm một metric semantic riêng với judge và rubric cố định; không trộn kết quả semantic vào Exact Match F1.

### 4.4. Premise F1

Chỉ 17/180 mẫu có `asserts_premise = true`, nên accuracy thông thường bị lệch lớp nặng: một model luôn trả `false` đã đạt 90.6%, trong khi model chưa fine-tune đạt 94.4% — chỉ hơn 3.8 điểm phần trăm.

Do đó dùng F1 trên lớp positive thay cho accuracy:

```text
TP = số mẫu gold = true và prediction = true
FP = số mẫu gold = false và prediction = true
FN = số mẫu gold = true và prediction = false

Premise F1 = 2 * Precision * Recall / (Precision + Recall)
```

Accuracy vẫn có thể báo kèm nhưng luôn phải đi cùng baseline "luôn trả `false`" để người đọc biết mốc sàn.

### 4.5. Breakdown theo `slice`

Tập test gồm `empty_state` (62), `single_scope` (45), `user_facts` (31), `multi_scope` (25) và `asserts_premise` (17). Cần báo riêng ít nhất:

- Scope F1 tách theo `single_scope` và `multi_scope`, vì trích một scope dễ hơn hẳn trích nhiều scope.
- User Facts F1 trên slice `user_facts`.
- Premise F1 trên slice `asserts_premise`, vì đây là nhóm duy nhất có nhãn `true`.

### Metric báo cáo

- Metric chính: **Structured EM (overall)**, **Structured EM (extraction)** và **Structured Restraint Rate**, báo song song.
- Metric bổ sung: **Scope Micro-F1** (toàn bộ 180 mẫu), **User Facts Micro-F1**, **Premise F1**, và breakdown theo `slice`.

## 5. `question_rewrite_compatibility`

Field được đánh giá chính là `question`.

### 5.1. Question Exact Match Accuracy

So sánh chuỗi `question` dự đoán với ground truth sau khi chuẩn hóa Unicode và khoảng trắng:

```text
Question Exact Match Accuracy
    = số câu khớp hoàn toàn với ground truth
      / tổng số mẫu question_rewrite_compatibility
```

Metric này nghiêm ngặt: hai câu khác cách diễn đạt vẫn bị tính là sai dù có thể cùng ý nghĩa.

### 5.2. Semantic Accuracy

Mỗi câu dự đoán được chấm đúng hoặc sai theo bốn điều kiện:

- Giữ đúng ý nghĩa câu hỏi.
- Giải quyết đúng tham chiếu từ các lượt hội thoại trước.
- Có thể hiểu độc lập mà không cần đọc lại hội thoại.
- Không bổ sung thông tin không xuất hiện trong input.

```text
Semantic Accuracy
    = số câu đáp ứng đầy đủ các điều kiện
      / tổng số mẫu question_rewrite_compatibility
```

Việc chấm đúng/sai có thể do người đánh giá hoặc một LLM judge sử dụng rubric cố định. Nếu dùng LLM judge, đầu ra nên là nhãn nhị phân `0` hoặc `1`.

Điều kiện bắt buộc khi dùng LLM judge, để hai lần chạy so sánh được với nhau:

- Không dùng chính model đang được đánh giá làm judge, vì model tự thiên vị đầu ra của mình và Qwen2.5-7B quá yếu cho vai trò này. Dùng một model mạnh hơn, độc lập.
- Đặt `temperature = 0` để giảm độ ngẫu nhiên và tăng khả năng tái lập; thiết lập này không bảo đảm kết quả deterministic tuyệt đối với mọi model hoặc API.
- Dùng **cùng một judge và cùng một rubric** cho cả lần chạy base lẫn lần chạy fine-tune. Nếu đổi judge giữa hai lần, hai con số không so sánh được.
- Lưu lại toàn bộ phán quyết của judge kèm lý do, để có thể kiểm tra lại thủ công khi số liệu bất thường.

Question Exact Match và Semantic Accuracy không được thay thế cho nhau: Exact Match đo khả năng khớp đúng chuỗi chuẩn, còn Semantic Accuracy chấp nhận cách diễn đạt khác nếu câu hỏi vẫn đúng nghĩa và tự đầy đủ.

### 5.3. Breakdown theo `slice`

Tập test gồm `anaphora` (62), `single_turn` (41), `topic_switch` (19), `continuation` (18), `null_discipline` (18), `title_only` (12), `document_copy` (10). Slice `anaphora` là nhóm khó nhất vì câu hỏi rút gọn, phải lấy chủ ngữ từ lượt trước; `single_turn` là nhóm dễ nhất vì không có ngữ cảnh trước. Báo riêng hai nhóm này cho biết model có thực sự resolve được tham chiếu hay chỉ chép lại câu hỏi.

### Metric báo cáo

- Metric chính: **Question Semantic Accuracy**.
- Metric bổ sung: **Question Exact Match Accuracy** và breakdown theo `slice`.

## 6. Metric định dạng dùng chung

### 6.1. JSON Valid Rate

Đo tỷ lệ prediction có thể parse thành JSON:

```text
JSON Valid Rate
    = số prediction parse được thành JSON / tổng số prediction
```

### 6.2. Schema Valid Rate

Đo tỷ lệ prediction có đầy đủ field bắt buộc và đúng kiểu dữ liệu:

```text
Schema Valid Rate
    = số prediction đúng schema / tổng số prediction
```

Một prediction parse được thành JSON nhưng thiếu field hoặc sai kiểu dữ liệu vẫn không đạt `Schema Valid Rate`.

## 7. Bộ metric tổng hợp

| Task | Metric chính | Metric bổ sung |
|---|---|---|
| `intent_understanding` | Intent Accuracy | Intent Macro-F1, F1 từng nhãn, breakdown `clean` / `boundary` |
| `legal_concept_understanding` | Concept Exact Match Accuracy | Concept Micro Precision/Recall/F1 (toàn tập), Concept Restraint Rate |
| `scope_user_facts_understanding` | Structured EM (overall) + Structured EM (extraction) + Structured Restraint Rate | Scope Micro-F1, User Facts Micro-F1, Premise F1 |
| `question_rewrite_compatibility` | Question Semantic Accuracy | Question Exact Match Accuracy |
| Toàn bộ prediction | JSON Valid Rate | Schema Valid Rate |

Mọi metric chính đều báo kèm điểm của baseline thoái hóa và breakdown theo `slice`.

Phiên bản tối giản để báo cáo kết quả chính:

```text
U1: Intent Accuracy + Macro-F1
U2: Concept Exact Match Accuracy + Restraint Rate
U3: Structured EM (overall) + Structured EM (extraction) + Structured Restraint Rate
C1: Question Semantic Accuracy
Chung: JSON Valid Rate
```

Các metric F1 được sử dụng để phân tích những trường hợp model dự đoán đúng một phần nhưng bị Exact Match tính là sai hoàn toàn.

## 8. Bảng đối chiếu baseline thoái hóa

Số liệu đo trên model chưa fine-tune (`Qwen2.5-7B-Instruct`, 4-bit) để làm mốc tham chiếu. Cột chênh lệch cho biết metric có đủ khả năng phân biệt hay không.

| Metric | Model chưa fine-tune | Baseline thoái hóa | Chênh lệch |
|---|---|---|---|
| U1 Intent Accuracy | 76.7% | 23.3% (luôn trả `rule_detail`) | +53.4 |
| U2 Concept Exact Match | 18.9% | 9.4% (luôn trả `[]`) | +9.5 |
| U3 Structured EM (overall, phải kèm hai metric phân nhóm) | 35.6% | 34.4% (luôn trả rỗng) | **+1.2** |
| U3 Premise Accuracy (chỉ dùng bổ sung) | 94.4% | 90.6% (luôn trả `false`) | **+3.8** |

Hai dòng cuối là lý do metric của `scope_user_facts_understanding` được tách lại theo mục 4.1 và 4.4.
