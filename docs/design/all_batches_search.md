# Tìm chung B01 + B02

## Phạm vi phát hành local

`retrieval.all_batches.enabled: true` bật mặc định **Tất cả dữ liệu — B01 + B02**
cho KIS, tìm ứng viên QA và TRAKE. Bộ lọc **Chỉ B01 / Chỉ B02** vẫn dùng đường
truy xuất singleton cũ. Batch dùng khởi động/health không bị đổi thành ID giả `all`.

Mỗi batch thực thi pipeline riêng, sau đó gộp theo thứ hạng đóng góp bằng nhau
`1 / (60 + rank)`, tie-break B01 trước B02. Với hai tập identity rời nhau, đây là
xen kẽ theo rank; không so sánh raw similarity giữa các model và không phải một
claim cải thiện chất lượng ranking đã được qrels chứng minh. Khi một batch không
có hit, phần còn lại tiếp tục lấp danh sách. Diagnostics giữ trạng thái từng channel
kèm batch và số kết quả từng nhánh.

- B01: giữ các channel đã provision.
- B02: Organizer-CLIP và objects; **chưa có OCR/ASR production**.
- Hai core TRAKE dùng cùng encoder nhưng giữ cache riêng, nạp tuần tự khi warm-up.
  Request không build/swap toàn cache. Chuỗi vẫn tăng nghiêm ngặt trong một video.
- B02 TRAKE chỉ M/S; loại N và M08_V001 theo contract đang có.
- Scope tổng chưa ready thì không giả vờ tìm đủ cả hai. Có thể chọn riêng batch đã ready.
- Feedback ảnh có batch/ordinal được phép tìm lại trên scope chung. Một lượt không
  trộn positive/negative từ hai batch; phải chọn một nguồn feedback rõ ràng.

## Ghim và CSV

Kết quả/QA candidate mang batch riêng. Danh sách CSV mới có thể chứa hai batch:
L/M/S dùng source frame, N dùng timestamp millisecond canonical từ PTS cho KIS/QA.
Ordinal B02 chỉ định ảnh, không được xuất thay source frame. TRAKE không nối video.
`POST /v1/submissions/preview` kiểm tra lại canonical mapping và active/release gate,
không gửi bài đến DRES và không ghi index.

CSV/ZIP mới nằm trong staging của phiên Streamlit, có nút tải ở panel và hub chung.
**Tải ZIP trước khi đóng/reset phiên**. Bài nộp B01 cũ được giữ riêng; không tự động
trộn vào ZIP mới. Việc tạo CSV cục bộ không đồng nghĩa DRES đã chấp nhận bài.

## Kiểm chứng và rollback

Checkpoint: `outputs/checkpoints/all-batches-20260925/`.
`pre-release-smoke.json` và `released-runtime.json` ghi các lần kiểm tra trên dịch vụ
thật, hai cache snapshot, KIS/QA/TRAKE, mapping CSV và count trước/sau. Các báo cáo
có fingerprint nguồn tương ứng từng lần; không dùng báo cáo cũ để khẳng định mã
nguồn sửa sau đó đã qua kiểm tra.

Full offline suite: **1.628 passed, 3 skipped** (quyền symlink Windows và hai
benchmark integration opt-in), Ruff/service probes/Streamlit AppTest đạt.
`final-runtime.json` giữ nguyên một lần **failed**: B01 cold request gặp
`tier2_deadline_exceeded`, lượt tiếp theo rerank thành công nên ranking không
giống nhau. Đây là giới hạn cold/warm của nhánh B01; không tăng deadline hay đổi
nhãn lỗi để che nó. Một lượt probe có `raw_refinement: channel_query_failed`
cũng được giữ trong diagnostics. Không tuyên bố mọi channel/query đều healthy
hoặc việc tìm chung đã tăng recall. Thử lại sau request warm-up được lưu riêng,
không ghi đè lần thất bại.

`final-warm-runtime.json` **passed** trên source cuối, gồm QA qua HTTP, KIS giữ
thứ tự từng nhánh sau query warm-up, TRAKE và CSV trộn batch, external count không
đổi. `release.json` ghi fingerprint và giới hạn đã biết. Đây là nghiệm thu chức
năng, không phải benchmark tăng recall hay xác nhận đã sửa deadline B01.

Rollback: đặt `retrieval.all_batches.enabled: false`, khởi động lại riêng Launcher
khi cần dùng. Mặc định quay lại batch đã chọn khi launch; không xóa cache, staging,
index hoặc dữ liệu. Bản source trước pha nằm trong `source-before/`; không dùng
reset/checkout/stash toàn worktree.
