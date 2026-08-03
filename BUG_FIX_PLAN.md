# Kế hoạch khắc phục lỗi sau kiểm thử local

Run ID: `20260803T140026Z`
Batch/profile: `aic25-b1-reference` / `laptop`
Phạm vi: Textual KIS, Visual Q&A, TRAKE, health/warm-up và CSV export. Đây là kiểm thử vận hành; không phải đánh giá độ chính xác vì không có bộ query/ground truth có nhãn.

## Remediation run `20260803T163203Z`

Toàn bộ issue `C-01`, `H-01..H-05`, `M-01..M-07` và `L-01` đã được sửa và xác minh. Run này không gọi Gemini, không ingest corpus, không thay model/weights và không sửa hai archive được bảo vệ.

| Bằng chứng | Kết quả |
|---|---|
| Ruff + pytest | `601 passed, 1 skipped`; skip duy nhất là symlink privilege Windows đã có |
| `scripts/check.ps1` | Pass |
| `scripts/check.ps1 -IncludeServices -IncludeStreamlit` | Pass; Qdrant/Meilisearch healthy, Streamlit AppTest pass |
| First snapshot build | Warm-up `87.968s`; snapshot build `58.641s`; dưới budget `90s` |
| Snapshot process kế tiếp | `source=disk`, build `0ms`, full warm-up `29.567s` |
| 10 startup process độc lập | `10/10 ready`, `source=disk`, max `61.773s`, không rebuild Qdrant trajectory |
| KIS warm, 50 iterations | p95 `759.904ms`, 50/50 thành công |
| KIS raw refinement, 10 iterations | p95 `1,642.620ms`, 10/10 thành công |
| TRAKE N=6, 10 iterations | p95 `959.621ms`, 10/10 thành công, 100 hit/lần |
| Resource peak quan sát | RSS `2.447GB`; CUDA allocated peak `780MB`, reserved `799MB` |
| Protected archives | `Keyframes_L21.zip` = `30412ab4c3c5f3bbd7cef0feb59e7670a6683d2d8d4a90b196cea9688d554221`; `Videos_L21_a.zip` = `2122c41762815a3cacb38d348480b7be5c0e26faa5541321e6524aebee3c52de`; đều khớp manifest |
| Final runtime | Streamlit `127.0.0.1:8504` trả `200 ok`; FastAPI `127.0.0.1:8080` đã dừng |

Bằng chứng benchmark nằm tại `outputs/benchmarks/remediation-kis.json`, `outputs/benchmarks/remediation-kis-raw.json`, `outputs/benchmarks/remediation-trake.json` và `outputs/benchmarks/startup-samples/`.

Các patch đã xác minh gồm: stable TRAKE selector/error boundary; strict stage/frame/time contracts; immutable reranked export snapshot; candidate-scoped persisted-frame membership; schema v2 `raw_refined_frames`; canonical/refined image resolver; mmap trajectory snapshot v3; workflow-specific warm-up/API gates; pin/unpin/clear callbacks; persistent validated CSV bytes; workflow segmented control; timeline ba stage mỗi hàng; tokenizer-aware dense-query chunking và diagnostics được sanitize.

## Kết quả tổng quan

| Hạng mục | Kết quả |
|---|---|
| Baseline | Ruff pass; 518 test pass, 1 test symlink Windows skip |
| Service/Streamlit checks | Pass; Qdrant và Meilisearch HTTP 200 |
| SQLite | Cả 3 store đều `integrity_check=ok` và `quick_check=ok` |
| Protected archives | SHA-256 tính lại khớp chính xác manifest |
| KIS API | 10/10 request HTTP 200; p95 1,870 giây; invariants pass |
| Q&A deterministic | 123 test pass; một live attempt bị chặn bởi `network_unreachable` |
| TRAKE API N=2..6 | HTTP 200; chậm nhất 1,247 giây; output thực tăng nghiêm ngặt theo frame và timestamp |
| Cold warm-up FastAPI | 109,562 giây — không đạt cổng 90 giây |
| Warm-up Streamlit sau cache hệ thống | 66,7 giây — đạt cổng 90 giây |
| Liveness trong warm-up | 79 mẫu, 0 failure, max 442,9 ms — không tái hiện nghẽn `/healthz` |
| Streamlit | Tiếng Việt và auto-unlock pass; KIS export/pin, ảnh ZIP và TRAKE result render fail như ghi dưới đây |

Bằng chứng đã sanitize nằm tại `outputs/test-runs/20260803T140026Z/`. Không lưu key, frame gửi Gemini, provider body, credential, hay endpoint có credential.

## Critical

### C-01 — TRAKE UI crash với mọi tập kết quả không rỗng và lộ traceback cục bộ

- **Trạng thái:** Fixed / Verified trong run `20260803T163203Z`.
- **Mô tả / expected / actual:** Sau một truy vấn TRAKE thành công, UI phải chọn ứng viên đầu tiên và render timeline. Thực tế bảng kết quả xuất hiện rồi ứng dụng ném `KeyError: None`; thành phần lỗi mặc định của Streamlit hiển thị traceback và đường dẫn `<workspace>\...`.
- **Repro:** Mở tab TRAKE, nhập chuỗi 6 giai đoạn cố định, chọn tìm kiếm. API trả kết quả hợp lệ nhưng render dừng tại candidate selector.
- **Bằng chứng:** `outputs/test-runs/20260803T140026Z/trake-keyerror.png`; DOM ghi nhận `KeyError: None` tại `app/streamlit_app.py:1886`. Bộ AppTest có result không rỗng cũng tái hiện độc lập.
- **Root cause — Confirmed:** `_invalidate_trake_results()` đặt `trake_selected_candidate=None` tại `app/streamlit_app.py:1739-1744`. Search thành công ghi result nhưng không xóa/sửa key tại `:1857-1861`. `st.selectbox` tái sử dụng `None`, rồi `choices[selected_label]` tại `:1876-1886` truy cập key không tồn tại.
- **Proposed fix / patch strategy:**
  1. `pop("trake_selected_candidate", None)` khi invalidate thay vì ghi `None`.
  2. Trước selectbox, nếu giá trị lưu không còn trong `choices`, đặt về label đầu tiên; dùng stable candidate ID thay cho label trình bày.
  3. Thêm production error boundary ghi event code đã sanitize; UI chỉ hiện mã tương quan, không hiện traceback/path.
- **Verification criteria:** AppTest và browser test với result không rỗng cho N=2..6 không crash; ứng viên đầu tiên được chọn; đổi candidate vẫn đúng; reload/search mới không giữ selection mồ côi; UI không chứa path/traceback.

## High

### H-01 — KIS export pin subset giữ rank gốc

- **Trạng thái:** Fixed / Verified trong run `20260803T163203Z`.
- **Mô tả / expected / actual:** Pin riêng kết quả hạng 5 phải tạo CSV một dòng với rank export 1. Thực tế UI hiển thị một dòng nhưng exporter từ chối vì rank vẫn là 5.
- **Repro:** Chạy KIS, chỉ chọn checkbox `Ghim` của `#5`; thông báo lỗi export xuất hiện ngay.
- **Bằng chứng:** `outputs/test-runs/20260803T140026Z/kis-pin-rank5-error.png`; `SubmissionValidationError: result ranks must be contiguous from 1` trong harness.
- **Root cause — Confirmed:** Pin giữ nguyên dict/rank retrieval tại `app/streamlit_app.py:522-530`. Bảng chỉ đánh rank lại để hiển thị tại `:605-612`, nhưng `:629-631` truyền raw pin sang validator; `src/submission.py:154-163` đúng đắn từ chối rank không liên tục.
- **Proposed fix / patch strategy:** Tạo immutable export snapshot theo đúng thứ tự pin đang hiển thị và gán lại `rank=1..m`; dùng snapshot cho bảng, bytes download và server save. Không sửa rank của kết quả retrieval gốc.
- **Verification criteria:** Pin riêng rank 5, pin thứ tự bất kỳ và pin từ nhiều lần search đều xuất `1..m`; CSV parse lại đúng thứ tự; duplicate vẫn bị từ chối.

### H-02 — KIS và QA chấp nhận source frame không tồn tại

- **Trạng thái:** Fixed / Verified trong run `20260803T163203Z`.
- **Mô tả / expected / actual:** Export phải chỉ nhận canonical `source_frame_idx` có thật. Harness với mapping chỉ chứa frame `10` vẫn chấp nhận frame `999999` cho cả KIS và QA.
- **Repro:** Cấu hình active batch và `FrameIndexEvidence` hợp lệ, truyền prediction cùng video nhưng source frame ngoài tập mapping.
- **Root cause — Confirmed:** `src/submission.py:126-130,170-190` và `src/qa_submission.py:154-159,213-242` kiểm tra active batch, ordinal evidence, format và duplicate nhưng thiếu callback membership canonical. API/UI chỉ truyền toàn bộ frame-map evidence tại `src/api.py:1537-1549,1581-1594` và `app/streamlit_app.py:629-636,1307-1315`.
- **Proposed fix / patch strategy:** Thêm `is_canonical_source_frame(batch, video, source_frame)` bắt buộc cho KIS/QA exporter. Raw-refined frame phải được persist thành `FrameRecord` canonical trước export. Dùng lookup theo candidate, không materialize toàn batch.
- **Verification criteria:** Reject frame thiếu/video thiếu/mixed batch/ordinal-only; accept canonical organizer và raw-refined đã persist; round-trip API/UI/export giữ đúng source frame.

### H-03 — TRAKE exporter bỏ qua `stage_index`

- **Trạng thái:** Fixed / Verified trong run `20260803T163203Z`.
- **Mô tả / expected / actual:** `stage_matches` phải có chỉ số strict integer đúng `1..N`. Exporter hiện chấp nhận missing, đảo thứ tự, duplicate và boolean cho N=2..6 nếu frame tăng.
- **Repro:** Gửi `stage_matches` có frame hợp lệ nhưng index `[2,2]`, reversed, missing hoặc `true`.
- **Root cause — Confirmed:** `_mapping_frames()` tại `src/trake_submission.py:50-78` chỉ đọc `source_frame_idx`; mapping đi thẳng qua parser tại `:81-97`, sau đó validation tại `:119-129,160-172` chỉ còn rank/count/frame nên không thể kiểm lại index.
- **Proposed fix / patch strategy:** Khi input có `stage_matches`, parse strict non-bool integer và bắt buộc tuple index đúng `1..N` trước khi tách frame. Giữ `frame_ids` là adapter đã có thứ tự rõ ràng.
- **Verification criteria:** Parameterized N=2..20 reject missing/bool/zero/gap/duplicate/reversed; valid `stage_matches` và direct `frame_ids` vẫn xuất bytes/width như cũ.

### H-04 — TRAKE contract/cache chưa bắt buộc timestamp tăng nghiêm ngặt

- **Trạng thái:** Fixed / Verified trong run `20260803T163203Z`.
- **Mô tả / expected / actual:** Mỗi alignment phải thỏa đồng thời frame và timestamp tăng nghiêm ngặt. Contract chấp nhận timestamp bằng/giảm; exact cache có thể phát `[100,100]` với frame `[10,20]`.
- **Repro:** Tạo response item hoặc cache points với frame tăng nhưng timestamp bằng/giảm.
- **Root cause — Confirmed:** `src/contracts.py:895-901,913-922` chỉ kiểm timestamp không âm và chronology frame. Exact cache tại `src/retrieval/trake_trajectory_cache.py:282-297` và fallback tại `src/retrieval/trake_alignment.py:476-482` chỉ retain theo frame; DP tại `:576-579,591-596` không thêm guard timestamp.
- **Proposed fix / patch strategy:** Thêm validator timestamp strict trong `TRAKEResponseItem`; lọc canonical point theo cả frame và timestamp ở cache/fallback; thêm defensive validation trước public response/export.
- **Verification criteria:** Contract/cache/pipeline reject equal/decreasing timestamp N=2..6; duplicate cùng frame vẫn merge deterministic; greedy trap, ties, `T<N` và output hợp lệ không regression.

### H-05 — Cold warm-up vượt ngân sách 90 giây

- **Trạng thái:** Fixed / Verified trong run `20260803T163203Z`; root cause snapshot đã được xác nhận bằng process thứ hai có zero Qdrant rebuild.
- **Mô tả / expected / actual:** Laptop cold warm-up phải ≤90 giây. FastAPI cold run mất 109,562 giây; lần Streamlit sau cache hệ thống mất 66,7 giây.
- **Repro:** Process mới, cache lạnh, Batch 1; poll warm-up đến terminal. Trajectory 61,016 giây, organizer CLIP 98,844 giây, raw refiner 6,406 giây; các stage overlap nhưng runtime chỉ publish terminal sau join.
- **Root cause — Hypothesis:** Mỗi process dựng lại process-local trajectory cache bằng cách tải/xử lý toàn bộ 177.321×512 vector (`src/retrieval/trake_trajectory_cache.py:188-315,326-360`; Qdrant scroll `src/indexing/vector_store.py:782-875`). Đồng thời model cold-load/first inference chạy tại `src/warmup.py:297-355`. `src/warmup.py:357-362` chờ trajectory hoàn tất trước khi publish usable. Giá trị 90 giây trong config chỉ được benchmark dùng, không được WarmupService thực thi.
- **Proposed fix / patch strategy:**
  1. Persist snapshot trajectory fingerprinted, checksummed và atomic; process mới mmap/load snapshot thay vì scroll toàn collection.
  2. Tách readiness theo workflow: mở KIS ngay khi kênh KIS đã warm, trong khi TRAKE trajectory tiếp tục warm single-flight.
  3. Benchmark staged-vs-concurrent I/O/model load để tránh tranh chấp CPU/disk; giữ một GPU-heavy model tại một thời điểm.
  4. Ghi phase timing và cảnh báo budget; không giả trạng thái ready khi vượt hạn.
- **Verification criteria:** 10 process cold độc lập đều ≤90 giây; KIS usable không phải chờ cache chỉ phục vụ TRAKE; snapshot fingerprint mismatch rebuild an toàn; kill/restart không đọc snapshot dở; RAM/VRAM vẫn dưới gate.

## Medium

### M-01 — Bỏ ghim/Xóa KIS không đồng bộ widget state

- **Trạng thái:** Fixed / Verified trong run `20260803T163203Z`.
- **Mô tả / expected / actual:** Uncheck hoặc Xóa phải loại pin ngay và không tái xuất hiện. Thực tế merge chỉ thêm; nút Clear xóa list nhưng widget `pin_*` còn state nên rerun có thể thêm lại.
- **Repro:** Ghim rank 5, uncheck; khu vực export/Clear vẫn tồn tại. Nhấn Clear; state hiển thị và state pin không thay đổi nguyên tử.
- **Root cause — Confirmed:** `app/components/result_gallery.py:39-45` chỉ trả checkbox đang chọn, không phát removal; merge `app/streamlit_app.py:523-530` chỉ add/overwrite; Clear tại `:638-640` không xóa widget keys.
- **Proposed fix / patch strategy:** Dùng callback `Ghim/Bỏ ghim` theo stable ID cập nhật một source of truth; Clear xóa pin và toàn bộ widget key liên quan trong cùng callback trước rerun.
- **Verification criteria:** Check/uncheck/clear qua nhiều rerun và search không hồi sinh pin; export snapshot khớp đúng danh sách nhìn thấy.

### M-02 — Export/UI scan toàn bộ khoảng 177 nghìn mappings

- **Trạng thái:** Fixed / Verified trong run `20260803T163203Z` bằng repository/API test cấm full-batch scan.
- **Mô tả / expected / actual:** Export vài candidate chỉ nên lookup các frame cần thiết. KIS/QA/TRAKE gọi mapping không có `video_id`, fetch và materialize toàn batch trên rerun.
- **Root cause — Confirmed:** Các call tại `app/streamlit_app.py:629,1307,1704`; `src/ingest/artifacts.py:600-626` chỉ lọc `batch_id` rồi tạo nested dict đầy đủ.
- **Proposed fix / patch strategy:** Thêm repository API batch lookup theo tập `(video_id, source_frame_idx)`; validate bằng query scoped/indexed. Cache CSV bytes theo fingerprint của result/spec/batch và invalidate khi pin đổi.
- **Verification criteria:** Query count/rows đọc tỷ lệ với candidate; export 1 và 100 row không scan full table; CSV và negative validators không đổi.

### M-03 — QA download gắn với server write và biến mất sau rerun

- **Trạng thái:** Fixed / Verified trong run `20260803T163203Z`.
- **Mô tả / expected / actual:** Nút download phải tồn tại sau rerun và không cần tạo file server. Hiện file server còn nhưng download widget biến mất ở rerun kế tiếp.
- **Root cause — Confirmed:** `app/streamlit_app.py:1300-1323` tạo file và `st.download_button` chỉ bên trong nhánh một-run của nút chuẩn bị; bytes/filename không lưu session.
- **Proposed fix / patch strategy:** Tạo validated CSV bytes thuần trong memory, lưu bytes + filename + fingerprint vào session; render download ngoài action branch. Server save là nút/action riêng.
- **Verification criteria:** Download còn sau rerun/tab switch/session restore; không có file server nếu chưa chọn Save; bytes numeric/general parse đúng và answered-only.

### M-04 — Lỗi UI thiếu event code/log correlation

- **Trạng thái:** Fixed / Verified trong run `20260803T163203Z`.
- **Mô tả / expected / actual:** Operator cần mã lỗi tra log đã sanitize. Các `except Exception` hiện chỉ hiển thị thông báo chung; exception không được ghi với workflow/batch/query/session/latency. Lỗi không được bắt như C-01 lại lộ raw traceback/path.
- **Root cause — Confirmed:** Các catch path tại `app/streamlit_app.py:498-501,594-598,1324-1328,1451-1458,1862-1866` không gọi structured logger/correlation helper.
- **Proposed fix / patch strategy:** Thêm helper event-code dùng allowlist fields; log server-side workflow, batch, sanitized query/session ID, stage và latency. UI chỉ hiện tiếng Việt + correlation ID; tuyệt đối không body/header/path/credential.
- **Verification criteria:** Mỗi failure path có một event; UI/API không chứa secret/path/raw exception; test redaction với URL/key/header/traceback fixtures.

### M-05 — KIS không render thumbnail Batch 1 dạng `zip://`

- **Trạng thái:** Fixed / Verified trong run `20260803T163203Z` cho local, `zip://` và refined WebP.
- **Mô tả / expected / actual:** Kết quả organizer phải hiển thị thumbnail. Truy vấn KIS trả 20 hit nhưng DOM không có ảnh; chỉ metadata/frame text xuất hiện.
- **Bằng chứng:** `outputs/test-runs/20260803T140026Z/kis-pin-rank5-error.png`.
- **Root cause — Confirmed:** Reference importer ghi `zip://...!/keyframes/...jpg` tại `src/ingest/reference_importer.py:261-264`; gallery chỉ render khi `Path(image_uri).is_file()` tại `app/components/result_gallery.py:27-28`.
- **Proposed fix / patch strategy:** Dùng canonical image resolver dùng chung với QA/TRAKE, có allowlist root, bounded read và MIME validation; cache bytes/thumb theo content fingerprint.
- **Verification criteria:** Local path và `zip://` đều render; archive member traversal/bad URI bị từ chối; rerun không giải nén toàn archive.

### M-06 — Tab TRAKE tự quay về KIS sau widget rerun

- **Trạng thái:** Fixed / Verified trong run `20260803T163203Z` bằng AppTest qua add/remove/search rerun.
- **Mô tả / expected / actual:** Thêm stage hoặc chỉnh input phải giữ tab TRAKE. Thực tế mỗi `st.rerun`/`on_change` quay lại tab KIS, buộc người dùng chọn lại tab nhiều lần để nhập N=6.
- **Repro:** Trong TRAKE, bấm `Thêm giai đoạn` hoặc commit text input; sau rerun tab KIS được chọn mặc định dù form TRAKE vẫn còn session state.
- **Root cause — Hypothesis:** Active `st.tabs` selection không được lưu trong session; các callback `_invalidate_trake_results` và explicit `st.rerun()` làm app dựng lại với tab đầu tiên.
- **Proposed fix / patch strategy:** Dùng navigation/segmented control có key session-state làm workflow selector, hoặc form để defer rerun đến submit. Sau add/remove, ghi active workflow là TRAKE trước rerun.
- **Verification criteria:** Add/remove và nhập 2..20 stage không rời TRAKE; switching KIS/QA chủ động vẫn đúng; state mỗi workflow không rò sang nhau.

### M-07 — Long-query dense truncation không có cảnh báo

- **Trạng thái:** Fixed / Verified trong run `20260803T163203Z` với query 145 và 400 token.
- **Mô tả / expected / actual:** Query dài phải được xử lý có chủ đích hoặc cảnh báo phần bị cắt. API giữ nguyên 645 ký tự trong query plan và trả 200, nhưng không báo dense channel chỉ thấy tiền tố.
- **Repro:** Gửi query tiếng Việt 145 từ có before/current/after và cụm trích dẫn. Public response không có token/truncation metadata.
- **Root cause — Confirmed:** Contract/UI không giới hạn (`src/contracts.py:207-233`, `app/components/query_input.py:24-29`). Batch-1 OpenCLIP tokenizer tại `src/embedding/clip_encoder.py:132-148` giữ context 77 (75 content tokens) và không warning; SigLIP2 dùng `truncation=True,max_length=64` tại `src/embedding/siglip_encoder.py:174-198`. Lexical channel nhận full query nên hành vi hybrid bất đối xứng nhưng không được công khai.
- **Proposed fix / patch strategy:** Encoder adapter trả token diagnostics; QueryPlan chunk theo clause/câu trong giới hạn model và fuse dense chunk ranks; giữ nguyên text lexical. API/UI hiển thị warning và số token/channels bị cắt nếu fallback truncation vẫn xảy ra.
- **Verification criteria:** Query 145 từ không bị cắt im lặng; mỗi chunk trong context limit; before/current/after cuối query vẫn ảnh hưởng dense retrieval; short-query embedding/rank không regression.

## Low

### L-01 — Timeline TRAKE N≥6 quá hẹp

- **Trạng thái:** Fixed / Verified trong run `20260803T163203Z` cho N=2..20.
- **Mô tả / expected / actual:** Timeline phải đọc được trên laptop. `st.columns(len(matches))` dồn cả 6+ stage vào một hàng.
- **Root cause — Confirmed:** `app/streamlit_app.py:1887-1891` tạo N cột trực tiếp, không chunk/responsive layout.
- **Proposed fix / patch strategy:** Chia theo hàng tối đa 3 stage, đánh số liên tục và giữ mũi tên/thứ tự giữa các hàng; có thể thêm horizontal summary riêng.
- **Verification criteria:** N=2..20 bảo toàn thứ tự; viewport laptop đọc được thumbnail/frame/timestamp; keyboard/screen-reader order đúng.

## Vấn đề không tái hiện

- **Health bị nghẽn khi trajectory warm-up:** Không tái hiện. `/healthz` có 79 mẫu, zero failure, max 442,9 ms. Một lần `/v1/health/status` đầy đủ mất hơn 3 giây khi warming, nhưng liveness vẫn phản hồi và đây chưa đủ bằng chứng để mở product bug riêng.

## Test limitations / blockers

- Không có ground truth có nhãn, vì vậy không kết luận độ chính xác KIS/QA/TRAKE.
- Live Gemini dùng đúng một attempt và trả `network_unreachable`. Runtime đã nhận cấu hình key; không thực hiện attempt thứ hai, không test 5 window live và không tạo QA session live để restore trên UI. Đây là blocker kết nối môi trường/provider, chưa được xếp product bug.
- Rate limit, 429/backoff, concurrency, multi-window 5/10/15/20, counting/color normalization và session state được kiểm bằng provider injected/offline; 123 test pass.
- Browser harness có viewport cố định 1280×720; không ép được đúng 1366×768. Responsive behavior còn được kiểm bằng AppTest.
- Docker config access bị sandbox hạn chế nhưng Qdrant/Meilisearch đang chạy và health pass; không coi là product bug.

## Trình tự triển khai khuyến nghị

1. C-01, H-03 và H-04: chặn crash và bảo đảm contract TRAKE trước.
2. H-01, H-02: sửa toàn bộ trust boundary của export KIS/QA.
3. H-05: snapshot/cache và tách workflow warm-up gate.
4. M-01..M-05: ổn định pin/export/error/ảnh.
5. M-06, M-07 và L-01: hoàn thiện UX và query dài.

Mỗi patch phải chạy lại `scripts/check.ps1`; patch chạm service/UI phải thêm `-IncludeServices -IncludeStreamlit`. Không gộp sửa dữ liệu, model hay ingestion corpus vào các patch này.
