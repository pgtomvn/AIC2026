# Hướng dẫn cài đầy đủ AIC 2026 trên từng máy

Tài liệu này dành cho thành viên cần tự dựng một máy có thể tìm kiếm dữ liệu thật bằng
Textual KIS, Visual Q&A và TRAKE.

Đây là scaffold đang phát triển, chưa phải hệ thống AIC 2026 hoàn chỉnh. Một số adapter,
model, evaluator và kết quả Streamlit vẫn là bản provisional hoặc placeholder. Cài đặt thành
công nghĩa là máy đã sẵn sàng chạy các workflow hiện có của repo; không có nghĩa là đã có
schema/chấm điểm chính thức cuối cùng của AIC 2026.

**Máy mới:** làm lần lượt mục 1–8. **Máy đã cài và import xong:** dùng mục 9 để khởi động
hằng ngày. **Gặp lỗi:** tra mục 10. Các lệnh dùng PowerShell tại thư mục gốc repo; ví dụ bên
dưới giả sử clone vào `D:\AIC2026`, hãy thay đường dẫn nếu máy bạn lưu ở nơi khác.

| Thành phần | Vai trò | Chạy bằng gì? | Địa chỉ mặc định |
|---|---|---|---|
| Qdrant | Lưu và tìm vector hình ảnh/văn bản | Docker Compose, service `qdrant` | REST `http://127.0.0.1:6333`, gRPC `6334` |
| Meilisearch | Tìm theo chữ, objects, metadata, OCR/ASR đã import | Docker Compose, service `meilisearch` | `http://127.0.0.1:7700` |
| Ollama | Phục vụ model trả lời Visual Q&A local | Ollama Windows, `ollama serve` hoặc launcher | `http://127.0.0.1:11434` |
| FastAPI | Backend truy vấn và readiness | Do `run_app.ps1` khởi động | `http://127.0.0.1:8080` |
| Streamlit | Giao diện cho người dùng | Do `run_app.ps1` khởi động | `http://127.0.0.1:8504` |

Luồng cài lần đầu: **tải công cụ và dữ liệu → bootstrap → bật hai database → tải model →
verify/import reference → kiểm tra readiness → mở UI**. Bật database không tự tải corpus hay
tạo các kết quả tìm kiếm.

## 1. Cần tải những gì?

### 1.1. GitHub không chứa dữ liệu lớn

Clone GitHub chỉ tải mã nguồn, test, cấu hình và script. Các file sau bị loại khỏi Git bởi
`.gitignore`, vì vậy clone xong sẽ **không có** ZIP dữ liệu, model, database, index hay `.env`:

- các archive `Keyframes_*.zip`, `Videos_*.zip`;
- bốn reference pack `map-keyframes-aic25-b1.zip`, `clip-features-32-aic25-b1.zip`,
  `objects-aic25-b1.zip`, `media-info-aic25-b1.zip`;
- model trong `models/`;
- state của Qdrant/Meilisearch, artifact SQLite, output và log.

Bốn reference pack và các ZIP media phải lấy từ nguồn dữ liệu chính thức hoặc thư mục chia sẻ
được team thống nhất. Không tự đổi tên file, không nén lại và không commit chúng vào Git.

Repo chưa cung cấp một URL tải công khai đã xác minh cho toàn bộ các pack. Thành viên mới cần
nhận link nguồn và quyền truy cập từ người phụ trách dữ liệu của team, tải đủ các tên ở mục 1.2,
rồi đặt vào repo. Không dùng link của một batch khác chỉ vì tên file gần giống nhau.

### 1.2. Bộ reference đầy đủ cho batch mặc định

Batch mặc định trong cấu hình là `aic25-b1-reference`. Để import reference đầy đủ, đặt đúng
tên và đúng vị trí như sau:

| Nhóm | File bắt buộc | Vị trí mặc định | Dùng cho |
|---|---|---|---|
| Map | `map-keyframes-aic25-b1.zip` | thư mục gốc repo | ánh xạ video/keyframe/`source_frame_idx` |
| Vector | `clip-features-32-aic25-b1.zip` | thư mục gốc repo | organizer CLIP 512-D, KIS/TRAKE |
| Objects | `objects-aic25-b1.zip` | thư mục gốc repo | lexical metadata/object search |
| Metadata | `media-info-aic25-b1.zip` | thư mục gốc repo | lexical metadata search |
| Keyframes | 14 ZIP bên dưới | thư mục gốc repo | canonical images, QA và provenance |

Danh sách **đúng 14** keyframe archive của bộ B1:

```text
Keyframes_L21.zip
Keyframes_L22.zip
Keyframes_L23.zip
Keyframes_L24.zip
Keyframes_L25.zip
Keyframes_L26_a.zip
Keyframes_L26_b.zip
Keyframes_L26_c.zip
Keyframes_L26_d.zip
Keyframes_L26_e.zip
Keyframes_L27.zip
Keyframes_L28.zip
Keyframes_L29.zip
Keyframes_L30.zip
```

Bốn pack reference phải đi cùng bộ keyframes tương ứng. Script `import_reference.py` mặc định
quét `Keyframes_*.zip` ở thư mục gốc; không trộn keyframes của batch khác vào đó. Cần đủ
873 video/177.321 mapping rows và ảnh tương ứng cho lộ trình này. Script có thể validate
map/features khi không tìm thấy ZIP ảnh, nên validation thành công riêng lẻ không bảo đảm
Visual Q&A có ảnh để dùng. Script stream trực tiếp từ ZIP, **không cần giải nén cả bộ**.

### 1.3. Có cần tải video gốc không?

Không cần cho đường chạy reference cơ bản:

- Textual KIS dùng organizer vectors + metadata;
- TRAKE dùng organizer vectors;
- Visual Q&A dùng canonical keyframe images + Ollama.

Chỉ tải thêm các video ZIP nếu team cần một trong các chức năng sau:

- raw-video refinement và giải mã frame chính xác từ video;
- ASR/audio;
- xem/phát lại video gốc;
- tự chạy native ingestion, tức tự trích keyframe/SigLIP2/OCR/ASR từ video.

Bộ video B1 tương ứng gồm:

```text
Videos_L21_a.zip
Videos_L22_a.zip
Videos_L23_a.zip
Videos_L24_a.zip
Videos_L25_a.zip
Videos_L26_a.zip
Videos_L26_b.zip
Videos_L26_c.zip
Videos_L26_d.zip
Videos_L26_e.zip
Videos_L27_a.zip
Videos_L28_a.zip
Videos_L29_a.zip
Videos_L30_a.zip
```

Giữ `Videos_L21_a.zip` ở thư mục gốc vì đây là archive được bảo vệ trong manifest của repo.
Các video còn lại nên để dưới `data/archives/aic25/b1/videos/`. Không di chuyển, đổi tên,
giải nén lại hoặc xóa hai archive được bảo vệ `Keyframes_L21.zip` và `Videos_L21_a.zip`.

Theo catalog hiện tại, 14 keyframe ZIP khoảng 28.7 GiB nén, 14 video ZIP khoảng 77.3 GiB
nén, và bốn reference pack khoảng 0.8 GiB. Toàn bộ khoảng 106.7 GiB **chưa tính** model,
cache, index và tối thiểu 30 GiB dung lượng trống dự phòng. Nếu chỉ chạy reference KIS/QA/TRAKE,
có thể bỏ qua khoảng 77.3 GiB video gốc.

## 2. Điều kiện máy

Lộ trình dưới đây dùng Windows PowerShell, Python 3.11 do `uv` quản lý và Docker Desktop.

### Bắt buộc

- Git for Windows;
- PowerShell 5.1 hoặc PowerShell 7;
- Docker Desktop chạy Linux containers, Docker Compose và backend WSL 2;
- tối thiểu khoảng 30 GiB trống ngoài dữ liệu/model; nên dành riêng một ổ đủ lớn nếu tải cả
  video;
- kết nối mạng trong lần đầu cài dependency/model;
- quyền mở các cổng loopback `6333`, `6334`, `7700`, `8080`, `8504`, `11434`.

### Khuyến nghị

- NVIDIA GPU và driver tương thích để SigLIP2/Whisper/Ollama chạy nhanh hơn;
- ít nhất 16 GiB RAM cho profile `laptop` theo cấu hình repo;
- khoảng 4 GiB VRAM là mức tham chiếu của profile laptop, nhưng thời gian ingest còn phụ thuộc
  video và model;
- FFmpeg/ffprobe nếu chạy native ingestion, audio hoặc raw refinement.

Docker Desktop cần được mở trước khi chạy service. Trên Windows, dùng Linux containers và
WSL 2. Có thể tham khảo [Git for Windows](https://git-scm.com/install/windows),
[Docker Desktop trên Windows](https://docs.docker.com/desktop/setup/install/windows-install/),
và [WSL install](https://learn.microsoft.com/en-us/windows/wsl/install).

Visual Q&A local cần Ollama bản Windows chính thức. Ollama phục vụ API loopback tại
`http://127.0.0.1:11434`; tài liệu chính thức ở [Ollama Windows](https://docs.ollama.com/windows).

### Cài công cụ trên máy Windows mới

1. Tải và cài [Git for Windows](https://git-scm.com/install/windows), sau đó mở lại PowerShell
   và chạy `git --version`.
2. Nếu chưa có WSL, mở PowerShell **Run as administrator**, chạy `wsl --install`, rồi khởi
   động lại Windows theo yêu cầu. Kiểm tra bằng `wsl --version` và `wsl --list --verbose`.
   Xem [hướng dẫn WSL của Microsoft](https://learn.microsoft.com/en-us/windows/wsl/install).
3. Tải [Docker Desktop cho Windows](https://docs.docker.com/desktop/setup/install/windows-install/),
   chọn backend WSL 2 khi được hỏi. Cài xong mở **Docker Desktop** từ Start menu, chờ engine
   chạy; lệnh `docker info --format '{{.OSType}}'` phải in `linux`.
4. Cài Ollama theo mục 6.2. Python và `uv` được bootstrap của repo cài local ở mục 4.

Sau bước WSL, các lệnh trong tài liệu chạy bằng PowerShell Windows bình thường, không cần mở
terminal Ubuntu hay giữ quyền Administrator cho toàn bộ quy trình.

## 3. Clone repo và kiểm tra ban đầu

Mở PowerShell, chọn thư mục làm việc rồi chạy:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

git clone https://github.com/pgtomvn/AIC2026.git
Set-Location .\AIC2026

git status --short
git branch --show-current
git log -1 --oneline
```

Nếu team đang thống nhất một tag/commit cụ thể, checkout đúng tag/commit đó sau `git clone`.
Không copy `.env`, `models/`, `data/`, Docker volume hoặc database từ máy khác vào máy mới;
mỗi thành viên phải có state local riêng. Nếu `git status` sau clone không sạch, dừng lại và
xác định đó là thay đổi có chủ ý trước khi tiếp tục.

Chạy preflight nhanh, chưa đọc toàn bộ checksum archive:

```powershell
.\scripts\preflight.ps1 -SkipArchiveHash
```

Các cảnh báo `archive absent`, thiếu Docker hoặc thiếu GPU ở bước này chỉ là cảnh báo kiểm kê.
Preflight thành công **không** có nghĩa là batch, model, index hay service đã sẵn sàng.
Khi muốn kiểm tra integrity toàn bộ archive đã catalog, chạy không có `-SkipArchiveHash`; thao
tác đó có thể đọc hàng chục/hàng trăm GiB và không nên chạy trong mỗi lần khởi động.

## 4. Cài Python, uv và hai runtime local

Repo dùng Python 3.11 và `uv` pinned 0.11.16. Script bootstrap tải `uv` vào `.tools/`, cài
Python local và tạo `.env` nếu chưa có. `.env` chứa key local ngẫu nhiên cho Qdrant/Meilisearch;
không gửi file này lên Git hay chat nhóm.

```powershell
.\scripts\bootstrap.ps1
.\scripts\check.ps1
```

Tiếp theo tạo runtime đầy đủ cho model/inference. Đây là bước bắt buộc cho launcher chính:

```powershell
.\scripts\bootstrap_runtime.ps1 -Profile laptop

$Python = (Resolve-Path .\.venv-runtime\Scripts\python.exe).Path
& $Python -c "import sys; print(sys.version)"
```

`bootstrap.ps1` tạo `.venv` cơ bản để test; `run_app.ps1` yêu cầu `.venv-runtime` cùng stamp
được tạo bởi `bootstrap_runtime.ps1`. Không dùng lẫn hai môi trường khi chạy script nặng.

Nếu máy có 12--24 GiB VRAM và team đã kiểm chứng profile đó, có thể dùng:

```powershell
.\scripts\bootstrap_runtime.ps1 -Profile accelerated
```

Profile accelerated bật cấu hình dùng thêm OpenCLIP lớn hơn; tải trọng số bằng bước
`prepare_models.py` ở mục 6.1. Profile này không được xem là đã kiểm chứng trên laptop 4 GiB
VRAM.

## 5. Cài và khởi động Qdrant/Meilisearch

Hai dịch vụ đã khai báo trong [`docker-compose.yml`](../docker-compose.yml). Bạn **không cần
tải riêng file `.exe` của Qdrant hay Meilisearch**, không cần mở Qdrant Cloud/Meilisearch Cloud.
Docker tải đúng image đã pin khi chạy lệnh bên dưới. Tại thời điểm viết, repo dùng Qdrant
`v1.18.2` và Meilisearch `v1.45.0`, kèm SHA-256 image; lấy file Compose làm nguồn chuẩn khi
team cập nhật phiên bản.

### 5.1. Chuẩn bị Docker và `.env`

Mở Docker Desktop, chờ engine sẵn sàng, rồi mở PowerShell:

```powershell
Set-Location D:\AIC2026
docker version
docker compose version
docker info --format '{{.OSType}}'
Test-Path .env
docker compose --env-file .env config --quiet
```

Kết quả cần thấy: `docker version` có cả Client và Server, loại engine là `linux`, `Test-Path`
trả về `True`, và `config --quiet` thoát thành công mà không in lỗi. Nếu thiếu `.env`, chạy
`bootstrap.ps1`. Nếu `.env` đã có nhưng key trống, bootstrap giữ nguyên file đó; cần điền key
local riêng cho `QDRANT_API_KEY` và `MEILISEARCH_MASTER_KEY` trước khi tiếp tục.

Cả hai key phải có ngay cả khi chỉ bật một service, vì Compose kiểm tra toàn bộ file cấu hình.
Dùng `config --quiet`; không chia sẻ output `docker compose config` đầy đủ vì có thể chứa key.

### 5.2. Ví dụ: chỉ tải và khởi động Meilisearch

```powershell
# Tải image một lần khi có mạng.
docker compose --env-file .env pull meilisearch
if ($LASTEXITCODE -ne 0) { throw "Khong tai duoc image Meilisearch." }

# Tạo/chạy container dưới nền; có thể đóng PowerShell sau lệnh này.
docker compose --env-file .env up -d meilisearch
if ($LASTEXITCODE -ne 0) { throw "Khong khoi dong duoc Meilisearch." }

docker compose --env-file .env ps -a meilisearch
Invoke-RestMethod -Uri http://127.0.0.1:7700/health -TimeoutSec 10
```

Container tên `aic2026-meilisearch` cần ở trạng thái `Up`/`healthy`. Health API trả về
`status: available` (PowerShell hiển thị thành bảng). Nếu mới hiện `health: starting`, đợi vài
giây rồi kiểm tra lại. `/health` không cần key; thành công chỉ xác nhận server hoạt động.
Xem [Meilisearch health API](https://www.meilisearch.com/docs/reference/api/health/get-health).

Xem log nếu container `Exited`, `Restarting` hoặc `unhealthy`:

```powershell
docker compose --env-file .env logs --tail 100 meilisearch
# Theo dõi log liên tục; Ctrl+C chỉ thoát theo dõi log, không dừng container.
docker compose --env-file .env logs -f --tail 100 meilisearch
```

Dữ liệu được ghi vào `docker_data/meilisearch/data/`. Mở
[Meilisearch local](http://127.0.0.1:7700) để kiểm tra giao diện development nếu cần; thao tác
được bảo vệ dùng `MEILISEARCH_MASTER_KEY` trong `.env`. Giao diện tìm kiếm AIC nằm ở cổng `8504`.

### 5.3. Ví dụ: chỉ tải và khởi động Qdrant

```powershell
docker compose --env-file .env pull qdrant
if ($LASTEXITCODE -ne 0) { throw "Khong tai duoc image Qdrant." }

docker compose --env-file .env up -d qdrant
if ($LASTEXITCODE -ne 0) { throw "Khong khoi dong duoc Qdrant." }

docker compose --env-file .env ps -a qdrant

# Đọc key vào biến, không in key ra màn hình/lịch sử lệnh.
$QdrantKey = (& .\.venv\Scripts\python.exe -c "from dotenv import dotenv_values; print(dotenv_values('.env').get('QDRANT_API_KEY') or '')").Trim()
if (-not $QdrantKey) { throw "Thieu QDRANT_API_KEY trong .env." }
Invoke-RestMethod -Uri http://127.0.0.1:6333/healthz `
  -Headers @{ 'api-key' = $QdrantKey } -TimeoutSec 10

# Xác nhận cả REST API và key hoạt động.
Invoke-RestMethod -Uri http://127.0.0.1:6333/collections `
  -Headers @{ 'api-key' = $QdrantKey } -TimeoutSec 10
```

Container tên `aic2026-qdrant` cần ở trạng thái `Up`/`healthy`; `/healthz` phải trả HTTP 200,
và `/collections` trả `status: ok`. Trên máy mới, danh sách collections rỗng là bình thường.
Healthcheck của container Qdrant trong repo chỉ thử mở cổng TCP; REST API ở trên kiểm tra thêm
khả năng trả lời và xác thực.

```powershell
docker compose --env-file .env logs --tail 100 qdrant
# Ctrl+C chỉ thoát theo dõi log.
docker compose --env-file .env logs -f --tail 100 qdrant
```

Có thể mở [Qdrant Dashboard](http://127.0.0.1:6333/dashboard) rồi nhập `QDRANT_API_KEY` khi được
yêu cầu. REST dùng `6333`; cổng `6334` dành cho gRPC, không mở bằng trình duyệt. Xem
[Qdrant local quickstart](https://qdrant.tech/documentation/quickstart/).

Dữ liệu nằm trong Docker volumes `aic2026-qdrant-storage` và `aic2026-qdrant-snapshots`.
Thư mục `docker_data/qdrant/` nếu còn trên máy là bản rollback cũ, không phải storage mà Compose
hiện tại đang mount. Giữ nguyên thư mục đó.

### 5.4. Bật cả hai và kiểm tra trước khi import

Sau khi hiểu từng service, có thể dùng một lệnh cho cả hai:

```powershell
docker compose --env-file .env up -d --wait --wait-timeout 120 qdrant meilisearch
if ($LASTEXITCODE -ne 0) { throw "Database chua san sang; xem ps va logs." }
docker compose --env-file .env ps -a
```

`up` tạo container nếu chưa có và chạy lại container đã dừng; `-d` chạy nền; `--wait` đợi
healthcheck, tối đa 120 giây. Lần đầu `up` cũng tự tải image còn thiếu, nên `pull` riêng ở trên
là để nhìn rõ bước tải. Xem [Docker Compose up](https://docs.docker.com/reference/cli/docker/compose/up/).

Xác nhận key Meilisearch có quyền xem indexes (khác với health không cần key):

```powershell
$MeiliKey = (& .\.venv\Scripts\python.exe -c "from dotenv import dotenv_values; print(dotenv_values('.env').get('MEILISEARCH_MASTER_KEY') or '')").Trim()
if (-not $MeiliKey) { throw "Thieu MEILISEARCH_MASTER_KEY trong .env." }
Invoke-RestMethod -Uri 'http://127.0.0.1:7700/indexes?limit=10' `
  -Headers @{ Authorization = "Bearer $MeiliKey" } -TimeoutSec 10
```

Máy mới có thể trả danh sách rỗng. Sau đó chạy kiểm tra repo:

```powershell
.\scripts\check.ps1 -IncludeCompose -IncludeServices
```

Lệnh này chạy lint/test trước rồi mới kiểm tra Compose và hai health API; có thể lâu hơn các
lệnh health riêng. Nó không tự khởi động database và không xác nhận đã import đủ corpus.

### 5.5. Dừng, chạy lại và giữ dữ liệu

Chọn lệnh tương ứng với việc cần làm, không chạy tuần tự cả bảng:

| Việc cần làm | Lệnh tại thư mục repo |
|---|---|
| Dừng riêng Meilisearch | `docker compose --env-file .env stop meilisearch` |
| Chạy lại Meilisearch đã có | `docker compose --env-file .env start meilisearch` |
| Khởi động lại Meilisearch | `docker compose --env-file .env restart meilisearch` |
| Dừng riêng Qdrant | `docker compose --env-file .env stop qdrant` |
| Chạy lại Qdrant đã có | `docker compose --env-file .env start qdrant` |
| Khởi động lại Qdrant | `docker compose --env-file .env restart qdrant` |
| Dừng cả hai, giữ database | `docker compose --env-file .env stop qdrant meilisearch` |
| Bật cả hai, kể cả lần đầu | `docker compose --env-file .env up -d qdrant meilisearch` |

`start` chỉ dùng được khi container đã được tạo; `restart` không áp dụng thay đổi biến môi
trường trong Compose. Khi chủ động thay key/config, dùng `up -d` để Compose áp dụng lại rồi
kiểm tra xác thực. Tắt database sẽ làm các workflow đang dùng database đó mất kết nối.

Compose chỉ bind loopback `127.0.0.1`. Không đổi thành `0.0.0.0` để chia sẻ dịch vụ ra mạng.
Không chạy `docker compose down -v`, xóa volume, reset Docker Desktop hoặc xóa
`docker_data/meilisearch/data/` để sửa lỗi thông thường: có thể mất database đã import.

Tham khảo cách đóng gói dịch vụ trong
[hướng dẫn Meilisearch Docker](https://www.meilisearch.com/docs/resources/self_hosting/getting_started/docker);
các lệnh của repo ở trên đã chọn sẵn image, key, cổng và storage phù hợp.

## 6. Cài model và công cụ local

### 6.1. Model Hugging Face/OpenCLIP/OCR/ASR/translation

Sau khi runtime đầy đủ đã tạo, chạy một lần có mạng:

```powershell
& $Python scripts/prepare_models.py
```

Lệnh mặc định theo profile `laptop` chuẩn bị các họ model được cấu hình: SigLIP2 native,
organizer OpenCLIP `ViT-B-32-quickgelu/openai`, Vietnamese embedding, translation vi-en,
production/reference Whisper và OCR. Nó tạo state dưới `models/` và cập nhật manifest checksum;
không commit thư mục này.

Có thể chạy chọn lọc nếu cần tiết kiệm thời gian, nhưng reference đầy đủ nên có ít nhất
organizer, translation, text và các model mà workflow team sử dụng:

```powershell
& $Python scripts/prepare_models.py `
  --only visual `
  --only organizer `
  --only text `
  --only translation `
  --only asr `
  --only ocr `
  --only reference-asr `
  --only reference-ocr
```

`florence` không nằm trong lần chuẩn bị mặc định; chỉ tải riêng khi team thực sự dùng task
Florence:

```powershell
& $Python scripts/prepare_models.py --only florence
```

Đừng dùng `latest` cho Ollama và đừng tự thay revision Hugging Face trong cấu hình. Các revision
đã pin để kết quả và checksum giữa các máy có thể so sánh được.

### 6.2. Ollama và Qwen Visual Q&A

Cài Ollama theo [hướng dẫn Windows chính thức](https://docs.ollama.com/windows), mở lại
PowerShell và kiểm tra `ollama --version`. Launcher cần Ollama được cài ngay cả khi lần mở UI
đó bạn chỉ định dùng KIS/TRAKE.

Nên tải model vào `models/ollama` ngay từ đầu để khớp nơi launcher sử dụng. Nếu Ollama tray app
đang chạy, chọn **Quit Ollama** ở system tray trước. Sau đó dùng hai cửa sổ PowerShell:

**Cửa sổ A — khởi động server Ollama và giữ cửa sổ mở:**

```powershell
Set-Location D:\AIC2026
New-Item -ItemType Directory -Force -Path .\models\ollama | Out-Null
$env:OLLAMA_MODELS = (Resolve-Path .\models\ollama).Path
$env:OLLAMA_HOST = "127.0.0.1:11434"
$env:OLLAMA_NO_CLOUD = "1"
ollama serve
```

`ollama serve` chạy foreground, nên phải dùng cửa sổ khác cho lệnh pull. Nếu báo cổng `11434`
đã bị chiếm, kiểm tra server đang chạy trước khi mở thêm server.

**Cửa sổ B — tải model vào server vừa mở và kiểm tra API:**

```powershell
Set-Location D:\AIC2026
$env:OLLAMA_HOST = "127.0.0.1:11434"
ollama pull qwen2.5vl:3b-q4_K_M
if ($LASTEXITCODE -ne 0) { throw "Chua tai xong model Qwen." }
ollama list
Invoke-RestMethod -Uri http://127.0.0.1:11434/api/version -TimeoutSec 10
(Invoke-RestMethod -Uri http://127.0.0.1:11434/api/tags -TimeoutSec 10).models |
  Select-Object name, digest
```

Danh sách phải có `qwen2.5vl:3b-q4_K_M`. Launcher không tự pull model; readiness yêu cầu đúng
tag, capability vision và digest bắt đầu bằng `fb90415cde1e`. Model 7B hoặc tag `latest` không
được coi là tương thích.

Nếu trước đây đã pull vào thư mục mặc định của Ollama, đổi `OLLAMA_MODELS` không tự chuyển
model sang thư mục mới. Hãy pull lại vào server dùng `models/ollama`. Nếu launcher tìm thấy
server đang chạy ở `11434`, nó tái sử dụng server đó; biến của terminal mới không đổi storage
của server cũ. Việc chọn storage phải thực hiện khi khởi động server.

Kiểm tra explicit, có hai lượt inference cold/warm và có thể tốn thời gian:

```powershell
$Python = (Resolve-Path .\.venv-runtime\Scripts\python.exe).Path
& $Python scripts/check_vlm_offline.py
```

Sau khi kiểm tra, có thể nhấn `Ctrl+C` ở cửa sổ A để dừng server thủ công; launcher ở mục 8 sẽ
tự mở lại server với `models/ollama`. Hoặc giữ cửa sổ A mở để launcher tái sử dụng.

`GEMINI_API_KEY` là cấu hình cloud tùy chọn trong `.env`; không cần key này để dùng Q&A local.
Chỉ gửi frame tới Gemini khi chủ động chọn và chấp thuận việc truyền ảnh trong phiên Q&A.

### 6.3. FFmpeg/ffprobe (chỉ cần cho raw/native/audio)

Reference import từ map/features/objects/metadata/keyframes không cần FFmpeg. Nếu chạy raw
refinement, native ingestion hoặc ASR audio, tải một Windows FFmpeg ZIP từ nguồn tin cậy và giữ
lại SHA-256 của nhà cung cấp. Không dùng một binary không rõ nguồn.

```powershell
$FfmpegArchive = Resolve-Path D:\installers\ffmpeg-release.zip
$TrustedFfmpegSha256 = "<SHA-256-64-ky-tu-tu-nguon-tin-cay>"

& $Python scripts/install_ffmpeg.py `
  --archive $FfmpegArchive `
  --sha256 $TrustedFfmpegSha256

& $Python scripts/freeze_assets.py models .tools/ffmpeg
```

Nếu chỉ chạy reference baseline, bỏ qua mục này.

## 7. Import reference: từ ZIP thành dữ liệu có thể tìm kiếm

Chỉ thực hiện sau khi đủ dữ liệu mục 1.2, database ở mục 5 đã chạy và model organizer đã được
chuẩn bị. Đây là bước nạp dữ liệu, cần thời gian và dung lượng; không chạy lại mỗi lần mở UI.

### 7.1. Kiểm tra pack trước khi ghi database

```powershell
Set-Location D:\AIC2026
$env:PYTHONUTF8 = "1"
$env:AIC_PROFILE = "laptop"
$Python = (Resolve-Path .\.venv-runtime\Scripts\python.exe).Path

# Đối chiếu tên với danh sách mục 1.2; phải có đủ 14 ZIP keyframe đúng batch.
Get-ChildItem -File .\Keyframes_*.zip | Select-Object Name, Length

# Validate cấu trúc/mapping; chưa ghi index.
& $Python scripts/import_reference.py
if ($LASTEXITCODE -ne 0) { throw "Reference validation chua dat; sua du lieu truoc khi import." }
```

Report phải có `videos: 873`, `frames: 177321`, `feature_rows: 177321`. Nếu thiếu file,
sai batch, hoặc số hàng map/features không khớp, bổ sung đúng nguồn dữ liệu rồi chạy lại;
không giảm `--expected-videos`/`--expected-frames` để bỏ qua lỗi của bộ B1 đầy đủ.

### 7.2. Verify encoder trên chính máy này

```powershell
& $Python scripts/import_reference.py --verify-encoder
if ($LASTEXITCODE -ne 0) { throw "Organizer encoder chua duoc xac minh." }
```

Bước này dùng 16 ảnh mẫu, yêu cầu median cosine tối thiểu `0.999`, rồi ghi
`data/state/organizer-clip-verification.json`. Chạy trước khi import vectors để batch có thể
được kích hoạt. Cấu hình mặc định dùng CUDA cho verification; máy không có CUDA cần chọn
`models.organizer_clip.verification_device: cpu` trong `config/settings.yaml` và tự verify
lại trên CPU, vẫn giữ nguyên model/threshold. Các model khác có yêu cầu thiết bị riêng;
đổi mục này không chuyển toàn bộ pipeline sang CPU.

### 7.3. Nạp vectors, objects, metadata và canonical frames

```powershell
& $Python scripts/import_reference.py `
  --batch-id aic25-b1-reference `
  --index-vectors `
  --index-objects `
  --index-metadata `
  --import-canonical
if ($LASTEXITCODE -ne 0) { throw "Import chua hoan tat; doc loi truoc khi khoi dong UI." }
```

Lệnh này ghi vectors vào Qdrant collection `aic25_b1_openclip_b32_quickgelu_openai_512d_v1`,
objects/metadata vào Meilisearch index `aic_metadata_v1`, và canonical records/mapping vào
`data/state/artifacts.sqlite3` cùng ledger `data/state/jobs.sqlite3`. Nó đọc ZIP tại chỗ,
không giải nén toàn corpus. Giữ nguyên đường dẫn ZIP sau khi import vì URI ảnh trỏ tới đó.

Report cuối cần có `organizer_encoder_verified: true`, `batch_active: true`; canonical gồm
`mapping_rows: 177321` và `canonical_frames: 176707`. Hai số khác nhau vì một số ordinal cùng
trỏ tới một source frame. Sau import vẫn cần kiểm tra readiness ở mục 8; batch active riêng
lẻ chưa xác nhận tất cả workflow/model đều dùng được.

Nếu lần trước đã nạp vectors thành công nhưng chưa verify encoder/kích hoạt batch, làm mục 7.2
rồi dùng lệnh sau để kiểm tra vectors đang có và bổ sung canonical state:

```powershell
& $Python scripts/import_reference.py `
  --batch-id aic25-b1-reference `
  --import-canonical --activate-existing-vectors
if ($LASTEXITCODE -ne 0) { throw "Vectors hien co chua dat dieu kien kich hoat batch." }
```

Lệnh sửa này không thay thế bước import objects/metadata nếu chúng còn thiếu. Không dùng
`--recreate` trong cài đặt hoặc khởi động thông thường vì cờ đó xóa/tạo lại collection đích.

## 8. Kiểm tra readiness và khởi động toàn bộ ứng dụng

### 8.1. Kiểm tra trước khi mở UI

Database và server Ollama phải đang chạy khi dùng doctor đầy đủ:

```powershell
Set-Location D:\AIC2026
$Python = (Resolve-Path .\.venv-runtime\Scripts\python.exe).Path
& $Python scripts/runtime_doctor.py --batch-id aic25-b1-reference --mode app
```

Đọc trạng thái `kis`, `qa`, các `checks`/channel và `reason_code` trong JSON. Exit code `0`
là đạt kiểm tra ở chế độ này; `1` là bị chặn/thiếu prerequisite; `2` là còn degraded nếu chưa
cho phép degraded. Doctor không tự tải model hoặc import dữ liệu. TRAKE còn phải hoàn thành
warm-up trajectory cache trong ứng dụng trước khi tìm kiếm.

Không dùng `--startup-only` làm bằng chứng hệ thống đã sẵn sàng tìm kiếm: cờ đó chỉ kiểm tra
hợp đồng launcher/batch, hoãn kiểm tra chi tiết tới warm-up.

### 8.2. Mở FastAPI và Streamlit bằng một launcher

```powershell
.\scripts\run_app.ps1 `
  -Profile laptop `
  -BatchId aic25-b1-reference `
  -ApiPort 8080 `
  -Port 8504
```

Giữ cửa sổ PowerShell này mở. Launcher dùng `.venv-runtime`, tạo FastAPI ở nền và chạy
Streamlit ở foreground. Nó tái sử dụng Ollama đang chạy hoặc khởi động Ollama nếu cổng còn
trống. Với lệnh trên, bạn phải bật Qdrant/Meilisearch trước ở mục 5.

Mở [giao diện AIC](http://127.0.0.1:8504) trong trình duyệt. Có thể xem
[FastAPI docs](http://127.0.0.1:8080/docs) để kiểm tra endpoint. UI có thể mở trước khi warm-up
xong; đợi trạng thái workflow cho phép tìm kiếm. Lượt khởi động đầu có thể phải đọc index,
nạp model và tạo cache TRAKE.

Nếu muốn launcher chạy lệnh bật cả hai database, dùng biến thể sau **thay cho** lệnh trên:

```powershell
.\scripts\run_app.ps1 -Profile laptop -BatchId aic25-b1-reference -StartServices
```

Docker Desktop vẫn phải được mở từ trước. `-StartServices` gọi Compose rồi kiểm tra readiness
đầy đủ với thời gian chờ 60 giây; thiếu index/model vẫn có thể làm launcher dừng. Lệnh này
không cài Docker, tải model hoặc import corpus.

### 8.3. Kiểm tra ứng dụng đang chạy

Mở PowerShell thứ hai và chạy:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8080/healthz -TimeoutSec 10
Invoke-RestMethod -Uri http://127.0.0.1:8504/_stcore/health -TimeoutSec 10
Invoke-RestMethod -Uri http://127.0.0.1:8080/v1/runtime/warmup -TimeoutSec 30 |
  ConvertTo-Json -Depth 12
Invoke-RestMethod -Uri 'http://127.0.0.1:8080/v1/readiness?batch_id=aic25-b1-reference' -TimeoutSec 120 |
  ConvertTo-Json -Depth 12
```

Hai health endpoint chỉ cho biết tiến trình đang trả lời. Đánh giá readiness và warm-up theo
từng workflow: `warming` là đang chuẩn bị; `ready` là sẵn sàng trong phạm vi check; `degraded`
là chỉ một phần kênh hoạt động; `blocked`/`failed` cần xử lý nguyên nhân hiển thị. Không coi
màn hình UI mở được hoặc HTTP 200 là đã hoàn thành nạp dữ liệu.

Thử thủ công sau khi workflow sẵn sàng:

- **Textual KIS:** chọn batch `aic25-b1-reference`, nhập một mô tả cảnh, kiểm tra ảnh và video
  của kết quả. Câu mẫu chỉ là truy vấn thử, không bảo đảm có cảnh tương ứng trong corpus.
- **Visual Q&A:** tìm ứng viên bằng mô tả sự kiện, chọn một frame rồi đặt câu hỏi về đúng ảnh
  đó; dùng provider local khi kiểm tra Ollama.
- **TRAKE:** nhập mô tả chung và ít nhất hai giai đoạn có thứ tự, kiểm tra timeline kết quả.

Muốn dừng ứng dụng, nhấn `Ctrl+C` trong cửa sổ launcher. Launcher dọn FastAPI và Ollama do
chính nó tạo; Ollama đã chạy sẵn và hai container Docker vẫn tiếp tục chạy. Dùng mục 5.5 nếu
muốn dừng database.

## 9. Khởi động hằng ngày sau khi đã cài/import xong

1. Mở Docker Desktop từ Start menu và chờ engine sẵn sàng.
2. Mở PowerShell tại repo, chạy khối lệnh sau.
3. Giữ terminal mở, truy cập [http://127.0.0.1:8504](http://127.0.0.1:8504), đợi warm-up.

```powershell
Set-Location D:\AIC2026
Set-ExecutionPolicy -Scope Process Bypass
$env:PYTHONUTF8 = "1"

docker compose --env-file .env up -d --wait --wait-timeout 120 qdrant meilisearch
if ($LASTEXITCODE -ne 0) { throw "Database chua san sang; kiem tra Docker va logs." }

.\scripts\run_app.ps1 -Profile laptop -BatchId aic25-b1-reference -ApiPort 8080 -Port 8504
```

Launcher tự xử lý Ollama như mục 8.2; model phải đã pull vào đúng storage ở mục 6.2. Không cần
bootstrap, pull model, giải nén hay import lại trong mỗi lần khởi động. Khi team thay `uv.lock`
hoặc profile và launcher báo runtime stale, chạy lại `bootstrap_runtime.ps1 -Profile laptop`
trước khi mở app.

Kết thúc buổi làm việc: `Ctrl+C` trong launcher, rồi chạy lệnh sau nếu muốn dừng database:

```powershell
docker compose --env-file .env stop qdrant meilisearch
```

## 10. Lỗi thường gặp và cách kiểm tra

| Hiện tượng | Kiểm tra và xử lý |
|---|---|
| PowerShell chặn `.ps1` | Chạy `Set-ExecutionPolicy -Scope Process Bypass` trong terminal hiện tại. |
| `docker` không được nhận diện | Cài Docker Desktop, mở lại PowerShell để nhận PATH, thử `docker version`. |
| Có Docker Client nhưng không kết nối được Server/daemon | Mở Docker Desktop, đợi engine chạy; kiểm tra WSL 2 và `docker info`. |
| `no configuration file provided` | Chuyển vào repo chứa `docker-compose.yml`, ví dụ `Set-Location D:\AIC2026`. |
| Compose báo thiếu `QDRANT_API_KEY` hoặc `MEILISEARCH_MASTER_KEY` | Kiểm tra `.env` ở repo có cả hai key không rỗng. Bootstrap không ghi đè `.env` cũ. |
| Pull image lỗi mạng/proxy | Kiểm tra mạng của Docker Desktop và thử lại `docker compose --env-file .env pull qdrant meilisearch`; giữ nguyên tag/digest đã pin. |
| Container `Exited`, `Restarting`, `unhealthy` | Dùng `docker compose --env-file .env ps -a` và `logs --tail 100 <service>`; xử lý lỗi cụ thể trong log. |
| Cổng `6333`, `6334`, `7700` đã bị chiếm | Kiểm tra lệnh bên dưới để xác định owner; tránh khởi động thêm một container/binary trùng cổng. |
| HTTP 401/403 | Qdrant dùng header `api-key`; Meilisearch dùng `Authorization: Bearer ...`. Kiểm tra key client khớp `.env` đang cấp cho container; sau khi chủ động đổi key, áp dụng lại bằng `up -d`. |
| `Competition runtime is missing/stale` hoặc sai profile | Chạy `.\scripts\bootstrap_runtime.ps1 -Profile laptop`; không chạy launcher bằng `.venv` cơ bản. |
| `Ollama is not installed` | Cài Ollama Windows, mở lại terminal và kiểm tra `ollama --version`. |
| Ollama báo `model not found`/sai digest | Kiểm tra `ollama list`, đúng tag và server/storage ở mục 6.2; không thay bằng `latest`. |
| CUDA lỗi khi verify organizer | Kiểm tra driver và cấu hình `verification_device` ở mục 7.2; không bỏ qua verification gate. |
| Doctor thiếu organizer verification/batch inactive | Hoàn thành mục 7.2 rồi import hoặc activate existing vectors như mục 7.3. |
| Database healthy nhưng kết quả rỗng/thiếu collection/index | Kiểm tra đã import mục 7.3, đúng batch và đúng endpoint; database mới khởi động ban đầu chưa có corpus. |
| QA thiếu canonical images hoặc ảnh lỗi sau khi di chuyển ZIP | Kiểm tra đủ ZIP keyframe và đường dẫn đã import; khôi phục vị trí đã ghi trong URI, giữ nguyên archive được bảo vệ. |
| UI mở nhưng nút tìm kiếm còn khóa | Kiểm tra `/v1/runtime/warmup` và readiness; nếu trạng thái lỗi, đọc `reason_code`, sửa prerequisite rồi khởi động lại launcher. |
| Cổng `8080`/`8504` đã có listener | Dừng launcher cũ bằng `Ctrl+C`; nếu chủ động chạy stack riêng, chọn `-ApiPort 8081 -Port 8505` và mở URL mới. |

Kiểm tra tiến trình nào đang nghe trên các cổng, chỉ đọc thông tin:

```powershell
Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
  Where-Object { $_.LocalPort -in @(6333, 6334, 7700, 8080, 8504, 11434) } |
  Select-Object LocalAddress, LocalPort, OwningProcess
# Thay 1234 bằng OwningProcess vừa thấy.
Get-Process -Id 1234
```

Không dừng hàng loạt tiến trình Python/Docker để sửa xung đột cổng. Xác định đúng ứng dụng
đang dùng cổng rồi dừng từ terminal hoặc giao diện của ứng dụng đó.

Nếu FastAPI không lên, đọc log launcher (đổi `8080` nếu dùng `-ApiPort` khác):

```powershell
Get-Content .\.cache\run\fastapi-8080.err.log -Tail 100
Get-Content .\.cache\run\fastapi-8080.out.log -Tail 100
```

Khi gửi lỗi cho team, kèm lệnh đã chạy, commit, profile, batch, thông báo lỗi và trạng thái
service; che key/token trước khi chia sẻ. Không gửi `.env`, dump cấu hình có secret hoặc
database/model lớn vào Git.
