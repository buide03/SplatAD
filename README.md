# Thực hành SplatAD trên Google Colab với PandaSet

## Bối cảnh và mục tiêu

Xe tự hành ghi lại cảnh đường phố bằng camera và LiDAR. Bài thực hành dựng lại cảnh ở từng thời điểm rồi đối chiếu ảnh gốc và ảnh render để quan sát cách SplatAD xử lý người đi bộ, người đi xe đạp và đối tượng bị che khuất một phần. Kết quả dự kiến gồm RGB, depth, point cloud LiDAR và một trang web so sánh các frame giữ lại.

## Dataset và thuật toán

**PandaSet** có ảnh camera, pose và hiệu chuẩn, điểm LiDAR và hộp giới hạn 3D theo frame. Mỗi đối tượng giữ một `uuid` qua các frame. Dự án dùng nhãn gốc, với bước kiểm duyệt và bổ sung khi có bằng chứng trên ảnh và LiDAR. [Bản tải PandaSet](https://huggingface.co/datasets/georghess/pandaset) · [Schema annotation](https://scaleapi.github.io/pandaset-devkit/annotations.html)

**SplatAD** dùng 3D Gaussian Splatting để biểu diễn nền tĩnh và các actor động rồi render cả camera và LiDAR. Ta chạy bản phát hành của tác giả trong **Nerfstudio qua `neurad-studio`**, với parser `pandaset-data`, datamanager RGB/LiDAR và `ns-train`/`ns-render` có sẵn. Actor động dùng pose cứng, nên bài thực hành xem kỹ chuyển động của tay chân, người và xe đạp, cùng hình dạng tại vùng che khuất. [Bài báo](https://arxiv.org/abs/2411.16816) · [NeuRAD Studio](https://github.com/georghess/neurad-studio) · [Fork gsplat](https://github.com/carlinds/splatad)

## Chạy notebook

1. Tải `SplatAD_Colab.ipynb` lên Colab và đặt `SplatAD-project.zip` vào `My Drive/SplatAD/project/`. Nếu dùng bản clone đã có trên Colab, đặt `PROJECT_DIR` đến thư mục chứa `pandaset_data.py`.
2. Chọn **CPU**, giữ `PHASE = "prepare"`, chạy các cell Task 1–5. Colab yêu cầu cấp quyền mount Drive. Notebook tải mặc định qua HTTP Range. Nếu dùng ZIP riêng đã tải, upload `pandaset_011.zip` vào `My Drive/SplatAD/datasets/` và đặt `SCAN_PAPER_SEQUENCES = False`. ZIP cần chứa `pandaset/011/…`. Có thể trỏ `LOCAL_FULL_ZIP` tới một bản `pandaset.zip` đầy đủ trên Drive để giữ bước quét.
3. Khi notebook báo đã lưu bản duyệt, đổi runtime sang **T4 GPU** và đặt `PHASE = "train"`, `TRAIN_STAGE = "smoke"`. Chạy lại từ đầu. Task 7 chạy thử 500 bước, xuất kết quả và kết thúc lượt chạy thử.
4. Khi muốn train chính, chuyển sang **A100**, giữ `PHASE = "train"`, đặt `TRAIN_STAGE = "main"` và chạy lại từ đầu để train 10.000 bước, render và đóng gói demo. Môi trường cài đặt theo mốc Python 3.10, PyTorch 2.0.1 và CUDA 11.8 của repo tác giả. Khi phiên ngắt giữa một lượt train, dùng cùng dữ liệu, chọn lại lượt đó và bật `RESUME` để đọc checkpoint gần nhất. [Recipe của tác giả](https://github.com/georghess/neurad-studio/blob/main/apptainer_recipe) · [Colab FAQ](https://research.google.com/colaboratory/faq.html)

Task 6 cài NumPy 1.26.4 cùng PyTorch. Task 7 chọn MLP PyTorch có sẵn cho bộ giải mã LiDAR của SplatAD. Rasterization camera và LiDAR chạy qua `gsplat` CUDA. [Cấu hình SplatAD](https://github.com/georghess/neurad-studio/blob/aea389a936890eb48e1b43a493a52c0194559b48/nerfstudio/models/splatad.py#L240)

### Các ca mặc định trong sequence `011`

| Mức | Ca | Camera | Frame đánh dấu |
|---|---|---|---|
| Dễ | Một người đi bộ thấy rõ | `front_right` | `67`, `71`, `77` |
| Dễ | Một người đi xe đạp thấy rõ | `front` | `25`, `27`, `29` |
| Vừa | Một người chạy bộ bị thùng rác che phần chân | `right` | `17`, `19`, `21` |
| Vừa | Người đi xe đạp đi qua vùng có cột và biển chắn tầm nhìn | `front` | `17`, `19`, `23` |
| Khó | Hai người đi song song và che nhau, mỗi người có UUID riêng | `front_right` | `01`, `05`, `09` |

Các UUID và khoảng frame nằm trong `cases.json`. Nếu Drive đã có bộ ca cũ, đặt `USE_PROJECT_CASES = True` và chạy Task 2 để lưu bộ ca mới. Khi dùng phiên Colab đã giải nén gói cũ, khởi động lại phiên và chọn một `PROJECT_DIR` mới để nạp đúng mã nguồn. Kiểm tra lại split ở Task 6 trước khi train.

Nếu bộ ca mới đã nằm trên Drive và bạn đã chạy Task 4, hãy chạy ô thiết lập với `PHASE = "prepare"`, rồi đến thẳng Task 5. Trong runtime mới, Task 5 tự nạp ZIP `011` và áp dụng lại các chỉnh sửa đã lưu trên Drive.

Nếu cập nhật ZIP đã duyệt, mở runtime GPU mới để nạp dữ liệu mới và đổi `RUNS` trong notebook sang thư mục khác để tách checkpoint theo phiên bản annotation.

## Chín task của pipeline

| Task | Xử lý | Đầu ra |
|---|---|---|
| **1. Chọn cảnh** | Quét cuboid của các sequence bài báo, tìm nhãn `Walking` và `With Rider`, xem ảnh để kiểm tra các ca và che khuất. Bật `SCAN_MORE` nếu cần quét phần còn lại. | `candidate_tracks.json` và `cases.json` |
| **2. Tải dữ liệu** | Đọc mục lục ZIP từ xa và lấy đủ tệp của sequence đã chọn. Nguồn dự phòng là ZIP đã tải sẵn. | `datasets/pandaset_<ID>.zip` |
| **3. Nạp vào Colab** | Chép ZIP từ Drive vào ổ runtime, giải nén, kiểm tra tên file, timestamps và phép chiếu Pandar64. Phase GPU dùng bản đã duyệt ở `/content/data/pandaset`. | Ảnh overlay và cây dữ liệu đã kiểm tra |
| **4. Rà soát hộp 3D** | Kiểm tra tâm, kích thước và hướng hộp trên ảnh và LiDAR. Nếu cần, thêm hoặc chỉnh hộp bằng `review/edits_<ID>.json` theo mẫu trong notebook. | Overlay sau rà soát và nhật ký `audit_<ID>.json` |
| **5. Rà soát tracking** | Giữ cùng UUID cho cùng đối tượng. Kiểm tra khoảng trống, bước nhảy và ID. Chỉ sửa nhãn khi có bằng chứng. | CSV theo track, clip overlay có ID, ZIP đã duyệt trong `review/datasets/` |
| **6. Audit parser Nerfstudio** | `pandaset-data` đọc ảnh, pose, LiDAR và cuboid; ghép trajectory và chia 50% train. Bật `--include-deformable-actors True` để nạp người đi bộ. | `split.json` với frame, scan và trajectory sau parser |
| **7. Train SplatAD** | `ns-train splatad … pandaset-data …`. Chạy thử 500 bước với một camera, Pandar64 và ảnh nửa độ phân giải, rồi chạy chính 10.000 bước mỗi sequence. | `config.yml`, checkpoint và log dưới `runs/` |
| **8. Render** | `ns-render dataset` xuất `gt-rgb`, `rgb`, `depth` và `raw-depth` trên frame giữ lại. API LiDAR của SplatAD xuất PLY thật và dự đoán trong cùng hệ tọa độ. | PNG, `.npy.gz`, PLY và `scans.json` |
| **9. Demo web** | Ghép cặp theo camera và frame. Đóng gói PNG, MP4 và manifest cho các ca đã chọn. Ghi nhận xét sau khi xem ảnh thật và ảnh dựng. | `My Drive/SplatAD/demo.zip` |

## Xem kết quả Nerfstudio

**Đối chiếu cùng camera và thời điểm.** Task 8 nạp checkpoint bằng `ns-render dataset` và xuất cặp PNG `gt-rgb` / `rgb` từ các frame giữ lại. Notebook hiển thị trực tiếp các ảnh đó. Demo web dùng bố cục lấy cảm hứng từ Viser, với bảng cảnh và đầu ra bên phải, thanh thời gian và vùng đối chiếu ở giữa. Demo đọc cùng cặp PNG, cho phép đổi RGB/depth và cung cấp video MP4 để trình chiếu.

**Đánh giá RGB.** Task 8 tính **PSNR** (dB, cao hơn tốt hơn), **SSIM** (cao hơn tốt hơn) và **LPIPS** (thấp hơn tốt hơn) bằng hàm đánh giá của SplatAD trong Nerfstudio. `renders/evaluation.json` chứa điểm từng ảnh và trung bình trên toàn bộ tập giữ lại của sequence. Viewer hiển thị điểm từng frame và trung bình riêng của tình huống đang xem. Các điểm tính trên toàn ảnh RGB trước khi mã hóa MP4, kể cả khi viewer đang phóng to hoặc hiển thị depth.

**Khám phá cảnh 3D tương tác.** Dùng viewer Viser có sẵn trong `neurad-studio` trên máy Linux có GPU và cùng phiên bản môi trường. Chép `config.yml` của run và thư mục `nerfstudio_models` đi kèm, rồi giải nén đúng bản PandaSet đã duyệt. Từ repo này chạy:

```bash
python colab_task.py viewer /path/to/run/config.yml --data /path/to/pandaset
```

Mở [Viser trên localhost](http://127.0.0.1:7007). Lệnh trên cập nhật đường dẫn trong một cấu hình tạm rồi gọi viewer chính thức. Mặc định chọn checkpoint mới nhất. Thêm `--step` với số bước trong `renders/lidar/scans.json` để nạp đúng checkpoint của demo nếu thư mục có nhiều checkpoint. Viser cho phép đổi góc nhìn, thời điểm và độ phân giải. Dùng PNG từ `ns-render dataset` để đánh giá tại pose gốc. [Viewer chính thức của NeuRAD Studio](https://github.com/georghess/neurad-studio/blob/aea389a936890eb48e1b43a493a52c0194559b48/nerfstudio/scripts/viewer/run_viewer.py)

### Mở demo ảnh và video

Tải `demo.zip` từ Drive sau Task 9, giải nén vào `demo` rồi chạy từ thư mục cha:

```bash
python -m http.server 8000 --directory demo
```

Mở [viewer trên localhost](http://localhost:8000). Chọn một tình huống, tua hoặc phát hai ảnh đồng bộ, rồi bật phóng to để xem cùng vùng đối tượng. Video MP4 có thể tải riêng. Depth thô và PLY nằm trong `runs/` trên Drive.

Ở ca che khuất, xem ba mốc liên tiếp trước khi nhận xét. Đối chiếu overlay và ảnh gốc để kiểm tra lỗi hộp, track hoặc căn chỉnh trước khi quy hiện tượng nhòe cho giả định actor cứng. Chỉ ghi lỗi đã quan sát vào `OBSERVATIONS` trong notebook.
