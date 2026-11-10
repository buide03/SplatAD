# Thực hành SplatAD trên Google Colab với PandaSet

## Bối cảnh và mục tiêu

Xe tự hành ghi lại cảnh đường phố bằng camera và LiDAR. Bài thực hành dựng lại cảnh ở từng thời điểm rồi đối chiếu ảnh gốc và ảnh render để quan sát cách SplatAD xử lý người đi bộ, người đi xe đạp và đối tượng bị che khuất một phần. Kết quả dự kiến gồm RGB, depth, point cloud LiDAR và một trang web so sánh các frame giữ lại.

## Dataset và thuật toán

**PandaSet** có ảnh camera, pose và hiệu chuẩn, điểm LiDAR và hộp giới hạn 3D theo frame. Mỗi đối tượng giữ một `uuid` qua các frame. Dự án dùng nhãn gốc, với bước kiểm duyệt và bổ sung khi có bằng chứng trên ảnh và LiDAR. [Bản tải PandaSet](https://huggingface.co/datasets/georghess/pandaset) · [Schema annotation](https://scaleapi.github.io/pandaset-devkit/annotations.html)

**SplatAD** dùng 3D Gaussian Splatting để biểu diễn nền tĩnh và các actor động rồi render cả camera và LiDAR. Ta chạy bản phát hành của tác giả trong **Nerfstudio qua `neurad-studio`**, với parser `pandaset-data`, datamanager RGB/LiDAR và `ns-train`/`ns-render` có sẵn. Actor động dùng pose cứng, nên bài thực hành xem kỹ chuyển động của tay chân, người và xe đạp, cùng hình dạng tại vùng che khuất. [Bài báo](https://arxiv.org/abs/2411.16816) · [NeuRAD Studio](https://github.com/georghess/neurad-studio) · [Fork gsplat](https://github.com/carlinds/splatad)

## Chạy notebook

1. Tải `SplatAD_Colab.ipynb` lên Colab và đặt `SplatAD-project.zip` vào `My Drive/SplatAD/project/`. Nếu dùng bản clone đã có trên Colab, đặt `PROJECT_DIR` đến thư mục chứa `pandaset_data.py`.
2. Chọn **CPU**, giữ `PHASE = "prepare"`, chạy các cell Task 1–5. Colab yêu cầu cấp quyền mount Drive. Notebook tải mặc định qua HTTP Range. Nếu dùng ZIP riêng đã tải, upload `pandaset_011.zip` vào `My Drive/SplatAD/datasets/` và đặt `SCAN_PAPER_SEQUENCES = False`. ZIP cần chứa `pandaset/011/…`. Có thể trỏ `LOCAL_FULL_ZIP` tới một bản `pandaset.zip` đầy đủ trên Drive để giữ bước quét.
3. Khi notebook báo đã lưu bản duyệt, đổi runtime sang **GPU** và đặt `PHASE = "train"`. Chạy lại từ đầu để mount Drive và thực hiện Task 6–9. Môi trường cài đặt theo mốc Python 3.10, PyTorch 2.0.1 và CUDA 11.8 của repo tác giả. Chọn T4, L4 hoặc A100 nếu có. Khi phiên ngắt giữa lúc train, dùng cùng dữ liệu và bật `RESUME` để đọc checkpoint gần nhất. [Recipe của tác giả](https://github.com/georghess/neurad-studio/blob/main/apptainer_recipe) · [Colab FAQ](https://research.google.com/colaboratory/faq.html)

### Các ca mặc định đã đối chiếu trên ảnh gốc

| Ca | Sequence | Camera | Frame đánh dấu |
|---|---|---|---|
| Người đi bộ | `011` | `front_right` | `01`, `05`, `09` |
| Người đi xe đạp | `011` | `front` | `21`, `25`, `29` |
| Người đi bộ bị cột đỏ che một phần | `011` | `front_right` | `13` trước, `21` đang bị che, `23` xuất hiện trở lại |

Các UUID và khoảng frame nằm trong `cases.json`. Muốn đổi ca, sửa bản `My Drive/SplatAD/review/cases.json` sau khi đối chiếu annotation và ảnh gốc. Kiểm tra lại split ở Task 6 trước khi train.

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
| **9. Demo web** | Ghép cặp theo camera và frame. Đóng gói PNG, MP4 và manifest cho ba ca. Ghi nhận xét sau khi xem ảnh thật và ảnh dựng. | `My Drive/SplatAD/demo.zip` |

## Xem demo

Tải `demo.zip` từ Drive sau Task 9, giải nén vào `demo` rồi chạy từ thư mục cha:

```bash
python -m http.server 8000 --directory demo
```

Mở [viewer trên localhost](http://localhost:8000). Chọn một tình huống, tua hoặc phát hai ảnh đồng bộ, rồi bật phóng to để xem cùng vùng đối tượng. Video MP4 có thể tải riêng. Depth thô và PLY nằm trong `runs/` trên Drive.

Ở ca che khuất, xem ba mốc liên tiếp trước khi nhận xét. Đối chiếu overlay và ảnh gốc để kiểm tra lỗi hộp, track hoặc căn chỉnh trước khi quy hiện tượng nhòe cho giả định actor cứng. Chỉ ghi lỗi đã quan sát vào `OBSERVATIONS` trong notebook.
