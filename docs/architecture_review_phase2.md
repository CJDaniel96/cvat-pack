# Phase 1 + Phase 2 架構審查報告

日期：2026-07-02
範圍：`src/cvat_packer/` 全部程式碼、`tests/` 全部測試、CLI 手動驗證

## 1. 目前架構摘要

```
cvat_packer/
  cli.py                  # argparse 入口，不含任何格式專屬邏輯
  core/
    config.py              # PackConfig（dataclass，統一輸入參數）
    models.py               # ValidationReport / PackageResult / Status
    registry.py              # @register 裝飾器 + get_adapter() / list_formats()
    archive.py                # build_zip() / list_zip_entries()（唯一的 zip 產生點）
    filesystem.py               # iter_files/safe_copy/safe_copytree/junk filtering
    errors.py                     # CvatPackerError 階層
    report.py                      # format_summary() / write_manifest()
  formats/
    base.py                 # FormatAdapter（抽象基底）+ SkeletonFormatAdapter
    coco.py / yolo.py / pascal_voc.py / cvat_image.py / cvat_video.py
    coco_keypoints.py / ultralytics_yolo.py（Detection/Segmentation/Pose/OBB/Classification）
    <15 個 Phase 3 stub 格式，皆繼承 SkeletonFormatAdapter>
  validators/
    common.py                # find_images() / unmatched_stems() / check_required_paths()
    annotation.py              # bbox / normalized-coordinate 泛用檢查
    image.py                     # Pillow-optional 影像尺寸檢查
    zip_structure.py               # 產出後的 zip 結構驗證
  utils/
    path_normalizer.py       # to_posix() / is_safe_relative_path() / basename()
    hashing.py / logging.py
```

**資料流**：`CLI → PackConfig → registry.get_adapter() → adapter.validate() → ValidationReport
→（--dry-run/--validate-only 則到此為止）→ adapter.build_package() → archive.build_zip()`。

**驗證結果**（本次審查手動執行）：
- `cvat-pack --help` ✅ 正常顯示所有選項
- `--format {coco,yolo,ultralytics-yolo-detection,pascal-voc,cvat-image}` 搭配 `--dry-run` 全數 exit 0 ✅
- `coco` 格式實際輸出 zip（無 `--dry-run`）✅，zip 內容為 `annotations/instances_default.json` + `images/*.jpg`
- Phase 3A 五個格式（見下）以 fixture 資料跑過 `validate → build_package`，皆成功產生 zip ✅

## 2. 已完成格式清單

Registry 目前共 **28 個格式**（`cvat-pack --list-formats` 可即時列出）。

| 分類 | 格式 | 備註 |
|---|---|---|
| 完整實作（Full） | COCO 1.0 | Phase 1 |
| | YOLO 1.0 | Phase 1 |
| | Pascal VOC 1.0 | Phase 1 |
| | CVAT for images 1.1 | Phase 1 |
| | CVAT for video 1.1 | 本次審查才發現已存在完整實作（非 stub），但使用者的 Phase 1+2 清單未提及 |
| | COCO Keypoints 1.0 | **本次審查發現已存在**，非本次新增；已依 Phase 3A 規格補強驗證 |
| | Ultralytics YOLO Detection/Segmentation/Pose/OBB/Classification 1.0（5 種） | **本次審查發現已存在**，非本次新增；已依 Phase 3A 規格補強驗證 |
| 結構驗證（Phase 3 stub，`SkeletonFormatAdapter`） | Datumaro, Segmentation Mask, ImageNet, WIDER Face, CamVid, VGGFace2, Market-1501, ICDAR×3, Open Images, Cityscapes, KITTI, LFW, MOT, MOTS PNG, LabelMe（15 種） | 僅做檔案存在性檢查，內容驗證留待未來 batch |

> **重要說明**：使用者原始需求把 COCO Keypoints 與 4 個 Ultralytics YOLO 子格式列為「Phase 3A 待新增」，但實際檢查程式碼後發現這 5 個 adapter **早已完整實作**並註冊在 registry 中（`coco_keypoints.py`、`ultralytics_yolo.py`）。真正缺少的是**測試 fixtures 與測試檔**（`tests/fixtures/coco_keypoints/` 等 5 組目錄、`tests/test_coco_keypoints.py` 等 5 個測試檔在審查前完全不存在，連已有 fixture 的 Ultralytics YOLO Detection 也沒有專屬測試檔）。本次已將這個落差補齊（見第 8 節與後續 Phase 3A 完成報告）。

## 3. 每個格式目前支援的能力

| 格式 | supported_tasks | supported_shapes | attributes | tracks | 驗證深度 |
|---|---|---|---|---|---|
| coco | detection, instance_segmentation | bbox, polygon, rle | ✓ | ✗ | 完整：image_id/category_id 對應、bbox 正值、polygon/RLE 結構、keypoints 長度 |
| coco-keypoints | keypoints | keypoints, skeleton | ✓ | ✗ | 完整（繼承 coco + 本次新增 num_keypoints 一致性、visibility∈{0,1,2}、bbox 必要性） |
| yolo | detection | bbox | ✗ | ✗ | 完整：class id 範圍、5 欄位、normalized [0,1]、image/label 配對 |
| ultralytics-yolo-detection | detection | bbox | ✗ | ✗ | 完整：同 yolo，但依 split（train/val/test）分別驗證 |
| ultralytics-yolo-segmentation | instance_segmentation | polygon | ✗ | ✗ | 完整：多邊形至少 3 點、(x,y) 成對、class id 範圍 |
| ultralytics-yolo-pose | keypoints | skeleton | ✗ | ✗ | 完整（本次修正）：依 data.yaml `kpt_shape` 驗證關鍵點數量與 visibility∈{0,1,2} |
| ultralytics-yolo-obb | oriented_detection | rotated_bbox | ✗ | ✗ | 完整：恰好 4 個角點（9 欄位）、normalized |
| ultralytics-yolo-classification | classification | （無 shape，純分類） | ✗ | ✗ | 完整：split/class 資料夾結構、空 class 警告、副檔名檢查（本次新增） |
| pascal-voc | detection, segmentation | bbox, polygon, mask | ✓ | ✗ | 完整：XML 結構、bndbox 座標合理性、image/annotation 配對 |
| cvat-image | detection, segmentation, keypoints | 全部 CVAT shape tag | ✓ | ✓ | 完整：shape label 必要性、attribute 命名、image 對應 |
| cvat-video | detection, segmentation, tracking | 全部 CVAT shape tag | ✓ | ✓ | 完整：track/label 必要性、frame 屬性整數檢查；**不解碼實際影片內容** |
| 其餘 15 個 Phase 3 stub | 各自宣告 | 各自宣告 | 依格式 | 依格式 | 僅結構檢查（`required_any` glob 是否命中），永遠附帶警告說明尚未做內容驗證 |

## 4. 目前已知限制

1. **15 個 Phase 3 stub 格式沒有內容級驗證**——只檢查「至少有一個符合 glob pattern 的檔案」，不解析標註內容，也不做 image/annotation 配對。
2. **`cvat-video` 不解碼影片**——只驗證 XML 結構與 track frame 屬性是否為整數，不檢查影片實際長度/fps 是否與最大 frame 一致。
3. **Pillow 為選用依賴**——未安裝時所有格式都退化為「檔名/存在性」檢查，無法驗證影像實際尺寸是否與標註（如分割遮罩）匹配。
4. **測試 fixture 影像皆為純文字佔位檔**（例如 `fake-jpeg-bytes-for-testing-img1`），不是真實 JPEG bytes。這在「不安裝 Pillow」的預設路徑下完全沒問題，但如果未來測試要涵蓋 Pillow 尺寸驗證路徑，需要另外準備真實影像 fixture。
5. **`--manifest` 只在 CLI 層寫檔**——`FormatAdapter` 本身不知道 manifest 這個概念，這是刻意的關注點分離，但代表任何未來想在 adapter 內部觸發 manifest 寫入的需求都做不到（目前也沒有這種需求）。

## 5. 潛在技術債（審查中發現並已修正）

以下項目在審查時被發現且**已於本次修正**：

1. **跨平台 basename 解析錯誤**（`coco.py`、`cvat_image.py`）：原本用 `Path(file_name).name` 取得標註檔中 `file_name`/`name` 欄位的檔名。此寫法只依主機作業系統的路徑分隔符號切割，若標註檔是在 Windows 上產生、`file_name` 內含反斜線（如 `images\img1.jpg`），在 macOS/Linux 執行時會把整個字串當成單一檔名，導致「明明檔案存在卻被判定為 missing image」。已新增 `path_normalizer.basename()`，先把反斜線正規化成正斜線再取檔名，兩處呼叫點已改用它，並補上 `tests/test_path_normalizer.py` 回歸測試。
2. **image/annotation 配對邏輯重複三份**：`yolo.py`、`ultralytics_yolo.py`（`UltralyticsYoloBoxBase`）、`pascal_voc.py` 各自用 `set` 差集算「有影像無標註」「有標註無影像」，實作幾乎一樣但分散三處、日後改規則容易漏改。已抽成 `validators/common.py` 的 `unmatched_stems()`，三個 adapter 都已改用共用函式，行為不變（45→131 個既有測試全數通過）。
3. **`is_safe_relative_path("")` 回傳 `True`**：這是 zip 安全性檢查的邊界案例漏洞——空字串或正規化後等於 `"."` 的路徑理應視為不安全（不是有效的檔案項目），但原本的實作沒有擋掉。目前的正常流程不會產生空 arcname，所以不是可被利用的漏洞，但既然函式的職責就是「這是唯一把關安全路徑的地方」，屬於防禦性programming的缺口。已修正並補上單元測試。
4. **Ultralytics YOLO Pose 的 visibility 驗證是錯的（功能性 bug，非僅技術債）**：`UltralyticsYoloPoseAdapter._validate_line()` 原本直接呼叫父類別的「所有數值必須 normalized 在 [0,1]」檢查，但 keypoint 的第三個值是 visibility flag（合法值為 0/1/2），`visibility=2` 一律會被誤判為「座標超出 [0,1] 範圍」而報錯。這代表**任何合法的 Ultralytics Pose 資料集只要有一個可見的關鍵點（v=2）就會驗證失敗**。已重寫為讀取 `data.yaml` 的 `kpt_shape` 並分開驗證 bbox 座標（[0,1]）與 keypoint 座標（[0,1]）+ visibility（{0,1,2}），並補上明確的回歸測試 `test_pose_visibility_2_is_valid`。

## 6. 建議重構項目（尚未執行，供未來參考）

- `UltralyticsYoloBoxBase._validate_root()` hook 目前只有 Pose 用到；如果 Phase 3B 有更多格式需要「先驗證 data.yaml 額外欄位、再逐行驗證」的模式，可以考慮把這個 hook 往上提升到 `FormatAdapter` 基底類別，讓非 Ultralytics 格式也能複用。
- `find_coco_json()`（coco.py）與 `find_xml()`（cvat_image.py）的「多個候選路徑、取第一個存在的」邏輯結構相同，可考慮抽成 `validators/common.py` 的泛用 `find_first_existing(candidates)`，但目前只有兩處用到，重構效益有限，建議等第三個格式出現類似需求再動手（YAGNI）。
- Phase 3 stub 格式若進入內容驗證階段，建議先處理資料量最大/使用者需求最高的 3–4 個（例如 MOT、KITTI、LabelMe），而不是一次全部展開。

## 7. Phase 3 開發風險

1. **驗證嚴格度 vs 相容性的取捨**：多數 Phase 3 stub 格式（MOT、KITTI、Cityscapes 等）的社群標註工具產出的檔案品質參差不齊，若驗證做得太嚴格容易讓合法但「不完美」的資料集被拒絕；做得太鬆則失去驗證意義。建議每個格式先參考 CVAT 官方文件與至少一個真實資料集樣本再定驗證規則。
2. **`data.yaml`/`kpt_shape` 這類「格式專屬設定檔」的一致性**：Ultralytics 系列已證明「同一份 data.yaml schema，不同任務有不同必要欄位」這種模式容易在子類別間產生驗證不一致（本次的 visibility bug 就是一例）。未來若有更多共用設定檔的格式家族，建議一開始就設計好 `_validate_root()` 這類 hook，而不是讓每個子類別各自 override `_validate_line()` 卻各自為政。
3. **image/annotation 檔案量大時的效能**：目前所有格式都是同步、單執行緒逐檔驗證，`iter_files()` 會對整個目錄樹排序。對於數萬張圖片的資料集，`validate()` 的 Big-O 目前是可接受的（線性掃描+dict lookup），但 `zip_structure.py` 的 post-build 驗證與 `safe_copytree()` 尚未在大型資料集上做過效能測試，Phase 3 若要支援大型資料集（如完整 MOT Challenge），建議先做一次效能基準測試。
4. **測試 fixture 是文字佔位檔**：只要格式驗證邏輯開始依賴 Pillow 做影像內容檢查（尺寸比對等），現有 fixture 會在該路徑下失敗或被跳過，需要準備真實的最小合法影像檔（例如 1x1 像素 PNG/JPEG）。

## 8. 測試覆蓋狀況

審查前：45 個測試（`test_cli.py`, `test_coco.py`, `test_pascal_voc.py`, `test_registry.py`, `test_yolo.py`, `test_zip_output.py`），且已發現以下缺口：
- 沒有 `--help` 測試
- 沒有 `PackConfig`/`ValidationReport.to_dict()` 的獨立單元測試
- 沒有 zip 絕對路徑拒絕的單元測試
- 已有 fixture 的 Ultralytics YOLO Detection 完全沒有測試檔
- CVAT for images 只被 `test_zip_output.py` 間接測到（junk file 過濾），沒有專屬的 XML 驗證測試
- Pascal VOC 沒有測試「缺少 JPEGImages 資料夾」
- YOLO 沒有測試「labels 資料夾整個不存在」（只測了缺 obj.names）
- COCO Keypoints、4 個 Ultralytics 子格式完全沒有 fixture 與測試

審查後（本次全部補齊）：**131 個測試全數通過**，新增：
- `tests/test_path_normalizer.py`（8 個測試）：`is_safe_relative_path` 的絕對路徑/drive letter/`..`/空字串拒絕、`basename()` 跨平台行為
- `tests/test_core_models.py`（6 個測試）：`PackConfig` 型別轉換與預設值、`ValidationReport` 狀態機（只升不降）與 `to_dict()` 序列化
- `tests/test_cli.py` 新增 `--help` 測試
- `tests/test_pascal_voc.py` 新增缺少 JPEGImages 測試
- `tests/test_yolo.py` 新增缺少 labels 資料夾測試
- `tests/test_ultralytics_detection.py`（10 個測試，全新檔案，之前完全沒有）
- `tests/test_cvat_image.py`（9 個測試，全新檔案，之前只有間接覆蓋）
- `tests/test_coco_keypoints.py`（11 個測試 + fixture）
- `tests/test_ultralytics_segmentation.py`（10 個測試 + fixture）
- `tests/test_ultralytics_pose.py`（10 個測試 + fixture，含 visibility=2 迴歸測試）
- `tests/test_ultralytics_obb.py`（9 個測試 + fixture）
- `tests/test_ultralytics_classification.py`（8 個測試 + fixture）

每個新格式測試檔都涵蓋：valid dataset、缺少必要檔案、無效標註列、class id 超界、缺少影像（適用時）、zip 結構、`--dry-run`、`--validate-only`。

## 9. 下一步建議

1. **Phase 3B 選型**：建議從 15 個 stub 格式中挑選 3–4 個開始（例如 MOT、KITTI、LabelMe、Cityscapes），理由是這幾個格式在 CVAT 使用者社群中出現頻率較高，且都有相對明確的官方 schema 文件可參照。
2. **建立真實影像 fixture**：在啟用 Pillow 驗證路徑前，先準備一組最小合法影像（1x1 或極小尺寸的真實 PNG/JPEG bytes），取代現有的文字佔位 fixture，讓 `validators/image.py` 的路徑也能被測試覆蓋到。
3. **README 同步**：`README.md` 的格式狀態表原本就已正確反映「COCO Keypoints + Ultralytics 5 種子格式屬於 Full tier」，本次審查沒有發現需要修正的落差，但建議之後每次 Phase 3B 格式從 stub 升級為 full 時，第一步就同步移動狀態表，避免文件與程式碼再次出現落差（就像這次 Phase 3A 一樣，程式碼領先文件認知）。
4. **CI 建議**（目前專案沒有 CI 設定）：至少在 push/PR 時跑 `uv run pytest`，避免未來重構時像本次一樣得靠人工全文閱讀才能抓到 visibility bug 這類邏輯錯誤。
