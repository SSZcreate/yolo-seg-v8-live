# YOLO Segmentation v8 Live API

YOLOv8セグメンテーションモデルを使用して、リアルタイムで傾斜角度を測定するFastAPI Webサーバー。

## 機能

- Webカメラからのリアルタイム推論
- 三角形の自動検出と角度計算
- REST API経由での制御
- WebSocket対応
- 画像付き結果取得
- 複数カメラ対応

## セットアップ

### 必要要件

- Python 3.8以上
- Webカメラ
- YOLOv8セグメンテーションモデル (`.pt`ファイル)

### インストール

```bash
# 仮想環境の作成と有効化
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate   # Linux/Mac

# 依存パッケージのインストール
pip install -r requirements.txt
```

### モデルファイル

`weight/`ディレクトリにYOLOv8モデルファイル (`best.pt`) を配置してください。

## 使い方

### サーバーの起動

```bash
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

### カメラ一覧の確認

```bash
python check_cameras.py
```

### API使用例

#### 1. 推論開始

```powershell
Invoke-WebRequest -Method POST -Uri "http://127.0.0.1:8000/api/v1/start?cam=0&imgsz=1280&conf=0.3&class_name=slopes&method=inscribed&dilate=1&target_fps=15&smooth=0.6"
```

**パラメータ:**
- `cam`: カメラ番号 (0, 1, 2...)
- `imgsz`: 推論画像サイズ (デフォルト: 960)
- `conf`: 信頼度閾値 (デフォルト: 0.30)
- `class_name`: 検出クラス名 (デフォルト: "slopes")
- `method`: 三角形検出方法 (`fast` または `inscribed`)
- `dilate`: マスク膨張回数 (デフォルト: 0)
- `target_fps`: 目標FPS (デフォルト: 15)
- `smooth`: 平滑化係数 0.0-0.95 (0.0で無効、デフォルト: 0.0)

#### 2. 最新結果の取得

```powershell
# 数値のみ
curl "http://127.0.0.1:8000/api/v1/latest"

# 画像付き
curl "http://127.0.0.1:8000/api/v1/latest?with_image=true"
```

#### 3. 画像の保存と表示 (PowerShell)

```powershell
$response = (curl "http://127.0.0.1:8000/api/v1/latest?with_image=true").Content | ConvertFrom-Json
$bytes = [Convert]::FromBase64String($response.image_jpg_b64)
[System.IO.File]::WriteAllBytes("$PWD\latest_result.jpg", $bytes)
Start-Process "latest_result.jpg"
```

#### 4. 推論停止

```powershell
Invoke-WebRequest -Method POST -Uri "http://127.0.0.1:8000/api/v1/stop"
```

#### 5. ヘルスチェック

```bash
curl "http://127.0.0.1:8000/api/v1/health"
```

## APIエンドポイント

| エンドポイント | メソッド | 説明 |
|--------------|---------|------|
| `/api/v1/health` | GET | サーバー状態確認 |
| `/api/v1/start` | POST | カメラ推論開始 |
| `/api/v1/stop` | POST | 推論停止 |
| `/api/v1/latest` | GET | 最新の推論結果取得 |
| `/api/v1/ws` | WebSocket | リアルタイムストリーム |
| `/api/v1/infer` | POST | 単発画像推論 (画像アップロード) |

## レスポンス例

```json
{
  "ok": true,
  "found": true,
  "ts": 1764044272.99,
  "theta_deg": 31.067,
  "slope_sun": 6.025,
  "rise_over_run": 0.602,
  "run_px": 55.986,
  "rise_px": 45.989,
  "class": "slopes",
  "conf": 0.85,
  "running": true
}
```

**フィールド:**
- `theta_deg`: 傾斜角度（度）
- `slope_sun`: n寸勾配 (10 × rise_over_run)
- `rise_over_run`: 高さ/幅の比率
- `run_px`: 底辺の長さ（ピクセル）
- `rise_px`: 高さ（ピクセル）

## 環境変数

```bash
WEIGHTS=./weight/best.pt          # モデルファイルパス
CLASS_NAME=slopes                  # デフォルトクラス名
METHOD=inscribed                   # デフォルト検出方法
IMGSZ=960                          # デフォルト画像サイズ
CONF=0.30                          # デフォルト信頼度閾値
DILATE=0                           # デフォルト膨張回数
```


## 作者

[Your Name]
