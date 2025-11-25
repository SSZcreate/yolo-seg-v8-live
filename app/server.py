# app/server.py
import os
import math
import time
import base64
import threading
import asyncio
from typing import Optional, Dict, Any, List

import numpy as np
import cv2
from fastapi import FastAPI, APIRouter, Query, WebSocket, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO

APP_VER = "v1"

# ===== 既定設定（環境変数で上書き可能） =====
WEIGHTS       = os.getenv("WEIGHTS", r"C:\Users\ok122\Python\yolo-seg-v8-live\weight\best.pt")
DEFAULT_CLASS = os.getenv("CLASS_NAME", "slopes")
DEFAULT_METH  = os.getenv("METHOD", "inscribed")  # "fast" or "inscribed"
DEFAULT_IMGSZ = int(os.getenv("IMGSZ", "960"))
DEFAULT_CONF  = float(os.getenv("CONF", "0.30"))
DEFAULT_DIL   = int(os.getenv("DILATE", "0"))

# ===== FastAPI 初期化 =====
app = FastAPI(title="Sashigane Slope Live API", version=APP_VER)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
rtr = APIRouter(prefix=f"/api/{APP_VER}")

# ===== 共有状態 =====
_state: Dict[str, Any] = {
    "running": False,
    "thread": None,
    "cap": None,
    "cfg": {
        "class_name": DEFAULT_CLASS,
        "method": DEFAULT_METH,
        "imgsz": DEFAULT_IMGSZ,
        "conf": DEFAULT_CONF,
        "dilate": DEFAULT_DIL,
        "cam": 0,
        "target_fps": 15,
        "width": 0,
        "height": 0,
        "smooth": 0.0,  # 0.0=無効（指数移動平均の係数）
    },
    "latest": {
        "ok": False, "found": False, "ts": 0.0,
        "theta_deg": None, "slope_sun": None, "rise_over_run": None,
        "run_px": None, "rise_px": None, "class": None, "conf": None,
        "image_jpg_b64": None
    },
    "lock": threading.Lock(),
    "stop_flag": False
}

# ========= ユーティリティ（あなたの処理を移植） =========
def _to_int(v):
    try:
        return int(round(float(v)))
    except Exception:
        return None

def contours_from_yolov8_result(r, W, H, target_classes=None, min_conf=0.0, min_area_px=50):
    """
    YOLOv8 segmentation 結果から、各インスタンスの凸包輪郭を抽出
    """
    out = []
    if r.masks is None:
        return out
    names = r.names
    boxes = r.boxes
    clss  = boxes.cls.cpu().numpy().astype(int) if boxes is not None else np.zeros(len(r.masks), dtype=int)
    confs = boxes.conf.cpu().numpy() if boxes is not None else np.ones(len(r.masks), dtype=float)
    polys = r.masks.xy  # list of (Ni,2)
    for i, poly in enumerate(polys):
        cls_name = names[int(clss[i])] if isinstance(names, (dict, list)) else str(clss[i])
        if target_classes and cls_name not in target_classes:
            continue
        conf = float(confs[i])
        if conf < min_conf:
            continue
        pts = []
        for x, y in poly:
            xi = _to_int(x); yi = _to_int(y)
            if xi is None or yi is None:
                continue
            if 0 <= xi < W and 0 <= yi < H:
                pts.append([xi, yi])
        if len(pts) < 3:
            continue
        cnt = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        area = abs(cv2.contourArea(cnt))
        if area < min_area_px:
            continue
        hull = cv2.convexHull(cnt)
        out.append({"id": i, "class": cls_name, "confidence": conf, "contour": hull})
    return out

def right_triangle_from_contour_fast(contour):
    """
    近似多角形→最大面積の3点→直角補正で、直角三角形を高速近似
    """
    cnt = contour
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True).reshape(-1, 2)
    if len(approx) >= 3:
        max_area, best = -1, None
        for i in range(len(approx)):
            for j in range(i+1, len(approx)):
                for k in range(j+1, len(approx)):
                    cand = np.array([approx[i], approx[j], approx[k]], dtype=np.float32)
                    area = abs(cv2.contourArea(cand))
                    if area > max_area:
                        max_area, best = area, cand
        tri = best
    else:
        tri = cnt.reshape(-1, 2).astype(np.float32)[:3]

    def _angle(a, b, c):
        ab = a - b; cb = c - b
        cosang = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-8)
        return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

    angs = [_angle(tri[(i-1) % 3], tri[i], tri[(i+1) % 3]) for i in range(3)]
    right_idx = int(np.argmin([abs(a - 90) for a in angs]))
    A = tri[right_idx]; B = tri[(right_idx + 1) % 3]; C = tri[(right_idx + 2) % 3]
    v1 = B - A; v2 = C - A
    v1n = v1 / (np.linalg.norm(v1) + 1e-8)
    C2 = A + (v2 - np.dot(v2, v1n) * v1n)
    len1 = float(np.linalg.norm(B - A)); len2 = float(np.linalg.norm(C2 - A))
    if len1 >= len2:
        width_px, height_px = len1, len2
        tri_pts = np.array([A, B, C2], dtype=np.float32)
        width_leg = (A, B); height_leg = (A, C2)
    else:
        width_px, height_px = len2, len1
        tri_pts = np.array([A, C2, B], dtype=np.float32)
        width_leg = (A, C2); height_leg = (A, B)
    return tri_pts, width_px, height_px, width_leg, height_leg

def triangle_best_fit_from_mask(mask, angle_step_deg=3, apex_stride=2, dilate_iterations=0):
    """
    マスク内に収まる最大の直角三角形を探索（厳密寄り）
    """
    seg = (mask > 0).astype(np.uint8)
    if dilate_iterations > 0:
        kernel = np.ones((3, 3), np.uint8)
        seg = cv2.dilate(seg, kernel, iterations=dilate_iterations)

    cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise ValueError("no segment")
    cnt = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(cnt); hull_pts = hull.reshape(-1, 2)
    H, W = seg.shape

    def _tri_mask(A, e, u, v):
        e = e / (np.linalg.norm(e) + 1e-8); ep = np.array([-e[1], e[0]], np.float32)
        B = A + u * e; C = A + v * ep
        tri = np.array([A, B, C], np.float32)
        m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [tri.astype(np.int32)], 1)
        return m, tri

    def _ray_len(A, d, step=1.0):
        d = d / (np.linalg.norm(d) + 1e-8); t = 0.0; last = 0.0
        for _ in range(int(math.hypot(W, H) / step) + 2):
            x = int(round(A[0] + t * d[0])); y = int(round(A[1] + t * d[1]))
            if x < 0 or x >= W or y < 0 or y >= H or seg[y, x] == 0:
                break
            last = t; t += step
        return last

    def _shrink_inside(A, e, u, v, iters=12):
        lo_u, hi_u = 0.0, u; lo_v, hi_v = 0.0, v
        for _ in range(iters):
            mu = 0.5 * (lo_u + hi_u); mv = 0.5 * (lo_v + hi_v)
            tm, _ = _tri_mask(A, e, mu, mv)
            if (tm & (1 - seg)).any():
                tm_u, _ = _tri_mask(A, e, mu, v)
                tm_v, _ = _tri_mask(A, e, u, mv)
                if (tm_u & (1 - seg)).any() and not (tm_v & (1 - seg)).any():
                    hi_u = mu
                elif (tm_v & (1 - seg)).any() and not (tm_u & (1 - seg)).any():
                    hi_v = mv
                else:
                    hi_u, hi_v = mu, mv
            else:
                lo_u, lo_v = mu, mv
        return lo_u, lo_v

    best = {"score": -1, "tri": None, "u": 0.0, "v": 0.0}
    for idx in range(0, len(hull_pts), max(1, apex_stride)):
        A = hull_pts[idx].astype(np.float32)
        for ang in range(0, 180, angle_step_deg):
            th = math.radians(ang)
            e = np.array([math.cos(th), math.sin(th)], np.float32)
            u0 = _ray_len(A, e); v0 = _ray_len(A, np.array([-e[1], e[0]], np.float32))
            if u0 < 2 or v0 < 2:
                continue
            u_in, v_in = _shrink_inside(A, e, u0, v0)
            tm, tri = _tri_mask(A, e, u_in, v_in)
            area = (u_in * v_in) / 2.0
            if area > best["score"]:
                best = {"score": area, "tri": tri, "u": u_in, "v": v_in}
    if best["tri"] is None:
        raise ValueError("no inscribed triangle")
    return best["tri"], float(best["u"]), float(best["v"])

def canonicalize_dims(tri_pts, policy="longer"):
    """
    2辺のうち長い方を width、短い方を height として正規化
    """
    A, B, C = tri_pts[0].astype(np.float32), tri_pts[1].astype(np.float32), tri_pts[2].astype(np.float32)
    v1, v2 = B - A, C - A
    l1, l2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if policy == "more_horizontal":
        h1, h2 = abs(v1[0]), abs(v2[0])
        if h1 >= h2:
            width, height = l1, l2; width_leg = (A, B); height_leg = (A, C)
        else:
            width, height = l2, l1; width_leg = (A, C); height_leg = (A, B)
    else:
        if l1 >= l2:
            width, height = l1, l2; width_leg = (A, B); height_leg = (A, C)
        else:
            width, height = l2, l1; width_leg = (A, C); height_leg = (A, B)
    ratio = height / width if width > 0 else float("nan")
    return width, height, ratio, width_leg, height_leg

def draw_overlay_on_frame(frame_bgr, tri_pts, width_leg, height_leg, ratio, alpha=0.35):
    """
    フレームに三角形と脚を描画してBase64(JPG)で返す
    """
    vis = frame_bgr.copy()
    overlay = vis.copy()
    cv2.fillPoly(overlay, [tri_pts.astype(np.int32)], (0, 255, 0))
    vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)
    cv2.polylines(vis, [tri_pts.astype(np.int32)], True, (255, 255, 255), 2)
    H, W = vis.shape[:2]
    th = max(2, int(min(H, W) / 400) + 1)
    for p0, p1 in (width_leg, height_leg):
        cv2.line(vis, tuple(np.int32(np.round(p0))), tuple(np.int32(np.round(p1))), (0, 0, 255), th)
    fs = max(0.5, min(H, W) / 800.0); tth = max(1, int(min(H, W) / 400))
    cv2.putText(vis, f"ratio(h/w)={ratio:.6f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 0), tth, cv2.LINE_AA)
    ok, jpg = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return base64.b64encode(jpg.tobytes()).decode("ascii") if ok else None

def draw_mask_only_on_frame(frame_bgr, contour, alpha=0.35, color=(255, 0, 0)):
    H, W = frame_bgr.shape[:2]
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [contour.reshape(-1, 2)], 255)
    overlay = np.zeros_like(frame_bgr); overlay[mask == 255] = color
    vis = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)
    vis = np.where(mask[..., None] == 255, vis, frame_bgr)
    ok, jpg = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return base64.b64encode(jpg.tobytes()).decode("ascii") if ok else None

# ===== YOLO モデル（起動時に一度だけロード） =====
_model = YOLO(WEIGHTS)

# ===== Webカメラ → 推論 → 最新値更新 のワーカ =====
def _worker_loop():
    cfg = _state["cfg"]
    class_name = cfg["class_name"] or None
    method = cfg["method"]
    imgsz = int(cfg["imgsz"])
    conf  = float(cfg["conf"])
    dilate= int(cfg["dilate"])
    cam   = int(cfg["cam"])
    target_fps = float(cfg["target_fps"])
    width = int(cfg["width"]); height = int(cfg["height"])
    smooth = float(cfg["smooth"])
    ema_ratio: Optional[float] = None  # EMA 平滑化

    # カメラ
    try:
        if os.name == "nt":
            cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(cam)
        if width > 0:  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        _state["cap"] = cap
    except Exception as e:
        with _state["lock"]:
            _state["latest"] = {"ok": False, "found": False, "ts": time.time(), "error": f"camera_open_fail: {e}"}
        return

    period = 1.0 / max(1.0, target_fps)

    while not _state["stop_flag"]:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        H, W = frame.shape[:2]
        try:
            r = _model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
            targets = [class_name] if class_name else None
            items = contours_from_yolov8_result(r, W, H, target_classes=targets, min_conf=0.2, min_area_px=100)

            if not items:
                with _state["lock"]:
                    _state["latest"] = {
                        "ok": True, "found": False, "ts": time.time(),
                        "theta_deg": None, "slope_sun": None, "rise_over_run": None,
                        "run_px": None, "rise_px": None, "class": None, "conf": None,
                        "image_jpg_b64": None
                    }
            else:
                it = max(items, key=lambda d: abs(cv2.contourArea(d["contour"])))
                cnt = it["contour"]

                if method == "fast":
                    tri_pts, width_px, height_px, width_leg, height_leg = right_triangle_from_contour_fast(cnt)
                else:
                    mask = np.zeros((H, W), np.uint8)
                    cv2.fillPoly(mask, [cnt.reshape(-1, 2)], 1)
                    tri_pts, u, v = triangle_best_fit_from_mask(mask, angle_step_deg=1, apex_stride=1, dilate_iterations=dilate)
                    width_px, height_px, _, width_leg, height_leg = canonicalize_dims(tri_pts, policy="longer")

                width_px, height_px, ratio, width_leg, height_leg = canonicalize_dims(tri_pts, policy="longer")

                # 指数移動平均で滑らかに（smooth=0なら無効）
                if smooth > 0.0 and not math.isnan(ratio):
                    if ema_ratio is None:
                        ema_ratio = ratio
                    else:
                        ema_ratio = (1 - smooth) * ratio + smooth * ema_ratio
                    ratio_to_use = ema_ratio
                else:
                    ratio_to_use = ratio

                theta_deg = math.degrees(math.atan(ratio_to_use)) if ratio_to_use == ratio_to_use else None
                n_sun = 10.0 * ratio_to_use if ratio_to_use == ratio_to_use else None
                b64 = draw_overlay_on_frame(frame, tri_pts, width_leg, height_leg, ratio_to_use, alpha=0.35)

                with _state["lock"]:
                    _state["latest"] = {
                        "ok": True, "found": True, "ts": time.time(),
                        "theta_deg": float(theta_deg) if theta_deg is not None else None,
                        "slope_sun": float(n_sun) if n_sun is not None else None,
                        "rise_over_run": float(ratio_to_use) if ratio_to_use is not None else None,
                        "run_px": float(width_px), "rise_px": float(height_px),
                        "class": it["class"], "conf": float(it["confidence"]),
                        "image_jpg_b64": b64
                    }
        except Exception as e:
            with _state["lock"]:
                _state["latest"] = {"ok": False, "found": False, "ts": time.time(), "error": str(e)}

        # 目標FPSに合わせてスリープ
        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)

    # 終了処理
    try:
        cap.release()
    except Exception:
        pass
    with _state["lock"]:
        _state["cap"] = None

# ===== ルーティング =====
@rtr.get("/health")
def health():
    return {"ok": True, "version": APP_VER, "running": _state["running"]}

@rtr.post("/start")
def start(
    cam: int = Query(0),
    imgsz: int = Query(DEFAULT_IMGSZ),
    conf: float = Query(DEFAULT_CONF),
    class_name: str = Query(DEFAULT_CLASS),
    method: str = Query(DEFAULT_METH, pattern="^(fast|inscribed)$"),
    dilate: int = Query(DEFAULT_DIL),
    target_fps: float = Query(15),
    width: int = Query(0), height: int = Query(0),
    smooth: float = Query(0.0, ge=0.0, le=0.95)
):
    if _state["running"]:
        return {"ok": True, "running": True, "note": "already running", "cfg": _state["cfg"]}
    _state["cfg"].update(dict(
        cam=cam, imgsz=imgsz, conf=conf, class_name=class_name, method=method,
        dilate=dilate, target_fps=target_fps, width=width, height=height, smooth=smooth
    ))
    _state["stop_flag"] = False
    th = threading.Thread(target=_worker_loop, daemon=True)
    _state["thread"] = th
    _state["running"] = True
    th.start()
    return {"ok": True, "running": True, "cfg": _state["cfg"]}

@rtr.post("/stop")
def stop():
    if not _state["running"]:
        return {"ok": True, "running": False}
    _state["stop_flag"] = True
    t = _state["thread"]
    if t is not None:
        t.join(timeout=2.0)
    _state["running"] = False
    return {"ok": True, "running": False}

@rtr.get("/latest")
def latest(with_image: bool = Query(False)):
    with _state["lock"]:
        out = dict(_state["latest"])
    if not with_image and "image_jpg_b64" in out:
        out["image_jpg_b64"] = None
    out["running"] = _state["running"]
    return JSONResponse(out)

@rtr.websocket("/ws")
async def ws_latest(websocket: WebSocket, hz: float = 15.0):
    await websocket.accept()
    period = 1.0 / max(1.0, hz)
    try:
        while True:
            with _state["lock"]:
                out = dict(_state["latest"])
                out["running"] = _state["running"]
                out["image_jpg_b64"] = None  # WSは数値中心（軽量）
            await websocket.send_json(out)
            await asyncio.sleep(period)
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass

@rtr.post("/infer")
async def infer_api(
    image: UploadFile = File(...),
    class_name: str = Query(DEFAULT_CLASS),
    method: str = Query(DEFAULT_METH, pattern="^(fast|inscribed)$"),
    imgsz: int = Query(DEFAULT_IMGSZ),
    conf: float = Query(DEFAULT_CONF),
    dilate: int = Query(DEFAULT_DIL),
    mask_only: bool = Query(False)
):
    """
    単発推論：画像をアップロードして角度・n寸を返す（APIだけで可視化確認したいとき用）
    """
    data = await image.read()
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return {"ok": False, "reason": "decode_failed"}

    H, W = frame.shape[:2]
    r = _model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
    targets = [class_name] if class_name else None
    items = contours_from_yolov8_result(r, W, H, target_classes=targets, min_conf=0.2, min_area_px=100)
    if not items:
        return {"ok": True, "found": False}

    it = max(items, key=lambda d: abs(cv2.contourArea(d["contour"])))
    cnt = it["contour"]

    if mask_only:
        b64 = draw_mask_only_on_frame(frame, cnt, alpha=0.35, color=(255, 0, 0))
        return {
            "ok": True, "found": True, "class": it["class"], "conf": float(it["confidence"]),
            "image_jpg_b64": b64
        }

    if method == "fast":
        tri_pts, width_px, height_px, width_leg, height_leg = right_triangle_from_contour_fast(cnt)
    else:
        mask = np.zeros((H, W), np.uint8); cv2.fillPoly(mask, [cnt.reshape(-1, 2)], 1)
        tri_pts, u, v = triangle_best_fit_from_mask(mask, angle_step_deg=1, apex_stride=1, dilate_iterations=dilate)
        width_px, height_px, _, width_leg, height_leg = canonicalize_dims(tri_pts, policy="longer")

    width_px, height_px, ratio, width_leg, height_leg = canonicalize_dims(tri_pts, policy="longer")
    theta_deg = math.degrees(math.atan(ratio)) if ratio == ratio else None
    n_sun = 10.0 * ratio if ratio == ratio else None
    b64 = draw_overlay_on_frame(frame, tri_pts, width_leg, height_leg, ratio, alpha=0.35)

    return {
        "ok": True, "found": True,
        "class": it["class"], "conf": float(it["confidence"]),
        "theta_deg": float(theta_deg) if theta_deg is not None else None,
        "slope_sun": float(n_sun) if n_sun is not None else None,
        "rise_over_run": float(ratio) if ratio is not None else None,
        "run_px": float(width_px), "rise_px": float(height_px),
        "image_jpg_b64": b64
    }

app.include_router(rtr)
