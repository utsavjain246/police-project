"""
Installation commands

git clone https://github.com/utsavjain246/police-project.git
cd police-project/CrowdCounting-P2PNet/
pip -q install -r requirements.txt

"""

import os, csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from models import build_model
from util.misc import nested_tensor_from_tensor_list

VIDEO_PATH  = "/people-walking/People-walking.mp4"
WEIGHT_PATH = ".CrowdCounting-P2PNet/weights/SHTechA.pth"
OUT_DIR     = "/p2pnet_video_out"

GPU_ID      = 0
THRESHOLD   = 0.5
STRIDE      = 1                # 1 = every frame
DRAW_POINTS = True              # draw dots on output video
SAVE_VIDEO  = True              
OUT_VIDEO_PATH = f"{OUT_DIR}/demo.avi"
OUT_FPS     = 0                # set 0 to use input fps

# Optional: speed-up by downscaling frames before inference
SCALE_FACTOR = 1.0              # e.g. 0.5 or 0.4 like your code; 1.0 means no resize

# =========================
# Build model (repo default)
# =========================
os.makedirs(OUT_DIR, exist_ok=True)
points_dir = Path(OUT_DIR) / "points"
points_dir.mkdir(parents=True, exist_ok=True)

device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
print("Device:", device)

class Args: pass
args = Args()
args.backbone = "vgg16_bn"
args.row = 2
args.line = 2

model = build_model(args)
ckpt = torch.load(WEIGHT_PATH, map_location="cpu")
model.load_state_dict(ckpt["model"])
model.to(device).eval()

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

# ============
# Open video
# ============
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

in_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
fps = OUT_FPS if OUT_FPS > 0 else (in_fps if in_fps > 0 else 30)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
print(f"Opened video. in_fps={in_fps:.2f}, out_fps={fps:.2f}, total_frames={total}")

# Prepare writers after first processed frame (so we know exact size)
writer = None
fourcc = cv2.VideoWriter_fourcc(*"XVID")

counts_csv = Path(OUT_DIR) / "counts.csv"
fcsv = open(counts_csv, "w", newline="")
csv_writer = csv.writer(fcsv)
csv_writer.writerow(["frame_idx", "time_sec", "count"])

frame_idx = -1
processed = 0

while True:
    ok, frame_bgr = cap.read()
    if not ok:
        break
    frame_idx += 1
    if frame_idx % STRIDE != 0:
        continue

    processed += 1

    # Optional downscale like your code
    if SCALE_FACTOR != 1.0:
        frame_bgr = cv2.resize(frame_bgr, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)

    h0, w0 = frame_bgr.shape[:2]
    t_sec = (frame_idx / in_fps) if in_fps > 0 else 0.0

    # BGR -> RGB for correct normalization
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(frame_rgb)

    img_t = transform(pil)                     # (3,H,W)
    samples = nested_tensor_from_tensor_list([img_t]).to(device)

    with torch.no_grad():
        outputs = model(samples)

    scores = torch.softmax(outputs["pred_logits"][0], dim=-1)[:, 1]  # (Q,)
    points = outputs["pred_points"][0]                               # (Q,2)

    keep = scores > THRESHOLD
    pts = points[keep].detach().cpu().numpy()
    count = int(keep.sum().item())

    csv_writer.writerow([frame_idx, f"{t_sec:.3f}", count])

    # Save points (x y)
    with open(points_dir / f"frame_{frame_idx:06d}.txt", "w") as fp:
        for x, y in pts:
            fp.write(f"{x:.2f} {y:.2f}\n")

    # Draw + save video (optional)
    if SAVE_VIDEO:
        vis = frame_bgr.copy()
        if DRAW_POINTS:
            for x, y in pts:
                cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
        cv2.putText(vis, f"Count: {count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

        if writer is None:
            writer = cv2.VideoWriter(OUT_VIDEO_PATH, fourcc, fps, (vis.shape[1], vis.shape[0]))
        writer.write(vis)

    if processed % 10 == 0:
        print(f"[processed={processed}] frame={frame_idx}/{total} t={t_sec:.2f}s count={count}")

cap.release()
if writer is not None:
    writer.release()
fcsv.close()

print("\nDONE ✅")
print("counts.csv:", counts_csv)
print("points dir:", points_dir)
if SAVE_VIDEO:
    print("output video:", OUT_VIDEO_PATH)


#STEP 2:- 

VIDEO_PATH = "/people-walking/People-walking.mp4"

# Step-2 outputs (the folder you used in step 2)
STEP2_OUT_DIR = "/p2pnet_video_out"   # contains counts.csv and points/
STEP2_COUNTS_CSV = f"{STEP2_OUT_DIR}/counts.csv"
STEP2_POINTS_DIR = f"{STEP2_OUT_DIR}/points"

WEIGHT_PATH = "./CrowdCounting-P2PNet/weights/SHTechA.pth"
OUT_DIR = "/p2pnet_video_refined"  

GPU_ID = 0
PRED_THRESHOLD = 0.5      # same threshold you used in step 2
SCALE_FACTOR = 1.0        # MUST match step 2 (if you used 0.4 there, set 0.4 here)

# Trigger logic
COUNT_THR = 20            # if rough count > 50 -> crowd present
GRID = 32                 # density grid cell size in pixels
DENSITY_THR = 0.02        # points per pixel^2 inside a GRID cell (tune!)
ROI_MIN_PX = 128          # ignore tiny dense blobs
ROI_MARGIN = 64           # expand ROI around dense area

# Tiling refinement
TILE_SIZE = 640
OVERLAP = 96
NMS_RADIUS = 6.0          # merge duplicates between overlapping tiles


# =========================
# Helpers
# =========================
def read_points_txt(path):
    pts = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            x, y = line.split()[:2]
            pts.append([float(x), float(y)])
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array(pts, dtype=np.float32)

def nms_points_radius(points_xy, radius=6.0):
    """Simple radius suppression to de-duplicate overlapping-tile points."""
    if len(points_xy) == 0:
        return points_xy
    pts = points_xy.astype(np.float32)
    keep = []
    taken = np.zeros(len(pts), dtype=bool)
    r2 = radius * radius
    for i in range(len(pts)):
        if taken[i]:
            continue
        keep.append(pts[i])
        d2 = np.sum((pts - pts[i])**2, axis=1)
        taken |= (d2 <= r2)
        taken[i] = False
    return np.array(keep, dtype=np.float32)

def dense_rois_from_points(points_xy, H, W, grid=32, density_thr=0.02, min_roi_px=128, margin=64):
    """
    Bin points into grid cells.
    density(cell) = points_in_cell / (grid*grid)
    If density > density_thr -> dense.
    Return ROIs as (x0,y0,x1,y1).
    """
    if len(points_xy) == 0:
        return []

    gh = int(np.ceil(H / grid))
    gw = int(np.ceil(W / grid))
    hist = np.zeros((gh, gw), dtype=np.int32)

    xs = np.clip((points_xy[:, 0] / grid).astype(int), 0, gw - 1)
    ys = np.clip((points_xy[:, 1] / grid).astype(int), 0, gh - 1)
    for x, y in zip(xs, ys):
        hist[y, x] += 1

    density = hist / float(grid * grid)
    mask = (density > density_thr).astype(np.uint8) * 255
    if mask.max() == 0:
        return []

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    rois = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        x0, y0 = x * grid, y * grid
        x1, y1 = min((x + w) * grid, W), min((y + h) * grid, H)

        if (x1 - x0) < min_roi_px or (y1 - y0) < min_roi_px:
            continue

        x0 = max(0, x0 - margin); y0 = max(0, y0 - margin)
        x1 = min(W, x1 + margin); y1 = min(H, y1 + margin)
        rois.append((x0, y0, x1, y1))

    return rois

def resize_keep_aspect(pil_img, target_min_side=800, max_side=1333):
    w, h = pil_img.size
    scale = target_min_side / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    if max(new_w, new_h) > max_side:
        scale2 = max_side / max(new_w, new_h)
        new_w, new_h = int(round(new_w * scale2)), int(round(new_h * scale2))
        scale *= scale2
    try:
        resample = Image.Resampling.LANCZOS
    except Exception:
        resample = getattr(Image, "LANCZOS", 1)
    pil_rs = pil_img.resize((new_w, new_h), resample)
    return pil_rs, scale

def infer_points_on_bgr(model, frame_bgr, device, transform, threshold=0.5):
    """
    Runs P2PNet on a BGR image and returns points in ORIGINAL image coords.
    Uses same resize approach as step 2 for stable results.
    """
    h0, w0 = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    pil_rs, scale = resize_keep_aspect(pil, target_min_side=800, max_side=1333)

    img_t = transform(pil_rs)
    samples = nested_tensor_from_tensor_list([img_t]).to(device)

    with torch.no_grad():
        outputs = model(samples)

    scores = torch.softmax(outputs["pred_logits"][0], dim=-1)[:, 1]  # (Q,)
    points = outputs["pred_points"][0]                               # (Q,2) in resized coords

    keep = scores > threshold
    pts = points[keep].detach().cpu().numpy()

    # map resized -> original
    pts[:, 0] /= max(scale, 1e-8)
    pts[:, 1] /= max(scale, 1e-8)
    pts[:, 0] = np.clip(pts[:, 0], 0, w0 - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h0 - 1)
    return pts

def refine_with_tiling(model, frame_bgr, rois, device, transform,
                       tile_size=640, overlap=96, threshold=0.5, nms_radius=6.0):
    """
    For each ROI: tile it and run P2PNet per tile.
    Return refined points in full-frame coordinates.
    """
    H, W = frame_bgr.shape[:2]
    refined_all = []

    step = max(1, tile_size - overlap)

    for (x0, y0, x1, y1) in rois:
        roi = frame_bgr[y0:y1, x0:x1]
        rh, rw = roi.shape[:2]

        for ty in range(0, rh, step):
            for tx in range(0, rw, step):
                cx0, cy0 = tx, ty
                cx1, cy1 = min(tx + tile_size, rw), min(ty + tile_size, rh)
                tile = roi[cy0:cy1, cx0:cx1]

                pts_tile = infer_points_on_bgr(model, tile, device, transform, threshold=threshold)
                if len(pts_tile) == 0:
                    continue

                # shift tile points into full-frame coords
                pts_full = pts_tile + np.array([x0 + cx0, y0 + cy0], dtype=np.float32)
                refined_all.append(pts_full)

    if len(refined_all) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    pts = np.concatenate(refined_all, axis=0)
    pts = nms_points_radius(pts, radius=nms_radius)
    # clip
    pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
    return pts

device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# =========================
# Read Step-2 counts -> frames to process
# =========================
counts = []
with open(STEP2_COUNTS_CSV, "r") as f:
    r = csv.DictReader(f)
    for row in r:
        frame_idx = int(row["frame_idx"])
        t_sec = float(row["time_sec"])
        rough = int(row["count"])
        counts.append((frame_idx, t_sec, rough))

need_frames = {fi for fi, _, rough in counts if rough >= COUNT_THR}
print(f"Frames with rough_count >= {COUNT_THR}: {len(need_frames)}")

# =========================
# Output folders
# =========================
out_dir = Path(OUT_DIR)
pts_ref_dir = out_dir / "points_refined"
out_dir.mkdir(parents=True, exist_ok=True)
pts_ref_dir.mkdir(parents=True, exist_ok=True)

ref_csv = out_dir / "counts_refined.csv"
fcsv = open(ref_csv, "w", newline="")
wcsv = csv.writer(fcsv)
wcsv.writerow(["frame_idx", "time_sec", "rough_count", "refined_count", "risk_level", "num_rois"])

# =========================
# Iterate video sequentially, refine only needed frames
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

in_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
print(f"Opened video. FPS={in_fps:.2f}, total_frames={total}")

frame_idx = -1
done = 0

# For fast lookup from counts list
rough_map = {fi: (t, rough) for fi, t, rough in counts}

while True:
    ok, frame_bgr = cap.read()
    if not ok:
        break
    frame_idx += 1

    if frame_idx not in need_frames:
        continue

    t_sec, rough = rough_map[frame_idx]

    # IMPORTANT: match Step-2 scaling
    if SCALE_FACTOR != 1.0:
        frame_bgr = cv2.resize(frame_bgr, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)

    # load Step-2 points for this frame
    pts_path = Path(STEP2_POINTS_DIR) / f"frame_{frame_idx:06d}.txt"
    if not pts_path.exists():
        # If missing, fall back to re-infer full frame (rare)
        pts_rough = infer_points_on_bgr(model, frame_bgr, device, transform, threshold=PRED_THRESHOLD)
    else:
        pts_rough = read_points_txt(str(pts_path))

    H, W = frame_bgr.shape[:2]

    # Risk logic
    risk_level = "CROWDED" if rough >= COUNT_THR else "OK"

    # Density trigger -> ROIs
    rois = dense_rois_from_points(
        pts_rough, H=H, W=W,
        grid=GRID, density_thr=DENSITY_THR,
        min_roi_px=ROI_MIN_PX, margin=ROI_MARGIN
    )

    if len(rois) == 0:
        refined_pts = pts_rough
        refined_count = int(len(refined_pts))
    else:
        risk_level = "SUPER_DENSE"
        # Tiling only on ROIs
        pts_dense_refined = refine_with_tiling(
            model, frame_bgr, rois,
            device=device, transform=transform,
            tile_size=TILE_SIZE, overlap=OVERLAP,
            threshold=PRED_THRESHOLD, nms_radius=NMS_RADIUS
        )

        # Replace points inside ROIs with refined ones
        inside_any = np.zeros(len(pts_rough), dtype=bool)
        for (x0, y0, x1, y1) in rois:
            inside_any |= (
                (pts_rough[:, 0] >= x0) & (pts_rough[:, 0] < x1) &
                (pts_rough[:, 1] >= y0) & (pts_rough[:, 1] < y1)
            )
        pts_outside = pts_rough[~inside_any]
        refined_pts = np.concatenate([pts_outside, pts_dense_refined], axis=0)
        refined_pts = nms_points_radius(refined_pts, radius=NMS_RADIUS)
        refined_count = int(len(refined_pts))

    # Save refined points
    out_pts_path = pts_ref_dir / f"frame_{frame_idx:06d}.txt"
    with open(out_pts_path, "w") as fp:
        for x, y in refined_pts:
            fp.write(f"{x:.2f} {y:.2f}\n")

    wcsv.writerow([frame_idx, f"{t_sec:.3f}", rough, refined_count, risk_level, len(rois)])
    done += 1
    if done % 10 == 0:
        print(f"[refined={done}] frame={frame_idx} rough={rough} refined={refined_count} rois={len(rois)}")

cap.release()
fcsv.close()

print("\nDONE ✅ Step 3 finished")
print("Refined counts:", ref_csv)
print("Refined points dir:", pts_ref_dir)





import os, csv, json
from pathlib import Path
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

VIDEO_PATH = "/people-walking/People-walking.mp4"

STEP3_DIR = "/p2pnet_video_refined"
REFINED_POINTS_DIR = f"{STEP3_DIR}/points_refined"
COUNTS_REFINED_CSV = f"{STEP3_DIR}/counts_refined.csv"

DENSITY_OUT_DIR = "/density_analysis_dbscan"
DENSITY_REPORT_CSV = f"{DENSITY_OUT_DIR}/density_report.csv"
SAVE_TOPK_VIS = True
TOPK_VIS = 10

SCALE_FACTOR = 1.0 

# --- Group/cluster settings ---
EPS_PX = 60                # DBSCAN radius in pixels (increase if groups are spread out)
MIN_CLUSTER_PEOPLE = 4     # ignore clusters smaller than this

# --- Hotspot definition (density of a cluster bbox) ---
# density_10k = (people / area_px2) * 10000
DENSITY_10K_THR = 1.2      # hotspot if >= 1.2 people per 10,000 px^2  (tune 0.8–2.5)

# --- ROI shaping ---
ROI_MARGIN = 40            # expand bbox around cluster
ROI_MIN_PX = 64            # ignore tiny boxes (width/height)

# --- Heatmap visualization from points (Gaussian smoothing) ---
HEAT_SIGMA_PX = 25         # bigger => smoother heatmap
CLIP_PERCENTILE = 99       # contrast (98–99.5)
# =========================

Path(DENSITY_OUT_DIR).mkdir(parents=True, exist_ok=True)
Path(f"{DENSITY_OUT_DIR}/vis").mkdir(parents=True, exist_ok=True)


def read_points_txt(path):
    """Reads points from txt where each line: x y"""
    pts = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            x, y = line.split()[:2]
            pts.append([float(x), float(y)])
    return np.array(pts, dtype=np.float32) if len(pts) else np.zeros((0, 2), np.float32)


def cluster_rois_dbscan(points_xy, H, W, eps_px=60, min_cluster_people=4,
                        density_10k_thr=1.2, roi_margin=40, roi_min_px=64):
    """
    1) Cluster points using DBSCAN
    2) For each cluster:
       - bbox = [minx,miny,maxx,maxy]
       - area = bbox_area
       - density_10k = count/area * 10000
    3) Hotspot if density_10k >= threshold AND count >= min_cluster_people.
    Returns:
      - rois: list of (x0,y0,x1,y1)
      - clusters_stats: list[{bbox,count,area,density,density_10k}]
      - max_density_10k among clusters (0 if none)
    """
    if len(points_xy) == 0:
        return [], [], 0.0

    # DBSCAN assigns label -1 to noise (points not in any cluster)
    labels = DBSCAN(eps=eps_px, min_samples=min_cluster_people).fit_predict(points_xy)

    rois = []
    stats = []
    max_density_10k = 0.0

    for lab in np.unique(labels):
        if lab == -1:
            continue
        cluster_pts = points_xy[labels == lab]
        cnt = int(len(cluster_pts))
        if cnt < min_cluster_people:
            continue

        x0 = float(cluster_pts[:, 0].min())
        y0 = float(cluster_pts[:, 1].min())
        x1 = float(cluster_pts[:, 0].max())
        y1 = float(cluster_pts[:, 1].max())

        # expand
        x0 = max(0, int(x0 - roi_margin))
        y0 = max(0, int(y0 - roi_margin))
        x1 = min(W, int(x1 + roi_margin))
        y1 = min(H, int(y1 + roi_margin))

        if (x1 - x0) < roi_min_px or (y1 - y0) < roi_min_px:
            continue

        area = float((x1 - x0) * (y1 - y0) + 1e-9)
        density = cnt / area
        density_10k = density * 10000.0
        max_density_10k = max(max_density_10k, density_10k)

        rec = {
            "bbox": [int(x0), int(y0), int(x1), int(y1)],
            "count": cnt,
            "area_px2": area,
            "density": density,
            "density_10k": density_10k,
        }
        stats.append(rec)

        # hotspot condition
        if density_10k >= density_10k_thr:
            rois.append((int(x0), int(y0), int(x1), int(y1)))

    # sort clusters by density_10k desc
    stats.sort(key=lambda d: d["density_10k"], reverse=True)
    return rois, stats, float(max_density_10k)


def density_map_from_points(points_xy, H, W, sigma_px=25):
    """
    Proper smooth density heatmap:
    impulse at each point -> Gaussian blur -> smooth map
    """
    m = np.zeros((H, W), dtype=np.float32)
    if len(points_xy) > 0:
        xs = np.clip(points_xy[:, 0].astype(int), 0, W - 1)
        ys = np.clip(points_xy[:, 1].astype(int), 0, H - 1)
        m[ys, xs] = 1.0

    dens = cv2.GaussianBlur(m, (0, 0), sigmaX=sigma_px, sigmaY=sigma_px)
    # keep total mass ~ number of people
    if dens.sum() > 1e-8:
        dens *= (m.sum() / dens.sum())
    return dens


def overlay_density_heatmap(frame_bgr, dens, rois, clip_percentile=99):
    """
    Overlay smooth density map onto frame + draw ROI boxes.
    """
    heat = dens.copy()
    vmax = np.percentile(heat, clip_percentile)
    if vmax <= 1e-9:
        out = frame_bgr.copy()
    else:
        heat_norm = np.clip(heat / vmax, 0, 1)
        heat_u8 = (heat_norm * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        out = cv2.addWeighted(frame_bgr, 0.65, heat_color, 0.35, 0)

    for (x0, y0, x1, y1) in rois:
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
    return out


# =========================
# Read counts to know frames
# =========================
rows = []
with open(COUNTS_REFINED_CSV, "r") as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append({
            "frame_idx": int(row["frame_idx"]),
            "time_sec": float(row["time_sec"]),
            "refined_count": int(row["refined_count"]),
            "risk_level": row["risk_level"],
        })

need = []
for row in rows:
    p = Path(REFINED_POINTS_DIR) / f"frame_{row['frame_idx']:06d}.txt"
    if p.exists():
        need.append(row)

need_set = set(d["frame_idx"] for d in need)
need_map = {d["frame_idx"]: d for d in need}
print("Frames with refined points:", len(need))

# =========================
# Analyze frames
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

report = []
frame_idx = -1

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_idx += 1
    if frame_idx not in need_set:
        continue

    if SCALE_FACTOR != 1.0:
        frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)

    H, W = frame.shape[:2]
    pts = read_points_txt(str(Path(REFINED_POINTS_DIR) / f"frame_{frame_idx:06d}.txt"))

    rois, cluster_stats, max_density_10k = cluster_rois_dbscan(
        pts, H, W,
        eps_px=EPS_PX,
        min_cluster_people=MIN_CLUSTER_PEOPLE,
        density_10k_thr=DENSITY_10K_THR,
        roi_margin=ROI_MARGIN,
        roi_min_px=ROI_MIN_PX
    )

    base = need_map[frame_idx]
    rec = {
        "frame_idx": frame_idx,
        "time_sec": base["time_sec"],
        "refined_count": base["refined_count"],
        "risk_level": base["risk_level"],
        "max_cluster_density_10k": max_density_10k,
        "num_hotspot_rois": len(rois),
        "hotspots": [s for s in cluster_stats if s["density_10k"] >= DENSITY_10K_THR][:5],
    }
    report.append(rec)

cap.release()

# =========================
# Save report CSV
# =========================
with open(DENSITY_REPORT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "frame_idx","time_sec","refined_count","risk_level",
        "max_cluster_density_10k","num_hotspot_rois","hotspots_json"
    ])
    for r in report:
        w.writerow([
            r["frame_idx"], f"{r['time_sec']:.3f}", r["refined_count"], r["risk_level"],
            f"{r['max_cluster_density_10k']:.3f}", r["num_hotspot_rois"],
            json.dumps(r["hotspots"])
        ])

print("Saved:", DENSITY_REPORT_CSV)

# =========================
# Save top-K visualizations
# =========================
if SAVE_TOPK_VIS and len(report) > 0:
    top = sorted(report, key=lambda d: d["max_cluster_density_10k"], reverse=True)[:TOPK_VIS]
    top_set = set(d["frame_idx"] for d in top)
    top_map = {d["frame_idx"]: d for d in top}

    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_idx = -1
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx not in top_set:
            continue

        if SCALE_FACTOR != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)

        H, W = frame.shape[:2]
        pts = read_points_txt(str(Path(REFINED_POINTS_DIR) / f"frame_{frame_idx:06d}.txt"))

        rois, cluster_stats, max_density_10k = cluster_rois_dbscan(
            pts, H, W,
            eps_px=EPS_PX,
            min_cluster_people=MIN_CLUSTER_PEOPLE,
            density_10k_thr=DENSITY_10K_THR,
            roi_margin=ROI_MARGIN,
            roi_min_px=ROI_MIN_PX
        )

        dens = density_map_from_points(pts, H, W, sigma_px=HEAT_SIGMA_PX)
        vis = overlay_density_heatmap(frame, dens, rois, clip_percentile=CLIP_PERCENTILE)

        txt = (f"frame={frame_idx} count={top_map[frame_idx]['refined_count']} "
               f"maxD10k={top_map[frame_idx]['max_cluster_density_10k']:.2f} "
               f"thr={DENSITY_10K_THR}")
        cv2.putText(vis, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        out_path = Path(DENSITY_OUT_DIR) / "vis" / f"heatmap_{frame_idx:06d}.jpg"
        cv2.imwrite(str(out_path), vis)
        saved += 1

    cap.release()
    print(f"Saved {saved} heatmaps to: {Path(DENSITY_OUT_DIR)/'vis'}")

print("\nDONE ✅ DBSCAN hotspot analysis complete")