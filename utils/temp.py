import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# =========================
# 配置区（直接在这里改）
# =========================
INPUT_DIR = Path("/mnt/mydata/yh/liming/workspace/future/mydata/pose/60_15_15_15")
OUTPUT_DIR = INPUT_DIR / "pose_compare_vis"
MAX_FRAMES_PER_FILE = None   # None = 全部画
MODE = "separate"            # 固定：separate（你现在要的）


# =========================
# skeleton
# =========================
SKELETON = (
    (0,1),(1,2),(2,3),(3,4),
    (1,5),(5,6),(6,7),
    (1,8),(8,9),(9,10),(10,11),
    (8,12),(12,13),(13,14),
    (0,15),(15,17),(0,16),(16,18),
    (14,19),(19,20),(14,21),
    (11,22),(22,23),(11,24)
)


# =========================
# 数据处理
# =========================
def load_pose(path):
    pose = np.load(path)

    if pose.ndim == 2:
        pose = pose.reshape(pose.shape[0], 25, -1)

    if pose.shape[-1] == 2:
        conf = np.ones((*pose.shape[:-1],1))
        pose = np.concatenate([pose, conf], axis=-1)

    return pose.astype(np.float32)


def fill_missing(pose):
    pose = pose.copy()
    T = pose.shape[0]

    for i in range(T-1):
        for j in range(25):
            if pose[i,j,2] == 0:
                for k in range(i+1,T):
                    if pose[k,j,2] != 0:
                        pose[i,j] = pose[k,j].copy()
                        break

    return pose


def normalize_pose(pose):
    pose = pose.copy()

    center = pose[:,8:9,:2]
    pose[:,:,:2] -= center

    l = pose[:,5,:2]
    r = pose[:,2,:2]

    scale = np.linalg.norm(l-r, axis=1)
    scale = np.clip(scale, 1e-2, None)

    pose[:,:,:2] /= scale[:,None,None]
    pose[:,:,:2] = np.tanh(pose[:,:,:2])

    return pose


def inverse_tanh(x):
    return np.arctanh(np.clip(x, -0.9999, 0.9999))


# =========================
# bbox
# =========================
def compute_bbox(points, padding=0.1):
    pts = points.reshape(-1,2)
    pts = pts[np.isfinite(pts).all(axis=1)]

    if len(pts)==0:
        return (-1,1),(-1,1)

    xmin,ymin = np.percentile(pts,5,axis=0)
    xmax,ymax = np.percentile(pts,95,axis=0)

    cx = (xmin+xmax)/2
    cy = (ymin+ymax)/2
    w = (xmax-xmin)*(1+padding)
    h = (ymax-ymin)*(1+padding)

    return (cx-w/2, cx+w/2),(cy-h/2, cy+h/2)


# =========================
# 绘图
# =========================
def draw_pose(ax, xy, mask, xlim, ylim, title):
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")

    rect = Rectangle((xlim[0],ylim[0]),
                     xlim[1]-xlim[0],
                     ylim[1]-ylim[0],
                     fill=False)
    ax.add_patch(rect)

    for i,j in SKELETON:
        if mask[i] and mask[j]:
            ax.plot([xy[i,0],xy[j,0]],
                    [xy[i,1],xy[j,1]])

    pts = xy[mask]
    ax.scatter(pts[:,0], pts[:,1], s=20)

    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)


# =========================
# 主处理
# =========================
def process_file(file_path):
    pose = load_pose(file_path)

    raw = pose.copy()
    filled = fill_missing(pose)
    norm = normalize_pose(filled)
    proc = inverse_tanh(norm[:,:,:2])

    raw_mask = raw[:,:,2] > 0
    filled_mask = filled[:,:,2] > 0

    # 👉 separate 坐标系（你要求的）
    raw_box = compute_bbox(raw[:,:,:2])
    proc_box = compute_bbox(proc)

    save_dir = OUTPUT_DIR / file_path.stem
    save_dir.mkdir(parents=True, exist_ok=True)

    T = raw.shape[0]
    indices = np.arange(T)

    if MAX_FRAMES_PER_FILE is not None:
        indices = np.random.choice(T, min(T, MAX_FRAMES_PER_FILE), replace=False)

    for i in indices:
        fig, ax = plt.subplots(1,2, figsize=(10,5))

        draw_pose(ax[0], raw[i,:,:2], raw_mask[i],
                  raw_box[0], raw_box[1], "Raw")

        draw_pose(ax[1], proc[i], filled_mask[i],
                  proc_box[0], proc_box[1], "Processed")

        plt.savefig(save_dir / f"{i:05d}.png", dpi=200)
        plt.close()


# =========================
# 批量处理
# =========================
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    files = sorted(INPUT_DIR.glob("*.npy"))

    for f in files:
        process_file(f)
        print(f"[OK] {f.name}")

    print("DONE")


if __name__ == "__main__":
    main()