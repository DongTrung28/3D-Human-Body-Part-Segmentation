import time
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from model import BodySeg


IN_DIR = Path("test_ply")
OUT_DIR = Path("pred_test_ply")
CKPT_PATH = Path("checkpoints/latest.pt")  # last checkpoint
# CKPT_PATH = Path("checkpoints/best.pt") ## checkpoint with best val acc


NUM_CLASSES = 12
DIM_MODEL = [32, 64, 128, 256, 512]
K_NEIGHBORS = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


COLORS = np.array(
    [
        [255, 255, 255],
        [220, 20, 60],
        [255, 140, 0],
        [30, 144, 255],
        [0, 191, 255],
        [34, 139, 34],
        [50, 205, 50],
        [138, 43, 226],
        [186, 85, 211],
        [255, 105, 180],
        [255, 215, 0],
        [160, 82, 45],
    ],
    dtype=np.uint8,
)


def read_ply_xyz(path: Path) -> np.ndarray:
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)
    return xyz


def write_colored_ply(path: Path, xyz: np.ndarray, pred: np.ndarray) -> None:
    rgb = COLORS[pred]

    vertex = np.empty(
        xyz.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("part_id", "i4"),
        ],
    )

    vertex["x"] = xyz[:, 0]
    vertex["y"] = xyz[:, 1]
    vertex["z"] = xyz[:, 2]
    vertex["red"] = rgb[:, 0]
    vertex["green"] = rgb[:, 1]
    vertex["blue"] = rgb[:, 2]
    vertex["part_id"] = pred.astype(np.int32)

    el = PlyElement.describe(vertex, "vertex")
    PlyData([el], text=True).write(str(path))


@torch.no_grad()
def predict_one(model: torch.nn.Module, xyz: np.ndarray) -> np.ndarray:
    pos = torch.tensor(xyz, dtype=torch.float32, device=DEVICE)
    x = torch.ones((pos.shape[0], 3), dtype=torch.float32, device=DEVICE)
    batch = torch.zeros((pos.shape[0],), dtype=torch.long, device=DEVICE)

    out = model(x, pos, batch)
    pred = out.argmax(dim=1).to("cpu").numpy().astype(np.int64)
    pred = np.clip(pred, 0, NUM_CLASSES - 1)
    return pred


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("device", DEVICE)
    print("cuda available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu name", torch.cuda.get_device_name(0))

    model = BodySeg(3, NUM_CLASSES, dim_model=DIM_MODEL, k=K_NEIGHBORS).to(DEVICE)
    state = torch.load(str(CKPT_PATH), map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    ply_paths = sorted(IN_DIR.glob("*.ply"))
    print("num ply", len(ply_paths))

    for i, p in enumerate(ply_paths):
        t0 = time.time()
        xyz = read_ply_xyz(p)
        pred = predict_one(model, xyz)
        out_path = OUT_DIR / p.name
        write_colored_ply(out_path, xyz, pred)
        dt = time.time() - t0
        uniq = np.unique(pred).tolist()
        print(i, p.name, "points", xyz.shape[0], "secs", round(dt, 3), "labels", uniq)


if __name__ == "__main__":
    main()
