# convert_ply_to_human3d.py
import numpy as np
from pathlib import Path
import json
import argparse
import open3d as o3d


def load_color_map(json_path: Path):
    """Load color_to_part_id.json and build both BGR and RGB lookup tables."""
    with open(json_path, "r") as f:
        raw = json.load(f)

    # stored as "b,g,r"
    bgr2id = {tuple(map(int, k.split(","))): int(v) for k, v in raw.items()}
    # and an RGB version matching what Open3D returns
    rgb2id = {(bgr[2], bgr[1], bgr[0]): pid for bgr, pid in bgr2id.items()}
    return bgr2id, rgb2id


def read_ply_any(path: Path):
    """Read PLY with Open3D and return xyz, rgb, inst, part."""
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=np.float32)

    cols = np.asarray(pcd.colors, dtype=np.float32)
    if cols.size:
        # Open3D colors are in [0,1] as RGB
        cols = np.clip((cols * 255.0).round(), 0, 255).astype(np.uint8)
    else:
        cols = np.zeros((pts.shape[0], 3), np.uint8)

    # For now we do not have instance or part ids stored in the PLY
    inst = np.zeros((pts.shape[0],), np.int32)
    part = np.zeros((pts.shape[0],), np.int32)
    return pts, cols, inst, part


def derive_part_ids_from_rgb(rgb_u8: np.ndarray, rgb2id: dict):
    """Map each RGB triplet to a part id using rgb2id, unknown colors become 0."""
    N = rgb_u8.shape[0]
    out = np.zeros(N, dtype=np.int32)
    for i in range(N):
        t = tuple(int(c) for c in rgb_u8[i])
        out[i] = rgb2id.get(t, 0)
    return out


def process_one(ply_path: Path, out_root: Path, rgb2id: dict, add_instance: bool):
    pts, rgb, inst, part = read_ply_any(ply_path)

    # If part ids are all zero, derive them from colors
    if part.max() == 0 and rgb.size:
        part = derive_part_ids_from_rgb(rgb, rgb2id)

    # Features for Human3D: xyz and rgb as float
    feats = np.concatenate([pts, rgb.astype(np.float32)], axis=1)  # N x 6

    stem = ply_path.stem
    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / f"{stem}.npy", feats.astype(np.float32))
    np.savetxt(out_dir / f"{stem}_gt_part.txt", part.astype(np.int32), fmt="%d")

    if add_instance:
        np.savetxt(out_dir / f"{stem}_gt_human.txt", inst.astype(np.int32), fmt="%d")
    else:
        np.savetxt(out_dir / f"{stem}_gt_human.txt", np.zeros_like(inst), fmt="%d")

    return out_dir


def resolve_ply_path(line: str, ply_root: Path) -> Path:
    """
    Accept both:
      absolute path: C:\\Waymo\\human3d_ply\\foo_fullpart.ply
      basename or relative: foo_fullpart or foo_fullpart.ply
    and return a valid Path to the PLY.
    """
    line = line.strip()
    if not line:
        return None

    p = Path(line)

    # Case 1: the line itself is a path to an existing file
    if p.is_file():
        return p

    # Case 2: the line is a basename without extension
    if p.suffix.lower() != ".ply":
        p = p.with_suffix(".ply")

    candidate = ply_root / p.name
    if candidate.is_file():
        return candidate

    # If nothing worked, return a non existing path so caller can skip it
    return candidate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply-root", required=True, help="folder with your colored PLYs")
    ap.add_argument("--train-list", required=True, help="txt with items for training")
    ap.add_argument("--val-list", required=True, help="txt with items for validation")
    ap.add_argument("--color-map", required=True, help="color_to_part_id.json")
    ap.add_argument("--out-root", required=True, help="destination folder")
    ap.add_argument(
        "--add-instance",
        action="store_true",
        help="write gt_human from PLY if you have it",
    )
    args = ap.parse_args()

    ply_root = Path(args.ply_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    _, rgb2id = load_color_map(Path(args.color_map))

    for split_name, txt_path in [("train", args.train_list), ("val", args.val_list)]:
        txt_path = Path(txt_path)
        lines = [x.strip() for x in txt_path.read_text().splitlines() if x.strip()]
        total = len(lines)
        print(f"{split_name} items in list: {total}", flush=True)

        processed = 0
        skipped = 0
        out_split_dirs = []

        for idx, line in enumerate(lines, start=1):
            ply_path = resolve_ply_path(line, ply_root)
            if ply_path is None or not ply_path.is_file():
                print(f"[{split_name}] {idx}/{total} skip missing {ply_path}", flush=True)
                continue

            stem = ply_path.stem
            out_dir = out_root / stem
            out_npy = out_dir / f"{stem}.npy"
            out_part = out_dir / f"{stem}_gt_part.txt"
            out_inst = out_dir / f"{stem}_gt_human.txt"

            # resume support: if all outputs exist, skip
            if out_npy.is_file() and out_part.is_file() and out_inst.is_file():
                skipped += 1
                if idx % 50 == 0:
                    print(
                        f"[{split_name}] {idx}/{total} already done, skip {stem} "
                        f"(processed {processed}, skipped {skipped})",
                        flush=True,
                    )
                out_split_dirs.append(str(out_dir.resolve()))
                continue

            out_dir = process_one(ply_path, out_root, rgb2id, args.add_instance)
            processed += 1
            out_split_dirs.append(str(out_dir.resolve()))

            # progress info every 20 samples
            if idx % 20 == 0 or idx == total:
                print(
                    f"[{split_name}] {idx}/{total} processed {stem} "
                    f"(processed {processed}, skipped {skipped})",
                    flush=True,
                )

        out_list_file = out_root / f"{split_name}_list.txt"
        out_list_file.write_text("\n".join(out_split_dirs) + "\n")
        print(
            f"{split_name} done, processed {processed}, skipped {skipped}, "
            f"list written to {out_list_file}",
            flush=True,
        )

    print("done, processed samples in", out_root)


if __name__ == "__main__":
    main()
