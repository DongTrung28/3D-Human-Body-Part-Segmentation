import argparse, json, random, re
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply_root", required=True, help="folder with many .ply files")
    ap.add_argument("--out_dir", required=True, help="where to write train.txt val.txt test.txt and label_map.json")
    ap.add_argument("--palette_json", default="", help="optional color_to_part_id.json if you created one")
    ap.add_argument("--pattern", default="*.ply", help="filename glob, default all PLYs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    return ap.parse_args()

def infer_context_stem(p: Path):
    # Works with your file names like: <ctx>_ts<ts>_cam<cam>_fullpart.ply
    # Context is everything before "_ts"
    m = re.match(r"^(.*)_ts\d+_cam\d+.*\.ply$", p.name)
    return m.group(1) if m else "ctx"

def main():
    args = parse_args()
    root = Path(args.ply_root)
    out  = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(root.glob(args.pattern))
    files = [f for f in files if f.is_file() and f.suffix.lower()==".ply"]
    if not files:
        raise SystemExit("No PLY files found")

    # group by context so temporal neighbors likely land in the same split
    groups = {}
    for f in files:
        ctx = infer_context_stem(f)
        groups.setdefault(ctx, []).append(f)

    random.seed(args.seed)
    for g in groups.values():
        random.shuffle(g)

    all_paths = []
    for ctx, plist in groups.items():
        all_paths.extend(plist)
    total = len(all_paths)

    # simple split across the concatenated list while preserving groups
    train_cut = int(total * args.train_ratio)
    val_cut   = int(total * (args.train_ratio + args.val_ratio))
    train_list = all_paths[:train_cut]
    val_list   = all_paths[train_cut:val_cut]
    test_list  = all_paths[val_cut:]

    def write_list(pathlist, txt_path):
        with open(txt_path, "w") as f:
            for p in pathlist:
                # write relative paths so you can move the folder
                try:
                    rel = p.relative_to(out)
                except ValueError:
                    # not inside out dir, keep relative to common parent or just write absolute
                    rel = p
                f.write(str(rel).replace("\\", "/") + "\n")

    write_list(train_list, out / "train.txt")
    write_list(val_list,   out / "val.txt")
    write_list(test_list,  out / "test.txt")

    # label map
    # if you already created color_to_part_id.json then ids come from there
    id_set = set()
    pal_path = Path(args.palette_json)
    if pal_path.is_file():
        cmap = json.loads(Path(pal_path).read_text())
        id_set.update(int(v) for v in cmap.values())

    # also scan a few PLY headers to pick up part_id range if present
    sample_n = min(50, len(all_paths))
    for p in all_paths[:sample_n]:
        with open(p, "rb") as f:
            head = f.read(4096).decode("latin1", errors="ignore")
        if "property int part_id" in head or "property int32 part_id" in head:
            # we cannot get values from header, but we know part_id exists
            # leave discovery to the Dataset at load time
            pass

    # build id to name with generic labels if none known
    if not id_set:
        # default to twelve parts like CDGNet, numbered 1..12
        id_set = set(range(1, 13))

    id_list = sorted(list(id_set))
    id_to_name = {str(i): f"part_{i}" for i in id_list}
    label_map = {
        "num_parts": len(id_list),
        "ids": id_list,
        "id_to_name": id_to_name
    }
    (out / "label_map.json").write_text(json.dumps(label_map, indent=2))
    print(f"Done. Wrote splits and label map to {out}")
    print(f"Counts  train {len(train_list)}  val {len(val_list)}  test {len(test_list)}")

if __name__ == "__main__":
    main()