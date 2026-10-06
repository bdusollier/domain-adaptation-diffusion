import json
import argparse
import random
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DETECTION = Path(__file__).resolve().parent

IMG_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

SYNTH_IMAGES = BASE / "data1" / "input" / "image"
REAL_IMAGES = BASE / "data1" / "output"
LABELS = BASE / "data1" / "input" / "labels"
TEST_DIR = BASE.parent / "Dataset" / "vehicles.coco" / "dataset_filtered"

COCO_JSON = TEST_DIR / "_annotations.coco.json"

TOTAL_IMAGES = 852


def make_yolo_dirs(name):
    for split in ["train", "val"]:
        (DETECTION / "dataset1" / name / "images" / split).mkdir(parents=True, exist_ok=True)
        (DETECTION / "dataset1" / name / "labels" / split).mkdir(parents=True, exist_ok=True)


def build_label_index():
    idx = {}
    for f in LABELS.iterdir():
        if f.suffix == ".txt":
            idx[f.stem] = f
    return idx


def read_and_convert_label(lbl_file):
    lines = []
    for line in lbl_file.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            lines.append(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
    return "\n".join(lines)


def prepare_training_split(name, image_dir, label_index, split_name, indices):
    img_out = DETECTION / "dataset1" / name / "images" / split_name
    lbl_out = DETECTION / "dataset1" / name / "labels" / split_name

    for i in indices:
        key = f"{i:05d}"
        lbl_file = label_index.get(key)

        candidates = (
            list(image_dir.glob(f"IMG.{key}.png"))
            + list(image_dir.glob(f"{key}*.png"))
            + list(image_dir.glob(f"{key}*.jpg"))
        )
        if not candidates:
            continue

        img_file = candidates[0]
        out_name = key + img_file.suffix

        target = img_out / out_name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(img_file.resolve())

        if lbl_file is not None:
            converted = read_and_convert_label(lbl_file)
            (lbl_out / (key + ".txt")).write_text(converted + "\n" if converted else "")
        else:
            (lbl_out / (key + ".txt")).write_text("")


def coco_to_yolo_test():
    with open(COCO_JSON) as f:
        coco = json.load(f)

    img_id_to_info = {img["id"]: img for img in coco["images"]}

    img_out = DETECTION / "dataset1" / "test" / "images"
    lbl_out = DETECTION / "dataset1" / "test" / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    ann_by_img = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    for img_id, info in img_id_to_info.items():
        file_name = info["file_name"]
        candidates = list(TEST_DIR.glob(f"*_{file_name}"))
        if not candidates:
            continue

        src_img = candidates[0]
        out_name = Path(file_name).stem + Path(file_name).suffix
        target = img_out / out_name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(src_img.resolve())

        lbl_lines = []
        img_w, img_h = info["width"], info["height"]
        for ann in ann_by_img.get(img_id, []):
            x, y, w, h = ann["bbox"]
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw, nh = w / img_w, h / img_h
            lbl_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        (lbl_out / (Path(file_name).stem + ".txt")).write_text(
            "\n".join(lbl_lines) + "\n" if lbl_lines else ""
        )

    print(f"Test: {len(list(img_out.iterdir()))} images")


def reorganize_flat_dataset(dataset_dir):
    """
    Réorganise un dataset à plat (images/ + labels/ sans train/val)
    pour suivre le format de dataset1/ : images/{train,val} et labels/{train,val}.
    Les images style '00000_ComfyUI_00001_.png' sont renommées '00000.png'
    pour correspondre à leurs labels (indispensable pour YOLO).
    """
    det = DETECTION / dataset_dir
    if not det.exists():
        print(f"Erreur : le dossier '{det}' n'existe pas.")
        return

    for name in ["realistic", "synthetic"]:
        img_src = det / name / "images"
        lbl_src = det / name / "labels"

        if not img_src.exists() or not lbl_src.exists():
            print(f"Erreur : '{img_src}' ou '{lbl_src}' absent. Ignoré.")
            continue

        if (img_src / "train").exists() or (img_src / "val").exists():
            print(f"{dataset_dir}/{name}: déjà au format train/val, ignoré.")
            continue

        pairs = []
        for img in sorted(img_src.glob("*")):
            if not img.is_file():
                continue
            if img.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                continue
            stem = img.stem.split("_ComfyUI_")[0]
            pairs.append((stem, img, img.suffix.lower()))

        pairs.sort(key=lambda x: x[0])
        split = int(len(pairs) * 0.8)

        for i, (stem, img, suffix) in enumerate(pairs):
            split_name = "train" if i < split else "val"
            img_out_dir = img_src / split_name
            lbl_out_dir = lbl_src / split_name
            img_out_dir.mkdir(parents=True, exist_ok=True)
            lbl_out_dir.mkdir(parents=True, exist_ok=True)

            img.rename(img_out_dir / (stem + suffix))
            lbl = lbl_src / (img.stem + ".txt")
            if not lbl.exists():
                lbl = lbl_src / (stem + ".txt")
            if lbl.exists():
                lbl.rename(lbl_out_dir / (stem + ".txt"))
            else:
                (lbl_out_dir / (stem + ".txt")).write_text("")

        print(f"{dataset_dir}/{name}: {split} train, {len(pairs) - split} val")

    print("Done.")


def split_real_val(dataset_dir, n=100, seed=0):
    """
    Déplace un échantillon d'images du test vers dataset<X>/val_real (images + labels).
    Ce val réel sert de validation d'entraînement (early stopping) et reste
    disjoint du test final (pas de fuite de données).
    """
    det = DETECTION / dataset_dir
    test_img = det / "test" / "images"
    test_lbl = det / "test" / "labels"
    val_img = det / "val_real" / "images"
    val_lbl = det / "val_real" / "labels"

    if not test_img.exists():
        print(f"Erreur : '{test_img}' n'existe pas.")
        return

    if val_img.exists() and any(val_img.iterdir()):
        print(f"'{val_img}' contient déjà des images : split déjà effectué, ignoré.")
        return

    img_files = sorted(
        f for f in test_img.iterdir() if f.is_file() and f.suffix.lower() in IMG_SUFFIXES
    )
    stems = [f.stem for f in img_files]
    if n >= len(stems):
        print(f"--n ({n}) >= nombre d'images de test ({len(stems)}). Aucun split.")
        return

    rng = random.Random(seed)
    rng.shuffle(stems)
    val_set = set(stems[:n])

    val_img.mkdir(parents=True, exist_ok=True)
    val_lbl.mkdir(parents=True, exist_ok=True)

    moved = 0
    for f in img_files:
        if f.stem in val_set:
            f.rename(val_img / f.name)
            lbl = test_lbl / (f.stem + ".txt")
            if lbl.exists():
                lbl.rename(val_lbl / lbl.name)
            moved += 1

    print(f"{dataset_dir}: {moved} images déplacées test -> val_real (seed={seed})")
    print(f"  test     : {len(list(test_img.iterdir()))} images")
    print(f"  val_real : {len(list(val_img.iterdir()))} images")


def cmd_prepare(args=None):
    label_index = build_label_index()
    all_indices = list(range(TOTAL_IMAGES))
    split = int(TOTAL_IMAGES * 0.8)
    train_indices = all_indices[:split]
    val_indices = all_indices[split:]

    for name, image_dir in [("synthetic", SYNTH_IMAGES), ("realistic", REAL_IMAGES)]:
        make_yolo_dirs(name)
        prepare_training_split(name, image_dir, label_index, "train", train_indices)
        prepare_training_split(name, image_dir, label_index, "val", val_indices)
        train_n = len(list((DETECTION / "dataset1" / name / "images" / "train").iterdir()))
        val_n = len(list((DETECTION / "dataset1" / name / "images" / "val").iterdir()))
        print(f"{name}: {train_n} train, {val_n} val")

    coco_to_yolo_test()
    print("Done.")


def _process_one_crop(args):
    img_path, lbl_path, out_img_path, out_lbl_path, target_size = args
    try:
        from PIL import Image
        with Image.open(img_path) as im:
            W, H = im.size
            S = min(W, H)
            xmin = (W - S) / 2
            ymin = (H - S) / 2
            crop_im = im.crop((xmin, ymin, xmin + S, ymin + S))
            if (S, S) != (target_size, target_size):
                crop_im = crop_im.resize((target_size, target_size), Image.Resampling.LANCZOS)
            crop_im.save(out_img_path)

        new_lbl_lines = []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = parts[0]
                cx, cy, w, h = map(float, parts[1:5])
                x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
                x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
                orig_area = max(0, x2 - x1) * max(0, y2 - y1)

                x1c = max(0, min(S, x1 - xmin))
                x2c = max(0, min(S, x2 - xmin))
                y1c = max(0, min(S, y1 - ymin))
                y2c = max(0, min(S, y2 - ymin))

                crop_area = max(0, x2c - x1c) * max(0, y2c - y1c)
                if crop_area > 0 and orig_area > 0 and (crop_area / orig_area) >= 0.25:
                    cx_n = (x1c + x2c) / (2 * S)
                    cy_n = (y1c + y2c) / (2 * S)
                    w_n = (x2c - x1c) / S
                    h_n = (y2c - y1c) / S
                    new_lbl_lines.append(f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")

        out_lbl_path.write_text("\n".join(new_lbl_lines) + ("\n" if new_lbl_lines else ""))
        return True
    except Exception as e:
        print(f"Error processing {img_path.name}: {e}")
        return False


def center_crop_dataset(dataset_dir, target_size=1024, splits=None):
    from multiprocessing import Pool, cpu_count
    import shutil

    if splits is None:
        splits = ["val_real", "test"]

    det = DETECTION / dataset_dir
    if not det.exists():
        print(f"Erreur : le dossier '{det}' n'existe pas.")
        return

    for split in splits:
        split_dir = det / split
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"

        if not img_dir.exists() or not lbl_dir.exists():
            print(f"Split '{split}' introuvable dans {dataset_dir}. Ignoré.")
            continue

        backup_dir = det / f"{split}_raw"
        if not backup_dir.exists():
            print(f"Création de la sauvegarde : {backup_dir}")
            shutil.copytree(split_dir, backup_dir)
        else:
            print(f"Sauvegarde existante trouvée : {backup_dir}")

        img_files = sorted(
            f for f in (backup_dir / "images").iterdir()
            if f.is_file() and f.suffix.lower() in IMG_SUFFIXES
        )
        print(f"\nRecadrage centré {split} ({len(img_files)} images $\to$ {target_size}x{target_size})...")

        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        tasks = []
        for img_f in img_files:
            lbl_f = backup_dir / "labels" / f"{img_f.stem}.txt"
            out_img = img_dir / f"{img_f.stem}.png"
            out_lbl = lbl_dir / f"{img_f.stem}.txt"
            tasks.append((img_f, lbl_f, out_img, out_lbl, target_size))

        workers = max(1, cpu_count() - 2)
        with Pool(workers) as pool:
            results = pool.map(_process_one_crop, tasks)

        success = sum(1 for r in results if r)
        print(f"Terminé pour {split}: {success}/{len(tasks)} images recadrées.")

        # Supprimer le cache ultralytics
        for cache_f in split_dir.glob("*.cache"):
            cache_f.unlink()

    print("\nRecadrage centré terminé avec succès pour tous les splits !")


def filter_test_boxes(
    dataset_dir="dataset3",
    src_split="test2",
    dst_split="test3",
    max_w=0.23125,
    max_h=0.16992,
    max_area=0.034813,
    copy_images=True,
):
    """
    Crée un sous-dossier de test dst_split à partir de src_split en filtrant
    les annotations des objets trop gros (hors des dimensions observées en entraînement).
    """
    import shutil

    det = DETECTION / dataset_dir
    src_dir = det / src_split
    dst_dir = det / dst_split

    src_img_dir = src_dir / "images"
    src_lbl_dir = src_dir / "labels"
    dst_img_dir = dst_dir / "images"
    dst_lbl_dir = dst_dir / "labels"

    if not src_img_dir.exists() or not src_lbl_dir.exists():
        print(f"Erreur : '{src_img_dir}' ou '{src_lbl_dir}' introuvable.")
        return

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(
        f for f in src_img_dir.iterdir() if f.is_file() and f.suffix.lower() in IMG_SUFFIXES
    )

    total_boxes = 0
    kept_boxes = 0
    removed_boxes = 0
    empty_images = 0

    for img_f in img_files:
        dst_img = dst_img_dir / img_f.name
        if not dst_img.exists():
            if copy_images:
                shutil.copy2(img_f, dst_img)
            else:
                dst_img.symlink_to(img_f.resolve())

        src_lbl = src_lbl_dir / f"{img_f.stem}.txt"
        dst_lbl = dst_lbl_dir / f"{img_f.stem}.txt"

        new_lines = []
        if src_lbl.exists():
            for line in src_lbl.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                total_boxes += 1
                w, h = float(parts[3]), float(parts[4])
                area = w * h

                # Critère Option A : suppression si w > max_w ou h > max_h ou area > max_area
                if (
                    (max_w is not None and w > max_w)
                    or (max_h is not None and h > max_h)
                    or (max_area is not None and area > max_area)
                ):
                    removed_boxes += 1
                else:
                    kept_boxes += 1
                    new_lines.append(line.strip())

        if not new_lines:
            empty_images += 1
            dst_lbl.write_text("")
        else:
            dst_lbl.write_text("\n".join(new_lines) + "\n")

    print(f"\nCréation de {dataset_dir}/{dst_split} terminée avec succès !")
    print(f"  Images traitées              : {len(img_files)}")
    print(f"  Boîtes initiales             : {total_boxes}")
    print(f"  Boîtes filtrées (supprimées) : {removed_boxes} ({removed_boxes / total_boxes * 100:.2f}%)" if total_boxes else "")
    print(f"  Boîtes conservées            : {kept_boxes} ({kept_boxes / total_boxes * 100:.2f}%)" if total_boxes else "")
    print(f"  Images sans boîte (négatifs) : {empty_images}")


def main():
    parser = argparse.ArgumentParser(description="Préparation des dataset1 de détection.")
    sub = parser.add_subparsers(dest="command")

    p_prep = sub.add_parser("prepare", help="Construit dataset1/ à partir des données de data1.")
    p_prep.set_defaults(func=cmd_prepare)

    p_reorg = sub.add_parser(
        "reorganize",
        help="Réorganise un dataset à plat (dataset1X) en train/val, comme dataset1/.",
    )
    p_reorg.add_argument("dataset_dir", help="Nom du dossier, ex: dataset2")
    p_reorg.set_defaults(func=lambda a: reorganize_flat_dataset(a.dataset_dir))

    p_split = sub.add_parser(
        "split-val",
        help="Découpe un échantillon du test en validation réelle (val_real), disjoint du test.",
    )
    p_split.add_argument("dataset_dir", help="Nom du dossier, ex: dataset2")
    p_split.add_argument("--n", type=int, default=100, help="Nombre d'images de validation réelle.")
    p_split.add_argument("--seed", type=int, default=0, help="Graine aléatoire pour la sélection.")
    p_split.set_defaults(func=lambda a: split_real_val(a.dataset_dir, a.n, a.seed))

    p_crop = sub.add_parser(
        "crop",
        help="Recadre au centre en carré (1024x1024) les images de val_real et test.",
    )
    p_crop.add_argument("dataset_dir", help="Nom du dossier, ex: dataset3")
    p_crop.add_argument("--size", type=int, default=1024, help="Taille cible carrée (ex: 1024).")
    p_crop.set_defaults(func=lambda a: center_crop_dataset(a.dataset_dir, a.size))

    p_filter = sub.add_parser(
        "filter-test",
        help="Filtre les annotations des objets trop gros d'un split de test.",
    )
    p_filter.add_argument("dataset_dir", help="Nom du dossier, ex: dataset3")
    p_filter.add_argument("--src", default="test2", help="Split source (ex: test2).")
    p_filter.add_argument("--dst", default="test3", help="Split cible (ex: test3).")
    p_filter.add_argument("--max-w", type=float, default=0.23125, help="Largeur max autorisée.")
    p_filter.add_argument("--max-h", type=float, default=0.16992, help="Hauteur max autorisée.")
    p_filter.add_argument("--max-area", type=float, default=0.034813, help="Aire max autorisée.")
    p_filter.set_defaults(
        func=lambda a: filter_test_boxes(
            a.dataset_dir,
            src_split=a.src,
            dst_split=a.dst,
            max_w=a.max_w,
            max_h=a.max_h,
            max_area=a.max_area,
        )
    )

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
