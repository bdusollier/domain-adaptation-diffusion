import cv2
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def load_labels(label_path):
    boxes = []
    if label_path.exists():
        for line in label_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
                boxes.append((cls, cx, cy, w, h))
    return boxes


def draw_boxes(img, boxes, color, class_names=None, conf=None):
    h, w = img.shape[:2]
    for i, (cls, cx, cy, bw, bh) in enumerate(boxes):
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = class_names[cls] if class_names and cls < len(class_names) else str(cls)
        if conf is not None:
            label = f"{label} {conf[i]:.2f}"
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def run_inference(model, img_path, is_coco=False):
    results = model(str(img_path), verbose=False)[0]
    boxes = []
    confs = []
    vehicle_classes = {2, 3, 4, 5, 6, 7, 8}
    for box in results.boxes:
        cls = int(box.cls[0])
        if is_coco and cls not in vehicle_classes:
            continue
        cx, cy, bw, bh = box.xywhn[0].tolist()
        c = float(box.conf[0])
        boxes.append((0 if is_coco else cls, cx, cy, bw, bh))
        confs.append(c)
    return boxes, confs


def find_latest_weights(runs_dir, prefix):
    candidates = []
    for d in runs_dir.glob(f"{prefix}*"):
        if not d.is_dir():
            continue
        w = d / "weights" / "best.pt"
        if w.exists():
            candidates.append((w.stat().st_mtime, w))
    if not candidates:
        return runs_dir / prefix / "weights" / "best.pt"
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def resolve_exp_dir(results_dir, exp_arg=None):
    if exp_arg:
        candidate = Path(exp_arg)
        if candidate.exists():
            return candidate.resolve()
        candidate = results_dir / exp_arg
        if candidate.exists():
            return candidate.resolve()
        print(f"Attention : dossier d'expérience '{exp_arg}' introuvable.")
        return None

    latest_link = results_dir / "latest"
    if latest_link.exists():
        return latest_link.resolve()

    exp_folders = [d for d in results_dir.glob("exp_*") if d.is_dir()]
    if exp_folders:
        exp_folders.sort(key=lambda x: x.stat().st_mtime)
        return exp_folders[-1]

    return None


def main():
    parser = argparse.ArgumentParser(description="Génère des comparaisons GT vs Synth vs Réel.")
    parser.add_argument(
        "--exp",
        type=str,
        default=None,
        help="Dossier ou nom de l'expérience dans Results (par défaut: dernière expérience).",
    )
    parser.add_argument(
        "--dataset",
        default="dataset1",
        help="Dossier du dataset à utiliser (dataset1, dataset2, dataset3...).",
    )
    parser.add_argument(
        "--test-split",
        type=str,
        default=None,
        help="Sous-dossier de test (ex: test ou test2).",
    )
    parser.add_argument("--n", type=int, default=50, help="Nombre d'images à traiter.")
    parser.add_argument(
        "--val-real",
        action="store_true",
        help="Utiliser les modèles entraînés avec validation réelle (mode legacy).",
    )
    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        help="Ne pas inclure le modèle Curriculum dans les comparaisons.",
    )
    parser.add_argument(
        "--no-coco",
        action="store_true",
        help="Ne pas inclure le modèle COCO de base.",
    )
    args = parser.parse_args()

    det_dir = Path(__file__).resolve().parent
    dataset = args.dataset

    if args.test_split:
        test_split = args.test_split
    elif (det_dir / dataset / "test2").exists() and not (det_dir / dataset / "test").exists():
        test_split = "test2"
    else:
        test_split = "test"

    test_dir = det_dir / dataset / test_split
    images_dir = test_dir / "images"
    labels_dir = test_dir / "labels"

    results_dir = det_dir / dataset / "Results"
    results_dir.mkdir(parents=True, exist_ok=True)

    exp_dir = resolve_exp_dir(results_dir, args.exp)
    if exp_dir:
        print(f"\nVisualisation pour l'expérience : {exp_dir.name}")
        models_dir = exp_dir / "models"
        synth_p = models_dir / "synthetic" / "weights" / "best.pt"
        real_p = models_dir / "realistic" / "weights" / "best.pt"
        comb_p = models_dir / "combined" / "weights" / "best.pt"
        curr_p = models_dir / "curriculum" / "weights" / "best.pt"
        out_dir = exp_dir / "comparisons"
    else:
        suffix = "-valreal" if args.val_real else ""
        runs_dir = det_dir / "runs" / dataset
        synth_p = find_latest_weights(runs_dir, f"synthetic{suffix}")
        real_p = find_latest_weights(runs_dir, f"realistic{suffix}")
        comb_p = find_latest_weights(runs_dir, f"combined{suffix}")
        curr_p = find_latest_weights(runs_dir, f"curriculum{suffix}")
        out_dir = results_dir / ("comparisons_realval" if args.val_real else "comparisons")
    out_dir.mkdir(parents=True, exist_ok=True)

    active_models = []
    coco_p = det_dir / "weights" / "yolov8s.pt"
    if not coco_p.exists():
        coco_p = det_dir / "weights" / "yolov8n.pt"
    if coco_p.exists() and not args.no_coco:
        active_models.append(("COCO-Base", YOLO(coco_p), (200, 200, 200), True))

    if synth_p.exists():
        active_models.append(("Synthetic", YOLO(synth_p), (255, 0, 0), False))
    if real_p.exists():
        active_models.append(("Realistic", YOLO(real_p), (0, 0, 255), False))
    if comb_p.exists():
        active_models.append(("Combined", YOLO(comb_p), (255, 0, 255), False))
    if curr_p.exists() and not args.no_curriculum:
        active_models.append(("Curriculum", YOLO(curr_p), (0, 215, 255), False))

    if not active_models:
        print("Erreur : aucun modèle trouvé pour la visualisation.")
        return

    class_names = ["target"]
    IMG_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    all_imgs = sorted(f for f in images_dir.glob("*") if f.suffix.lower() in IMG_SUFFIXES)

    from collections import defaultdict
    import math

    # Filtrer les images avec au moins une boîte annotée
    valid_imgs = []
    for f in all_imgs:
        lbl_f = labels_dir / f"{f.stem}.txt"
        if lbl_f.exists() and lbl_f.stat().st_size > 0:
            lines = [l for l in lbl_f.read_text().splitlines() if l.strip()]
            if lines:
                valid_imgs.append(f)

    if not valid_imgs:
        valid_imgs = all_imgs

    # Regrouper par séquence/scène
    seq_map = defaultdict(list)
    for f in valid_imgs:
        s = f.stem
        seq = s.split("_frame_")[0] if "_frame_" in s else s.rsplit("_", 1)[0]
        seq_map[seq].append(f)

    # Échantillonnage stratifié équitable
    per_seq = math.ceil(args.n / max(1, len(seq_map)))
    img_files = []
    for seq, imgs in sorted(seq_map.items()):
        imgs.sort()
        step = max(1, len(imgs) // per_seq)
        img_files.extend(imgs[::step][:per_seq])

    img_files = img_files[: args.n]
    print(f"Sélection de {len(img_files)} images variées réparties sur {len(seq_map)} séquences distinctes.")

    for i, img_path in enumerate(img_files):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        stem = img_path.stem
        label_path = labels_dir / (stem + ".txt")

        gt_boxes = load_labels(label_path)
        gt_img = img.copy()
        draw_boxes(gt_img, gt_boxes, (0, 255, 0), class_names)

        panels = [gt_img]
        panel_labels = [("Ground Truth", (0, 255, 0))]

        for name, model, color, is_coco in active_models:
            m_boxes, m_confs = run_inference(model, img_path, is_coco=is_coco)
            m_img = img.copy()
            draw_boxes(m_img, m_boxes, color, class_names, m_confs)
            panels.append(m_img)
            panel_labels.append((name, color))

        h, w = img.shape[:2]
        separator = np.ones((h, 3, 3), dtype=np.uint8) * 255

        stacked_elements = []
        for idx, p in enumerate(panels):
            if idx > 0:
                stacked_elements.append(separator)
            stacked_elements.append(p)

        concat = np.hstack(stacked_elements)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        section_w = w + 3

        for j, (label, color) in enumerate(panel_labels):
            tx = j * section_w + 10
            cv2.putText(concat, label, (tx, 25), font, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(concat, label, (tx, 25), font, font_scale, color, thickness)

        out_path = out_dir / f"comparison_{i:02d}.jpg"
        cv2.imwrite(str(out_path), concat)
        print(f"[{i+1}/{len(img_files)}] {out_path.name}")

    print(f"\nDone. Saved to {out_dir}")


if __name__ == "__main__":
    main()
