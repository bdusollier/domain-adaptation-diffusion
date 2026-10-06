from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou, ap_per_class
import torch
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime


def evaluate(weights, data_yaml, split="val", out_dir=None, name="eval"):
    model = YOLO(str(weights))
    kwargs = dict(data=str(data_yaml), split=split, save_json=True, plots=True)
    if out_dir is not None:
        kwargs["project"] = str(out_dir)
        kwargs["name"] = name
    return model.val(**kwargs)


def evaluate_coco_baseline(coco_weights, test_img_dir, test_lbl_dir, imgsz=1024):
    model = YOLO(str(coco_weights))
    # COCO vehicle classes: car(2), motorcycle(3), airplane(4), bus(5), train(6), truck(7), boat(8)
    vehicle_classes = {2, 3, 4, 5, 6, 7, 8}
    
    correct_list = []
    confs_list = []
    pred_cls_list = []
    target_cls_list = []

    IMG_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    img_files = sorted(f for f in test_img_dir.glob("*") if f.suffix.lower() in IMG_SUFFIXES)
    for img_path in img_files:
        stem = img_path.stem
        lbl_path = test_lbl_dir / f"{stem}.txt"
        
        gt_boxes = []
        gt_cls = []
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
                    gt_boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                    gt_cls.append(0)
        
        gt_boxes = torch.tensor(gt_boxes) if gt_boxes else torch.zeros((0, 4))
        gt_cls = torch.tensor(gt_cls) if gt_cls else torch.zeros((0,))
        target_cls_list.append(gt_cls)
        
        results = model(str(img_path), verbose=False, imgsz=imgsz)[0]
        pred_boxes = []
        pred_confs = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in vehicle_classes:
                pred_boxes.append(box.xyxyn[0].tolist())
                pred_confs.append(float(box.conf[0]))
                
        pred_boxes = torch.tensor(pred_boxes) if pred_boxes else torch.zeros((0, 4))
        pred_confs = torch.tensor(pred_confs) if pred_confs else torch.zeros((0,))
        
        nl = len(gt_cls)
        npr = len(pred_confs)
        if npr == 0:
            continue
            
        confs_list.append(pred_confs)
        pred_cls_list.append(torch.zeros(npr))
        
        if nl == 0:
            correct_list.append(torch.zeros((npr, 10), dtype=torch.bool))
            continue
            
        ious = box_iou(pred_boxes, gt_boxes)
        iouv = torch.linspace(0.5, 0.95, 10)
        correct = torch.zeros((npr, 10), dtype=torch.bool)
        for i, iou_thresh in enumerate(iouv):
            matches = (ious >= iou_thresh).nonzero(as_tuple=False)
            if matches.shape[0] > 0:
                match_ious = ious[matches[:, 0], matches[:, 1]]
                sorted_idx = match_ious.argsort(descending=True)
                matches = matches[sorted_idx]
                pred_used = set()
                gt_used = set()
                for p_idx, g_idx in matches.tolist():
                    if p_idx not in pred_used and g_idx not in gt_used:
                        pred_used.add(p_idx)
                        gt_used.add(g_idx)
                        correct[p_idx, i] = True
        correct_list.append(correct)

    if not correct_list:
        return {"mAP50": 0.0, "mAP50-95": 0.0, "precision": 0.0, "recall": 0.0}

    correct_all = torch.cat(correct_list, 0)
    confs_all = torch.cat(confs_list, 0)
    pred_cls_all = torch.cat(pred_cls_list, 0)
    target_cls_all = torch.cat(target_cls_list, 0)

    res = ap_per_class(
        correct_all.numpy(),
        confs_all.numpy(),
        pred_cls_all.numpy(),
        target_cls_all.numpy(),
        plot=False,
    )
    p, r, ap = res[2], res[3], res[5]
    return {
        "mAP50": round(float(ap[:, 0].mean()), 4),
        "mAP50-95": round(float(ap.mean()), 4),
        "precision": round(float(p.mean()), 4),
        "recall": round(float(r.mean()), 4),
    }


def metrics_dict(results):
    d = results.results_dict
    return {
        "mAP50": round(d["metrics/mAP50(B)"], 4),
        "mAP50-95": round(d["metrics/mAP50-95(B)"], 4),
        "precision": round(d["metrics/precision(B)"], 4),
        "recall": round(d["metrics/recall(B)"], 4),
    }


def print_table(title, rows, headers):
    col_w = 12
    hdr_line = f"  {'Metric':<12}" + "".join(f"{h:>{col_w}}" for h in headers)
    sep_line = "  " + "-" * (len(hdr_line) - 2)
    print(f"\n{'='*len(hdr_line)}")
    print(f"  {title}")
    print(f"{'='*len(hdr_line)}")
    print(hdr_line)
    print(sep_line)
    for row in rows:
        metric_name = row[0]
        vals = row[1:]
        val_strs = "".join(f"{v:>{col_w}.4f}" if isinstance(v, float) else f"{str(v):>{col_w}}" for v in vals)
        print(f"  {metric_name:<12}{val_strs}")
    print()


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

    # Check latest symlink
    latest_link = results_dir / "latest"
    if latest_link.exists():
        return latest_link.resolve()

    # Find latest exp_* folder
    exp_folders = [d for d in results_dir.glob("exp_*") if d.is_dir()]
    if exp_folders:
        exp_folders.sort(key=lambda x: x.stat().st_mtime)
        return exp_folders[-1]

    return None


def update_training_summary_test(summary_file, summary_data):
    if not summary_file.exists():
        return
    text = summary_file.read_text()
    
    test_lines = [
        "=" * 70,
        "Évaluation sur le test",
        "=" * 70,
        f"  Date de l'évaluation : {summary_data.get('timestamp', '?')}",
    ]
    for key, label in [
        ("synthetic_test", "Synthétique"),
        ("realistic_test", "Réaliste"),
        ("combined_test", "Combiné"),
        ("curriculum_test", "Curriculum"),
    ]:
        m = summary_data.get(key)
        if m:
            test_lines.append(
                f"  {label:<11} : mAP50={m['mAP50']:.4f}  mAP50-95={m['mAP50-95']:.4f}"
                f"  Precision={m['precision']:.4f}  Recall={m['recall']:.4f}"
            )
    test_section = "\n".join(test_lines)

    if "=" * 70 + "\nÉvaluation sur le test" in text:
        parts = text.split("=" * 70 + "\nÉvaluation sur le test")
        new_text = parts[0].rstrip() + "\n\n" + test_section + "\n"
    else:
        new_text = text.rstrip() + "\n\n" + test_section + "\n"

    summary_file.write_text(new_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        type=str,
        default=None,
        help="Dossier ou nom de l'expérience dans Results (par défaut: dernière expérience).",
    )
    parser.add_argument("--weights-synth", type=str, default=None)
    parser.add_argument("--weights-real", type=str, default=None)
    parser.add_argument("--weights-comb", type=str, default=None)
    parser.add_argument("--weights-curr", type=str, default=None)
    parser.add_argument(
        "--dataset",
        default="dataset1",
        help="Dossier du dataset à utiliser (dataset1, dataset2, dataset3...).",
    )
    parser.add_argument(
        "--test-split",
        type=str,
        default=None,
        help="Nom du sous-dossier ou config de test (ex: test ou test2).",
    )
    parser.add_argument(
        "--val-real",
        action="store_true",
        help="Évaluer les modèles avec suffixe -valreal (mode legacy).",
    )
    args = parser.parse_args()

    det_dir = Path(__file__).resolve().parent
    dataset = args.dataset
    configs_dir = det_dir / "configs" / dataset

    results_dir = det_dir / dataset / "Results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Determine test split and yaml config
    if args.test_split:
        test_split = args.test_split
    elif (configs_dir / "test.yaml").exists():
        test_split = "test"
    elif (configs_dir / "test2.yaml").exists():
        test_split = "test2"
    else:
        test_split = "test"

    test_cfg = configs_dir / f"{test_split}.yaml"
    if not test_cfg.exists():
        test_cfg = configs_dir / "test.yaml"

    exp_dir = resolve_exp_dir(results_dir, args.exp)
    if exp_dir:
        print(f"\nUtilisation de l'expérience : {exp_dir.name}")
        models_dir = exp_dir / "models"
        eval_dir = exp_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)

        synth_w = Path(args.weights_synth) if args.weights_synth else (models_dir / "synthetic" / "weights" / "best.pt")
        real_w = Path(args.weights_real) if args.weights_real else (models_dir / "realistic" / "weights" / "best.pt")
        comb_w = Path(args.weights_comb) if args.weights_comb else (models_dir / "combined" / "weights" / "best.pt")
        curr_w = Path(args.weights_curr) if args.weights_curr else (models_dir / "curriculum" / "weights" / "best.pt")
        out_project = eval_dir
    else:
        # Legacy fallback
        print("\nAucun dossier d'expérience 'exp_*' trouvé, bascule sur le dossier legacy runs/...")
        runs_dir = det_dir / "runs" / dataset
        suffix = "-valreal" if args.val_real else ""
        synth_w = Path(args.weights_synth) if args.weights_synth else find_latest_weights(runs_dir, f"synthetic{suffix}")
        real_w = Path(args.weights_real) if args.weights_real else find_latest_weights(runs_dir, f"realistic{suffix}")
        comb_w = Path(args.weights_comb) if args.weights_comb else find_latest_weights(runs_dir, f"combined{suffix}")
        curr_w = Path(args.weights_curr) if args.weights_curr else find_latest_weights(runs_dir, f"curriculum{suffix}")
        out_project = results_dir

    models_to_eval = []
    if synth_w.exists():
        models_to_eval.append(("synthetic", "Synthetic", synth_w, "synthetic"))
    else:
        print(f"Note: Synthetic weights not found ({synth_w})")

    if real_w.exists():
        models_to_eval.append(("realistic", "Realistic", real_w, "realistic"))
    else:
        print(f"Note: Realistic weights not found ({real_w})")

    if comb_w.exists():
        models_to_eval.append(("combined", "Combined", comb_w, "combined"))
    else:
        print(f"Note: Combined weights not found ({comb_w})")

    if curr_w.exists():
        models_to_eval.append(("curriculum", "Curriculum", curr_w, "curriculum"))
    else:
        print(f"Note: Curriculum weights not found ({curr_w})")

    if not models_to_eval:
        print("Erreur : aucun modèle trouvé à évaluer.")
        return

    results_by_model = {}
    for key, label, weights_path, run_subname in models_to_eval:
        print(f"\nEvaluating {label} model ({weights_path}) on test set ({test_cfg.name})...")
        eval_res = evaluate(weights_path, test_cfg, "val", out_project, name=run_subname)
        results_by_model[key] = metrics_dict(eval_res)

    coco_weights = det_dir / "weights" / "yolov8s.pt"
    if not coco_weights.exists():
        coco_weights = det_dir / "weights" / "yolov8n.pt"

    coco_baseline_res = None
    if coco_weights.exists():
        test_img_dir = det_dir / dataset / test_split / "images"
        test_lbl_dir = det_dir / dataset / test_split / "labels"
        if not test_img_dir.exists():
            test_img_dir = det_dir / dataset / "test" / "images"
            test_lbl_dir = det_dir / dataset / "test" / "labels"
        if test_img_dir.exists() and test_lbl_dir.exists():
            print(f"\nEvaluating COCO Baseline ({coco_weights.name}) on test set ({test_split})...")
            coco_baseline_res = evaluate_coco_baseline(coco_weights, test_img_dir, test_lbl_dir)
            results_by_model["coco_baseline"] = coco_baseline_res

    headers = [label for _, label, _, _ in models_to_eval]
    if coco_baseline_res:
        headers.insert(0, "COCO-Base")
    if len(headers) > 1:
        headers.append("Best")

    table_rows = []
    for metric in ["mAP50", "mAP50-95", "precision", "recall"]:
        row = [metric.capitalize() if metric != "mAP50-95" else "mAP50-95"]
        best_val = -1
        best_label = ""
        
        if coco_baseline_res:
            val = coco_baseline_res[metric]
            row.append(val)
            if val > best_val:
                best_val = val
                best_label = "COCO-Base"

        for key, label, _, _ in models_to_eval:
            val = results_by_model[key][metric]
            row.append(val)
            if val > best_val:
                best_val = val
                best_label = label
        if len(headers) > 1:
            row.append(best_label)
        table_rows.append(row)

    exp_title = exp_dir.name if exp_dir else dataset
    print_table(f"Test set Evaluation — {exp_title}", table_rows, headers)

    now = datetime.now()
    summary = {
        "dataset": dataset,
        "exp_name": exp_dir.name if exp_dir else "legacy",
        "timestamp": now.isoformat(),
    }
    if coco_baseline_res:
        summary["coco_baseline"] = coco_baseline_res
    if "synthetic" in results_by_model:
        summary["synthetic_test"] = results_by_model["synthetic"]
    if "realistic" in results_by_model:
        summary["realistic_test"] = results_by_model["realistic"]
    if "combined" in results_by_model:
        summary["combined_test"] = results_by_model["combined"]
    if "curriculum" in results_by_model:
        summary["curriculum_test"] = results_by_model["curriculum"]

    if exp_dir:
        # Save JSON directly in experiment folder
        exp_json = exp_dir / "comparison_results.json"
        exp_json.write_text(json.dumps(summary, indent=2))
        
        # Update summary in experiment folder
        update_training_summary_test(exp_dir / "training_summary.txt", summary)
        print(f"Results saved to: {exp_json}")
    else:
        legacy_json = results_dir / "comparison_results.json"
        legacy_json.write_text(json.dumps(summary, indent=2))
        print(f"Results saved to: {legacy_json}")

    # Append to global history log
    history_file = results_dir / "evaluation_history.jsonl"
    with open(history_file, "a") as hf:
        hf.write(json.dumps(summary) + "\n")
    print(f"Evaluation appended to: {history_file}")


if __name__ == "__main__":
    main()
