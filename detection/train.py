from ultralytics import YOLO
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


import gc
import torch


def train(
    weights_path,
    data_yaml,
    project,
    name,
    epochs=100,
    patience=10,
    imgsz=640,
    batch=16,
    device=0,
    lr0=None,
    lrf=None,
    freeze=None,
    workers=4,
):
    model = YOLO(str(weights_path))
    kwargs = dict(
        data=str(data_yaml),
        epochs=epochs,
        patience=patience,
        imgsz=imgsz,
        batch=batch,
        project=str(project),
        name=name,
        device=device,
        save=True,
        plots=True,
        workers=workers,
    )
    if lr0 is not None:
        kwargs["lr0"] = lr0
    if lrf is not None:
        kwargs["lrf"] = lrf
    if freeze is not None:
        kwargs["freeze"] = freeze

    results = model.train(**kwargs)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def read_best_metrics(results_csv):
    if not results_csv.exists():
        return None
    rows = list(csv.DictReader(results_csv.open()))
    if not rows:
        return None
    best = max(rows, key=lambda r: float(r["metrics/mAP50(B)"] or 0))
    return {
        "epoch": int(best["epoch"]),
        "epochs_done": len(rows),
        "mAP50": float(best["metrics/mAP50(B)"]),
        "mAP50-95": float(best["metrics/mAP50-95(B)"]),
        "precision": float(best["metrics/precision(B)"]),
        "recall": float(best["metrics/recall(B)"]),
        "val_box_loss": float(best["val/box_loss"]),
        "val_cls_loss": float(best["val/cls_loss"]),
        "val_dfl_loss": float(best["val/dfl_loss"]),
    }


def fmt_model_section(title, weights, data):
    lines = [f"=== {title} ===", f"  Poids : {weights}"]
    if data is None:
        lines.append("  (aucun résultat d'entraînement trouvé)")
        return "\n".join(lines)
    lines += [
        f"  Epochs effectuées : {data['epochs_done']}  (meilleure epoch : {data['epoch']})",
        f"  mAP50      : {data['mAP50']:.4f}",
        f"  mAP50-95   : {data['mAP50-95']:.4f}",
        f"  Precision  : {data['precision']:.4f}",
        f"  Recall     : {data['recall']:.4f}",
        f"  Val losses (box/cls/dfl) : {data['val_box_loss']:.4f} / {data['val_cls_loss']:.4f} / {data['val_dfl_loss']:.4f}",
    ]
    return "\n".join(lines)


import sys

def find_run_info(model_dir):
    w = model_dir / "weights" / "best.pt"
    csv_file = model_dir / "results.csv"
    if w.exists() and csv_file.exists():
        return w, read_best_metrics(csv_file)
    if w.exists():
        return w, None
    return model_dir / "weights" / "best.pt", None


def write_params_files(exp_dir, args, exp_name):
    now = datetime.now()
    existing_params = {}
    params_file = exp_dir / "params.json"
    if params_file.exists():
        try:
            existing_params = json.loads(params_file.read_text())
        except Exception:
            pass

    history_cmds = existing_params.get("commands_history", [])
    cmd_str = f"python {' '.join(sys.argv)}"
    if cmd_str not in history_cmds:
        history_cmds.append(cmd_str)

    modes_trained = existing_params.get("modes_trained", [])
    if args.mode not in modes_trained:
        modes_trained.append(args.mode)

    params_dict = {
        "exp_name": exp_name,
        "timestamp": existing_params.get("timestamp", now.isoformat()),
        "last_updated": now.isoformat(),
        "dataset": args.dataset,
        "model": f"yolov8{args.model}.pt",
        "epochs": args.epochs,
        "patience": args.patience,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "freeze": args.freeze,
        "mode": args.mode,
        "modes_trained": modes_trained,
        "val_real": args.val_real,
        "val_real_split": getattr(args, "val_real_split", "val_real") if args.val_real else None,
        "command": cmd_str,
        "commands_history": history_cmds,
    }
    params_file.write_text(json.dumps(params_dict, indent=2))

    lines = [
        "=" * 70,
        f"Paramètres d'entraînement — {exp_name}",
        "=" * 70,
        f"Date        : {params_dict['timestamp']}",
        f"Dernière MAJ: {now.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset     : {args.dataset}",
        f"Modèle base : yolov8{args.model}.pt",
        f"Modes       : {', '.join(modes_trained)}",
        f"Epochs      : {args.epochs} (patience={args.patience})",
        f"Batch size  : {args.batch}",
        f"Image size  : {args.imgsz}",
        f"Device      : {args.device}",
        f"lr0         : {args.lr0 if args.lr0 is not None else 'default'}",
        f"lrf         : {args.lrf if args.lrf is not None else 'default'}",
        f"Freeze      : {f'{args.freeze} couches (backbone gelé)' if args.freeze else 'aucun (tout entraînable)'}",
        f"Validation  : {f'Échantillon réel ({args.val_real_split})' if args.val_real else 'Validation même domaine (val)'}",
        f"Commandes   :",
    ]
    for c in history_cmds:
        lines.append(f"  - {c}")
    lines.append("=" * 70)
    (exp_dir / "params.txt").write_text("\n".join(lines) + "\n")


def write_training_summary(exp_dir, dataset, args):
    models_dir = exp_dir / "models"
    now = datetime.now()

    synth_w, synth_metrics = find_run_info(models_dir / "synthetic")
    real_w, real_metrics = find_run_info(models_dir / "realistic")
    comb_w, comb_metrics = find_run_info(models_dir / "combined")
    curr_w, curr_metrics = find_run_info(models_dir / "curriculum")

    val_split_label = getattr(args, "val_real_split", "val_real")
    val_desc = f"échantillon réel ({val_split_label})" if args.val_real else "val du même domaine"

    rows = [
        "=" * 70,
        f"Récapitulatif d'entraînement — {exp_dir.name}",
        "=" * 70,
        f"Dataset    : {dataset}",
        f"Date       : {now.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Modèle     : yolov8{args.model}.pt",
        f"Epochs     : {args.epochs} (patience {getattr(args, 'patience', 10)})",
        f"Batch      : {args.batch}",
        f"imgsz      : {args.imgsz}",
        f"Device     : {args.device}",
        f"lr0        : {args.lr0 if getattr(args, 'lr0', None) is not None else 'default'}",
        f"Freeze     : {f'{args.freeze} couches gelées' if getattr(args, 'freeze', None) else 'aucun'}",
        f"Validation : {val_desc}",
        "",
        fmt_model_section("Modèle SYNTHÉTIQUE", synth_w, synth_metrics),
        "",
        fmt_model_section("Modèle RÉALISTE", real_w, real_metrics),
        "",
        fmt_model_section("Modèle COMBINÉ (Synthétique + Réaliste)", comb_w, comb_metrics),
        "",
        fmt_model_section("Modèle CURRICULUM (Synthétique -> Réaliste)", curr_w, curr_metrics),
        "",
    ]

    comp_file = exp_dir / "comparison_results.json"
    if comp_file.exists():
        data = json.loads(comp_file.read_text())
        rows += [
            "=" * 70,
            "Évaluation sur le test",
            "=" * 70,
            f"  (fichier : {comp_file.name})",
            f"  Date de l'évaluation : {data.get('timestamp', '?')}",
        ]
        for key, label in [
            ("synthetic_test", "Synthétique"),
            ("realistic_test", "Réaliste"),
            ("combined_test", "Combiné"),
            ("curriculum_test", "Curriculum"),
        ]:
            m = data.get(key)
            if m:
                rows.append(
                    f"  {label:<11} : mAP50={m['mAP50']:.4f}  mAP50-95={m['mAP50-95']:.4f}"
                    f"  Precision={m['precision']:.4f}  Recall={m['recall']:.4f}"
                )
    else:
        rows += [
            "=" * 70,
            "Évaluation sur le test",
            "=" * 70,
            "  Aucune évaluation trouvée (lancer evaluate.py).",
        ]

    summary_file = exp_dir / "training_summary.txt"
    summary_file.write_text("\n".join(rows) + "\n")
    return summary_file


def update_latest_symlink(results_dir, exp_dir):
    latest_link = results_dir / "latest"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(exp_dir.name)
    except Exception as e:
        print(f"Note : impossible de créer le lien latest ({e})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="n", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10, help="Patience d'early stopping (défaut: 10).")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate (e.g. 0.001)")
    parser.add_argument("--lrf", type=float, default=None, help="Final learning rate factor (e.g. 0.01)")
    parser.add_argument(
        "--freeze",
        type=int,
        default=None,
        help="Nombre de couches à geler (ex: 10 pour geler le backbone)",
    )
    parser.add_argument(
        "--dataset",
        default="dataset1",
        help="Dossier du dataset à utiliser (dataset1, dataset2, dataset3...).",
    )
    parser.add_argument(
        "--mode",
        default="all",
        choices=["all", "synthetic", "realistic", "combined", "curriculum"],
        help="Modèle(s) à entraîner : all, synthetic, realistic, combined, ou curriculum (2-étapes).",
    )
    parser.add_argument(
        "--val-real",
        action="store_true",
        help=(
            "Valider l'entraînement sur un échantillon réel du test (val_real), "
            "disjoint du test, pour un early stopping orienté domaine cible."
        ),
    )
    parser.add_argument(
        "--val-real-split",
        type=str,
        default="val_real",
        help="Nom du sous-dossier pour la validation réelle (ex: val_real ou val_real2).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Nom personnalisé pour le dossier de l'expérience dans Results (optionnel).",
    )
    args = parser.parse_args()

    det_dir = Path(__file__).resolve().parent
    dataset = args.dataset
    configs_dir = det_dir / "configs" / dataset
    weights_dir = det_dir / "weights"
    base_weights = weights_dir / f"yolov8{args.model}.pt"

    if args.val_real:
        val_real = det_dir / dataset / args.val_real_split / "images"
        if not val_real.exists() or not any(val_real.iterdir()):
            print(f"Erreur : validation réelle ({args.val_real_split}) absente. Lancez d'abord :")
            print(f"  python prepare_data.py split-val {dataset}")
            return

    # Create dedicated experiment folder in Results
    results_dir = det_dir / dataset / "Results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.name:
        exp_name = args.name
    else:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        flags_parts = [
            f"yolov8{args.model}",
            f"img{args.imgsz}",
            f"ep{args.epochs}",
            f"b{args.batch}",
        ]
        if args.lr0 is not None:
            flags_parts.append(f"lr{args.lr0}")
        if args.lrf is not None:
            flags_parts.append(f"lrf{args.lrf}")
        if args.freeze is not None:
            flags_parts.append(f"freeze{args.freeze}")
        if args.mode != "all":
            flags_parts.append(args.mode)
        if args.val_real:
            val_tag = "valreal" if args.val_real_split == "val_real" else f"valreal{args.val_real_split.removeprefix('val_real')}"
            flags_parts.append(val_tag)
        exp_name = f"exp_{timestamp_str}_{'_'.join(flags_parts)}"

    exp_dir = results_dir / exp_name
    models_dir = exp_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save parameter files and update latest pointer
    write_params_files(exp_dir, args, exp_name)
    update_latest_symlink(results_dir, exp_dir)

    print(f"\n{'='*70}")
    print(f"Nouvelle expérience créée dans : {exp_dir}")
    print(f"{'='*70}\n")

    val_suffix = "_valreal"
    if args.val_real:
        candidates = [
            f"_{args.val_real_split}",
            f"_{args.val_real_split.replace('_', '')}",
            f"_{args.val_real_split.replace('val_real', 'valreal')}",
            "_valreal",
        ]
        for c in candidates:
            if (configs_dir / f"synthetic{c}.yaml").exists():
                val_suffix = c
                break

    synth_cfg = configs_dir / (f"synthetic{val_suffix}.yaml" if args.val_real else "synthetic.yaml")
    real_cfg = configs_dir / (f"realistic{val_suffix}.yaml" if args.val_real else "realistic.yaml")
    comb_cfg = configs_dir / (f"combined{val_suffix}.yaml" if args.val_real else "combined.yaml")

    print(f"Configurations utilisées :")
    print(f"  - Synthétique : {synth_cfg.name}")
    print(f"  - Réaliste    : {real_cfg.name}")
    print(f"  - Combiné     : {comb_cfg.name}\n")

    if args.mode in ["all", "synthetic"]:
        synth_best = models_dir / "synthetic" / "weights" / "best.pt"
        if synth_best.exists() and (models_dir / "synthetic" / "results.csv").exists():
            print(f"\n=== Found existing SYNTHETIC checkpoint: {synth_best} (skipping) ===")
        else:
            print(f"\n=== Training on SYNTHETIC data ({dataset}) ===")
            train(
                base_weights,
                synth_cfg,
                models_dir,
                "synthetic",
                epochs=args.epochs,
                patience=args.patience,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                lr0=args.lr0,
                lrf=args.lrf,
                freeze=args.freeze,
            )

    if args.mode in ["all", "realistic"]:
        real_best = models_dir / "realistic" / "weights" / "best.pt"
        if real_best.exists() and (models_dir / "realistic" / "results.csv").exists():
            print(f"\n=== Found existing REALISTIC checkpoint: {real_best} (skipping) ===")
        else:
            print(f"\n=== Training on REALISTIC data ({dataset}) ===")
            train(
                base_weights,
                real_cfg,
                models_dir,
                "realistic",
                epochs=args.epochs,
                patience=args.patience,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                lr0=args.lr0,
                lrf=args.lrf,
                freeze=args.freeze,
            )

    if args.mode in ["all", "combined"]:
        comb_best = models_dir / "combined" / "weights" / "best.pt"
        if comb_best.exists() and (models_dir / "combined" / "results.csv").exists():
            print(f"\n=== Found existing COMBINED checkpoint: {comb_best} (skipping) ===")
        else:
            print(f"\n=== Training on COMBINED data ({dataset}) ===")
            train(
                base_weights,
                comb_cfg,
                models_dir,
                "combined",
                epochs=args.epochs,
                patience=args.patience,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                lr0=args.lr0,
                lrf=args.lrf,
                freeze=args.freeze,
            )

    if args.mode == "curriculum":
        synth_best = models_dir / "synthetic" / "weights" / "best.pt"
        if not synth_best.exists():
            print(f"\n=== Curriculum Step 1: Pre-training on SYNTHETIC data ({dataset}) ===")
            train(
                base_weights,
                synth_cfg,
                models_dir,
                "synthetic",
                epochs=args.epochs,
                patience=args.patience,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                lr0=args.lr0,
                lrf=args.lrf,
                freeze=args.freeze,
            )
        else:
            print(f"\n=== Curriculum Step 1: Found existing synthetic checkpoint: {synth_best} ===")

        curr_lr0 = args.lr0 if args.lr0 is not None else 0.001
        print(f"\n=== Curriculum Step 2: Fine-tuning on REALISTIC data with lr0={curr_lr0} ({dataset}) ===")
        train(
            synth_best,
            real_cfg,
            models_dir,
            "curriculum",
            epochs=args.epochs,
            patience=args.patience,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            lr0=curr_lr0,
            lrf=args.lrf,
            freeze=args.freeze,
        )

    summary_file = write_training_summary(exp_dir, dataset, args)

    print(f"\n{'='*70}")
    print(f"Entraînement terminé !")
    print(f"Tous les résultats sont sauvegardés dans : {exp_dir}")
    print(f"  - Modèles & poids : {models_dir}/")
    print(f"  - Paramètres      : {exp_dir}/params.json & params.txt")
    print(f"  - Résumé          : {summary_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
