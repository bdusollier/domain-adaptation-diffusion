import os
import glob
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from scipy.stats import gaussian_kde

from PIL import Image
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from torch_fidelity.feature_extractor_dinov2 import FeatureExtractorDinoV2, MODEL_METADATA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Chemins des images (synthétiques) et des images générées par dataset
def get_dataset_paths(dataset: str):
    candidates_initial = [
        os.path.join(BASE_DIR, "data", dataset, "input", "images"),
        os.path.join(BASE_DIR, "data", dataset, "input", "image"),
        os.path.join(BASE_DIR, "data", dataset, "input"),
        os.path.join(BASE_DIR, "..", dataset, "input", "images"),
        os.path.join(BASE_DIR, "..", dataset, "input", "image"),
        os.path.join(BASE_DIR, "..", dataset, "input"),
    ]
    candidates_output = [
        os.path.join(BASE_DIR, "data", dataset, "output"),
        os.path.join(BASE_DIR, "..", dataset, "output"),
    ]
    initial_dir = next((c for c in candidates_initial if os.path.exists(c)), candidates_initial[0])
    output_dir = next((c for c in candidates_output if os.path.exists(c)), candidates_output[0])
    return initial_dir, output_dir

def get_ref_dir(dataset: str, user_ref: str = None):
    if user_ref:
        return user_ref if os.path.isabs(user_ref) else os.path.join(BASE_DIR, user_ref)
    candidates = [
        os.path.join(BASE_DIR, "Reference"),
        os.path.join(BASE_DIR, "Ref2" if dataset == "data3" else "Ref_filtered"),
        os.path.join(BASE_DIR, "Ref_filtered"),
        os.path.join(BASE_DIR, "Ref2"),
    ]
    for c in candidates:
        if os.path.exists(c) and (os.path.isdir(c) and len(os.listdir(c)) > 1):
            return c
    return os.path.join(BASE_DIR, "Reference")

RESULTS_DIR = os.path.join(BASE_DIR, "Results_measures")

COLORS = {
    "reference": "#2ecc71",
    "initial": "#e74c3c",
    "output": "#3498db",
}

VALID_EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')


def list_images(directory):
    paths = [p for p in glob.glob(os.path.join(directory, "*")) if p.lower().endswith(VALID_EXTS)]
    paths.sort()
    if len(paths) == 0:
        raise ValueError(f"Aucune image valide trouvée dans {directory}")
    return paths


def extract_inception_features(directory, extractor, device, batch_size=32):
    """Extrait les caractéristiques Inception-v3 (couche pool3, 2048 dims).

    Identique à l'espace de représentation utilisé par le FID (torch_fidelity) :
    entrée uint8, redimensionnement bilinéaire interne à 299x299, normalisation.
    """
    paths = list_images(directory)
    all_features = []

    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        batch_tensors = []

        for path in batch_paths:
            try:
                with Image.open(path).convert('RGB') as img:
                    img = img.resize((299, 299), Image.BILINEAR)
                    batch_tensors.append(torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1))
            except Exception as e:
                print(f"Impossible de charger l'image {path}: {e}")

        if not batch_tensors:
            continue

        x = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            feats = extractor(x)[0].cpu().numpy()
        all_features.append(feats)

    return np.concatenate(all_features, axis=0)


def extract_gram_features(directory, extractor, device, batch_size=4):
    """Extrait les vecteurs de Gram VGG19 (relu4_1), espace du Gram-MMD."""
    from Gram_MMD import extract_features_batch
    return extract_features_batch(directory, extractor, device, batch_size=batch_size)


def extract_dinov2_features(directory, extractor, device, batch_size=32):
    """Extrait les embeddings DINOv2 (token de classe, sans tête de classification).

    L'extracteur (torch_fidelity) redimensionne en 224x224 et normalise selon
    ImageNet en interne ; on pré-redimensionne en 224 pour pouvoir batchter.
    """
    paths = list_images(directory)
    all_features = []

    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        batch_tensors = []

        for path in batch_paths:
            try:
                with Image.open(path).convert('RGB') as img:
                    img = img.resize((224, 224), Image.BILINEAR)
                    batch_tensors.append(torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1))
            except Exception as e:
                print(f"Impossible de charger l'image {path}: {e}")

        if not batch_tensors:
            continue

        x = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            feats = extractor(x)[0].cpu().numpy()
        all_features.append(feats)

    return np.concatenate(all_features, axis=0)


def normalize_features(features, norm="std"):
    """Normalisation des caractéristiques avant ACP (évite qu'une seule
    dimension à forte variance écrase la projection).

    - 'std' : centrage-réduction globale (StandardScaler).
    - 'l2'  : normalisation L2 de chaque vecteur image (chaque image sur la sphère unité).
    - 'none': aucune normalisation (ACP sur la covariance brute).
    """
    if norm == "std":
        mu = features.mean(axis=0, keepdims=True)
        sigma = features.std(axis=0, keepdims=True) + 1e-6
        return (features - mu) / sigma
    if norm == "l2":
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-6
        return features / norms
    return features


def fit_pca(features):
    """ACP (2 premières composantes) sur les caractéristiques, économique en mémoire.

    La normalisation (std / l2 / aucune) est appliquée avant appel via
    normalize_features ; ici on centre simplement les données puis on
    diagonalise la matrice la plus petite entre covariance DxD et Gram NxN.

    Pour les données « larges » (D >> N, ex. caractéristiques de Gram 131 328 dims),
    on diagonalise la matrice de Gram NxN au lieu de la matrice de covariance DxD,
    ce qui évite de matérialiser une matrice DxD ou une SVD complète en mémoire.
    """
    X = features - features.mean(axis=0, keepdims=True)

    N, D = X.shape
    X32 = X.astype(np.float32)

    if D <= N:
        # Matrice de covariance DxD (petite quand D <= N)
        cov = X32.T @ X32
        evals, evecs = np.linalg.eigh(cov.astype(np.float64))
        idx = np.argsort(evals)[::-1]
        scores = X32 @ evecs[:, idx[:2]].astype(np.float32)
        variance = evals[idx[:2]]
    else:
        # Matrice de Gram NxN (petite quand D >> N)
        gram = X32 @ X32.T
        evals, evecs = np.linalg.eigh(gram.astype(np.float64))
        idx = np.argsort(evals)[::-1]
        scores = evecs[:, idx[:2]].astype(np.float32) * np.sqrt(evals[idx[:2]])
        variance = evals[idx[:2]]

    explained = variance / evals.sum() * 100.0
    return scores, explained


def subsample(pts, n, rng):
    if n > 0 and len(pts) > n:
        idx = rng.choice(len(pts), size=n, replace=False)
        return pts[idx]
    return pts


def feature_cache_path(dataset, features, model=None, ref_name=None):
    cache_dir = os.path.join(RESULTS_DIR, dataset, "cache")
    ref_suffix = f"_{ref_name}" if ref_name else ""
    name = f"features_{features}" + (f"_{model}" if model else "") + ref_suffix
    return os.path.join(cache_dir, name + ".npz")


def load_features_from_cache(path):
    if os.path.exists(path):
        data = np.load(path)
        return data["feats_ref"], data["feats_init"], data["feats_out"]
    return None


def save_features_to_cache(path, feats_ref, feats_init, feats_out):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, feats_ref=feats_ref, feats_init=feats_init, feats_out=feats_out)


def filter_class_outliers(pts, percentile=0.0):
    """Filtre les points extrêmes (outliers) d'une classe selon la distance de Mahalanobis dans le plan 2D."""
    if percentile <= 0.0 or len(pts) <= 10:
        return pts
    mu = np.median(pts, axis=0)
    diff = pts - mu
    cov_cls = np.cov(pts.T) + 1e-6 * np.eye(2)
    inv_cov = np.linalg.pinv(cov_cls)
    d2 = np.sum((diff @ inv_cov) * diff, axis=1)
    thresh = np.percentile(d2, 100.0 - percentile)
    return pts[d2 <= thresh]


def plot_pca(scores_dict, explained, output_path, max_points=300, kde_points=2000,
             norm="std", clip=0.0, filter_outliers=0.0, rng_seed=0):
    rng = np.random.default_rng(rng_seed)

    if filter_outliers > 0.0:
        scores_dict = {k: filter_class_outliers(v, filter_outliers) for k, v in scores_dict.items()}

    all_scores = np.concatenate(list(scores_dict.values()))
    # Limites robustes basées sur les percentiles [0.5, 99.5]
    xmin, xmax = np.percentile(all_scores[:, 0], [0.5, 99.5])
    ymin, ymax = np.percentile(all_scores[:, 1], [0.5, 99.5])
    mx = max(0.08 * (xmax - xmin), 0.05)
    my = max(0.08 * (ymax - ymin), 0.05)
    xmin, xmax = xmin - mx, xmax + mx
    ymin, ymax = ymin - my, ymax + my

    xg = np.linspace(xmin, xmax, 200)
    yg = np.linspace(ymin, ymax, 200)
    Xg, Yg = np.meshgrid(xg, yg)
    grid = np.vstack([Xg.ravel(), Yg.ravel()])

    fig, ax = plt.subplots(figsize=(9, 7))
    handles = []

    for key, label in [("reference", "Référence (réel)"),
                       ("initial", "Initial (synthétique)"),
                       ("output", "Amélioré (output)")]:
        pts = scores_dict[key]
        color = COLORS[key]

        # Surface de densité (KDE) : la classe est mieux lisible qu'un nuage dense
        kde_pts = subsample(pts, kde_points, rng)
        try:
            kde = gaussian_kde(kde_pts.T)
            Z = kde(grid).reshape(Xg.shape)
            cmap = LinearSegmentedColormap.from_list(key, [(0, 0, 0, 0), to_rgba(color, 0.9)])
            ax.contourf(Xg, Yg, Z, levels=10, cmap=cmap)
            ax.contour(Xg, Yg, Z, levels=10, colors=color, linewidths=0.6, alpha=0.7)
        except Exception:
            pass

        # Nuage de points sous-échantillonné
        scatter_pts = subsample(pts, max_points, rng)
        # Affichage seulement des points dans la fenêtre visible
        mask = (scatter_pts[:, 0] >= xmin) & (scatter_pts[:, 0] <= xmax) & \
               (scatter_pts[:, 1] >= ymin) & (scatter_pts[:, 1] <= ymax)
        scatter_pts = scatter_pts[mask]
        ax.scatter(scatter_pts[:, 0], scatter_pts[:, 1], s=6, alpha=0.5, color=color)

        handles.append(mpatches.Patch(color=color, label=f"{label} (n={len(pts)})"))

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    norm_notes = {"std": " (caractéristiques standardisées)",
                  "l2": " (caractéristiques normalisées L2)",
                  "none": ""}
    clip_note = f" — bornées [{clip:.0f},{100 - clip:.0f}]" if clip > 0 else ""
    outlier_note = f" — sans {filter_outliers:.0f}% outliers" if filter_outliers > 0 else ""
    ax.set_xlabel(f"Composante principale 1 ({explained[0]:.1f} % de variance)")
    ax.set_ylabel(f"Composante principale 2 ({explained[1]:.1f} % de variance)")
    ax.set_title(f"Projection ACP des distributions d'images{norm_notes[norm]}{clip_note}{outlier_note}")
    ax.legend(handles=handles, loc="best", framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Projection ACP des distributions d'images (référence / initial / output)."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="data1",
        choices=sorted(DATASETS.keys()),
        help="Nom du dataset à visualiser (data1, data2 ou data3). Par défaut : data1.",
    )
    parser.add_argument(
        "--ref",
        default=None,
        help="Chemin ou nom du dossier de référence (par défaut : Ref2 pour data3, Ref_filtered sinon).",
    )
    parser.add_argument(
        "--features",
        choices=["inception", "gram", "dinov2"],
        default="inception",
        help="Espace de caractéristiques : 'inception' (comme le FID, défaut), "
             "'gram' (comme le Gram-MMD) ou 'dinov2' (embedding de classe, meilleure variance).",
    )
    parser.add_argument(
        "--dinov2-model",
        choices=sorted(MODEL_METADATA.keys()),
        default="dinov2-vit-b-14",
        help="Modèle DINOv2 (si --features dinov2). Défaut : dinov2-vit-b-14 (dim 768).",
    )
    parser.add_argument(
        "--norm",
        choices=["std", "l2", "none"],
        default="std",
        help="Normalisation avant ACP : 'std' (StandardScaler, défaut), "
             "'l2' (normalisation L2 par image) ou 'none'.",
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=5.0,
        help="Winsorisation : les valeurs de chaque dimension sont bornées aux "
             "percentiles [clip, 100-clip] avant normalisation (0 = désactivé). "
             "Réduit l'impact des outliers qui aplatissent la projection. Défaut : 5.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=300,
        help="Nombre maximum de points affichés par classe dans le nuage (0 = tous). Défaut : 300.",
    )
    parser.add_argument(
        "--filter-outliers",
        type=float,
        default=0.0,
        help="Pourcentage d'outliers 2D à filtrer par classe (via distance de Mahalanobis) "
             "pour nettoyer les contours KDE et les limites du graphique (ex. 2.0 pour 2%%). Défaut : 0 (désactivé).",
    )
    args = parser.parse_args()

    dataset = args.dataset
    ref_dir = get_ref_dir(dataset, args.ref)
    ref_name = os.path.basename(ref_dir.rstrip("/"))
    initial_dir, output_dir = get_dataset_paths(dataset)

    for folder in [ref_dir, initial_dir, output_dir]:
        if not os.path.exists(folder):
            print(f"Erreur : Le dossier '{folder}' n'existe pas.")
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du périphérique : {device}")

    if args.features == "inception":
        print("Initialisation de l'extracteur Inception-v3 (espace FID)...")
        extractor = FeatureExtractorInceptionV3(
            name="inception_v3", features_list=["2048"]
        ).to(device)
        extract_fn = extract_inception_features
        batch_size = 32
    elif args.features == "dinov2":
        print(f"Initialisation de l'extracteur DINOv2 ({args.dinov2_model}, espace dim "
              f"{MODEL_METADATA[args.dinov2_model].replace('dinov2_vit', '')})...")
        extractor = FeatureExtractorDinoV2(
            name=args.dinov2_model, features_list=["dinov2"]
        ).to(device)
        extract_fn = extract_dinov2_features
        batch_size = 32
    else:
        print("Initialisation de l'extracteur de caractéristiques de texture (Gram Matrices)...")
        from Gram_MMD import GramFeatureExtractor
        extractor = GramFeatureExtractor().to(device)
        extract_fn = extract_gram_features
        batch_size = 4

    model_name = args.dinov2_model if args.features == "dinov2" else None
    cache_p = feature_cache_path(dataset, args.features, model=model_name, ref_name=ref_name)
    cached = load_features_from_cache(cache_p)
    if cached is not None:
        print(f"Chargement des caractéristiques depuis le cache ({cache_p})...")
        feats_ref, feats_init, feats_out = cached
    else:
        try:
            print(f"Extraction des caractéristiques — Référence '{ref_dir}'...")
            feats_ref = extract_fn(ref_dir, extractor, device, batch_size=batch_size)
            print(f"Extraction des caractéristiques — Initial '{initial_dir}'...")
            feats_init = extract_fn(initial_dir, extractor, device, batch_size=batch_size)
            print(f"Extraction des caractéristiques — Amélioré '{output_dir}'...")
            feats_out = extract_fn(output_dir, extractor, device, batch_size=batch_size)
        except Exception as e:
            print(f"Une erreur est survenue lors de l'extraction : {e}")
            import traceback
            traceback.print_exc()
            return
        print("Sauvegarde des caractéristiques dans le cache...")
        save_features_to_cache(cache_p, feats_ref, feats_init, feats_out)

    print(f"Normalisation des caractéristiques ({args.norm})...")
    combined = np.concatenate([feats_ref, feats_init, feats_out], axis=0)
    if args.clip > 0:
        print(f"Winsorisation des caractéristiques aux percentiles [{args.clip:.0f}, {100 - args.clip:.0f}]...")
        lo = np.percentile(combined, args.clip, axis=0, keepdims=True)
        hi = np.percentile(combined, 100 - args.clip, axis=0, keepdims=True)
        combined = np.clip(combined, lo, hi)
    combined_norm = normalize_features(combined, norm=args.norm)

    print("Calcul de l'ACP...")
    scores, explained = fit_pca(combined_norm)

    n_ref, n_init, n_out = len(feats_ref), len(feats_init), len(feats_out)
    scores_dict = {
        "reference": scores[:n_ref],
        "initial": scores[n_ref:n_ref + n_init],
        "output": scores[n_ref + n_init:],
    }

    result_dir = os.path.join(RESULTS_DIR, dataset)
    os.makedirs(result_dir, exist_ok=True)
    norm_suffix = "" if args.norm == "std" else f"_{args.norm}"
    filter_suffix = f"_clean" if args.filter_outliers > 0 else ""
    plot_path = os.path.join(result_dir, f"pca_{args.features}{norm_suffix}{filter_suffix}.png")
    plot_pca(scores_dict, explained, plot_path, max_points=args.max_points,
             norm=args.norm, clip=args.clip, filter_outliers=args.filter_outliers)

    report_file = os.path.join(result_dir, f"pca_{args.features}{norm_suffix}{filter_suffix}.txt")
    clip_txt = f"[{args.clip:.0f}, {100 - args.clip:.0f}]" if args.clip > 0 else "non"
    outlier_txt = f"{args.filter_outliers:.1f} %" if args.filter_outliers > 0 else "non"
    report_content = f"""==================================================
PROJECTION ACP DES DISTRIBUTIONS D'IMAGES
==================================================
Espace de caractéristiques : {args.features}
  Modèle                : {args.dinov2_model if args.features == "dinov2" else "-"}
  Normalisation avant ACP : {args.norm}
  Winsorisation (clip)   : {clip_txt}
  Filtrage outliers 2D   : {outlier_txt}
Dossier de Référence (Réel)     : {ref_dir}   ({n_ref} images)
Dossier Initial (Synthétique)   : {initial_dir} ({n_init} images)
Dossier Amélioré (Output)       : {output_dir} ({n_out} images)
--------------------------------------------------
> Variance PC1 : {explained[0]:.2f} %
> Variance PC2 : {explained[1]:.2f} %
> Variance cumulée (PC1+PC2) : {explained.sum():.2f} %
--------------------------------------------------
Graphique sauvegardé : {plot_path}
==================================================
"""
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(report_content)
    print(f"\nCalcul terminé ! Le graphique a été sauvegardé dans '{plot_path}'.")


if __name__ == "__main__":
    main()
