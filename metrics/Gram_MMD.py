import os
import glob
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from scipy.spatial.distance import cdist
import numpy as np

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

# --- CONFIGURATION DE LA MÉTRIQUE GMMD ---
from torchvision.models import vgg19, VGG19_Weights

class GramFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        self.features = nn.Sequential(*list(vgg.children())[:22]) # Jusqu'à relu4_1
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        act = self.features(x) # [B, C, H, W]
        B, C, H, W = act.shape
        F = act.view(B, C, H * W)
        G = torch.bmm(F, F.transpose(1, 2)) / (H * W) # [B, C, C]
        
        vectors = []
        tri_indices = torch.triu_indices(C, C)
        for i in range(B):
            v = G[i][tri_indices[0], tri_indices[1]]
            vectors.append(v)
            
        return torch.stack(vectors) # [B, d*(d+1)/2]

def extract_features_batch(directory, extractor, device, batch_size=16):
    """
    Charge les images et extrait leurs vecteurs de Gram par batchs successifs 
    pour éviter de saturer la mémoire GPU (CUDA Out Of Memory).
    """
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    img_paths = [p for p in glob.glob(os.path.join(directory, "*")) if p.lower().endswith(valid_exts)]
    
    if len(img_paths) == 0:
        raise ValueError(f"Aucune image valide trouvée dans {directory}")
        
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    all_vectors = []
    
    # Traitement séquentiel par batch
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i + batch_size]
        batch_images = []
        
        for path in batch_paths:
            try:
                with Image.open(path).convert('RGB') as img:
                    batch_images.append(transform(img))
            except Exception as e:
                print(f"Impossible de charger l'image {path}: {e}")
                
        if not batch_images:
            continue
            
        # Création du mini-batch et envoi sur le GPU
        X_batch = torch.stack(batch_images).to(device)
        
        # Extraction hors gradient
        with torch.no_grad():
            v_batch = extractor(X_batch).cpu().numpy() # Transfert immédiat sur CPU (RAM)
            all_vectors.append(v_batch)
            
    return np.concatenate(all_vectors, axis=0)

def compute_gmmd(eval_dir, anchor_dir, extractor, device, batch_size=16):
    """
    Calcule le score GRAM-MMD (GMMD) en extrayant les caractéristiques par batchs.
    """
    # Extraction sécurisée sur GPU par blocs, stockage intermédiaire en RAM
    v_eval = extract_features_batch(eval_dir, extractor, device, batch_size=batch_size)     # [N_E, D]
    v_anchor = extract_features_batch(anchor_dir, extractor, device, batch_size=batch_size) # [N_A, D]
        
    # Standardisation basée sur l'ancrage (Équation 7)
    mu_anchor = v_anchor.mean(axis=0, keepdims=True)
    sigma_anchor = v_anchor.std(axis=0, keepdims=True) + 1e-6
    
    v_eval_scaled = (v_eval - mu_anchor) / sigma_anchor
    v_anchor_scaled = (v_anchor - mu_anchor) / sigma_anchor
    
    # Calcul de l'heuristique de la médiane pour gamma (Section 3.4)
    dist_anchor = cdist(v_anchor_scaled, v_anchor_scaled, metric='sqeuclidean')
    median_dist = np.median(dist_anchor[dist_anchor > 0]) if np.any(dist_anchor > 0) else 1.0
    gamma = 1.0 / (2.0 * median_dist)
    
    # Calcul des matrices de noyaux RBF (sur CPU avec scipy pour économiser la VRAM)
    K_AA = np.exp(-gamma * dist_anchor)
    K_EE = np.exp(-gamma * cdist(v_eval_scaled, v_eval_scaled, metric='sqeuclidean'))
    K_AE = np.exp(-gamma * cdist(v_anchor_scaled, v_eval_scaled, metric='sqeuclidean'))
    
    # Estimateur non biaisé de la MMD^2 (Équation 8)
    N_A = v_anchor.shape[0]
    N_E = v_eval.shape[0]
    
    np.fill_diagonal(K_AA, 0)
    term_AA = K_AA.sum() / (N_A * (N_A - 1))
    
    np.fill_diagonal(K_EE, 0)
    term_EE = K_EE.sum() / (N_E * (N_E - 1))
    
    term_AE = 2.0 * K_AE.sum() / (N_A * N_E)
    
    gmmd_squared = term_AA + term_EE - term_AE
    return float(gmmd_squared)

def main():
    parser = argparse.ArgumentParser(description="Calcul du score Gram-MMD.")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="data1",
        choices=sorted(DATASETS.keys()),
        help="Nom du dataset à évaluer (data1, data2 ou data3). Par défaut : data1.",
    )
    parser.add_argument(
        "--ref",
        default=None,
        help="Chemin ou nom du dossier de référence (par défaut : Ref2 pour data3, Ref_filtered sinon).",
    )
    args = parser.parse_args()

    dataset = args.dataset
    ref_dir = get_ref_dir(dataset, args.ref)
    initial_dir, output_dir = get_dataset_paths(dataset)
    
    for folder in [ref_dir, initial_dir, output_dir]:
        if not os.path.exists(folder):
            print(f"Erreur : Le dossier '{folder}' n'existe pas.")
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du périphérique : {device}")
    
    print("Initialisation de l'extracteur de caractéristiques de texture (Gram Matrices)...")
    extractor = GramFeatureExtractor().to(device)

    # Paramètre de batch_size ajustable en fonction de ta VRAM restante
    # Avec 16 ou 32, l'empreinte mémoire reste minime.
    BATCH_SIZE = 4

    try:
        print(f"\nCalcul du GMMD pour le dossier Initial '{initial_dir}' par rapport à la Référence (Batch size={BATCH_SIZE})...")
        gmmd_initial = compute_gmmd(initial_dir, ref_dir, extractor, device, batch_size=BATCH_SIZE)
        
        print(f"Calcul du GMMD pour le dossier Amélioré '{output_dir}' par rapport à la Référence (Batch size={BATCH_SIZE})...")
        gmmd_improved = compute_gmmd(output_dir, ref_dir, extractor, device, batch_size=BATCH_SIZE)
    except Exception as e:
        print(f"Une erreur est survenue lors du calcul : {e}")
        import traceback
        traceback.print_exc()
        return

    gmmd_diff = gmmd_initial - gmmd_improved
    
    if gmmd_improved < gmmd_initial:
        conclusion = (
            f"Succès ! Le réalisme des textures s'est amélioré.\n"
            f"Le score GMMD a diminué de {gmmd_diff:.6f} points."
        )
    elif gmmd_improved > gmmd_initial:
        conclusion = (
            f"Échec. Le réalisme micro-structural et textural s'est dégradé.\n"
            f"Le score GMMD a augmenté de {abs(gmmd_diff):.6f} points. Les modifications ont altéré le grain ou le style par rapport aux images réelles."
        )
    else:
        conclusion = "Aucun changement significatif du score GMMD."

    result_dir = os.path.join(RESULTS_DIR, dataset)
    os.makedirs(result_dir, exist_ok=True)
    ref_name = os.path.basename(ref_dir.rstrip("/"))
    result_file = os.path.join(result_dir, f"results_gmmd_{ref_name}.txt" if args.ref else "results_gmmd.txt")
    report_content = f"""==================================================
BILAN DE L'ÉVALUATION DU RÉALISME TEXTURAL (GMMD)
==================================================
Dossier d'Ancrage (Réel)     : {ref_dir}
Dossier Initial (Évaluation) : {initial_dir}
Dossier Amélioré (Output)     : {output_dir}
--------------------------------------------------
> GMMD (Initial vs Référence)  : {gmmd_initial:.6f}
> GMMD (Amélioré vs Référence) : {gmmd_improved:.6f}
--------------------------------------------------
CONCLUSION :
{conclusion}
==================================================
"""

    with open(result_file, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\nCalcul terminé ! Les résultats ont été sauvegardés dans '{result_file}'.")
    print(report_content)

if __name__ == "__main__":
    main()
