import os
import argparse
from torch_fidelity import calculate_metrics

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

def get_fid(images_dir, reference_dir):
    """
    Calcule le score FID entre deux dossiers d'images.
    """
    print(f"Calcul du FID entre '{images_dir}' et '{reference_dir}'...")
    
    metrics = calculate_metrics(
        input1=images_dir,
        input2=reference_dir,
        cuda=True,
        fid=True,
        verbose=False,
        batch_size=1,
    )
    return metrics['frechet_inception_distance']

def main():
    parser = argparse.ArgumentParser(description="Calcul du score FID.")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="data1",
        choices=["data1", "data2", "data3"],
        help="Nom du dataset à évaluer (data1, data2 ou data3). Par défaut : data1.",
    )
    parser.add_argument(
        "--ref",
        default=None,
        help="Chemin ou nom du dossier de référence (par défaut : Reference ou Ref_filtered).",
    )
    args = parser.parse_args()

    dataset = args.dataset
    ref_dir = get_ref_dir(dataset, args.ref)
    initial_dir, output_dir = get_dataset_paths(dataset)
    
    # Vérification de l'existence des dossiers
    for folder in [ref_dir, initial_dir, output_dir]:
        if not os.path.exists(folder):
            print(f"Erreur : Le dossier '{folder}' n'existe pas.")
            return

    # 2. Calcul des scores FID
    try:
        fid_initial = get_fid(initial_dir, ref_dir)
        fid_improved = get_fid(output_dir, ref_dir)
    except Exception as e:
        print(f"Une erreur est survenue lors du calcul : {e}")
        return

    # 3. Analyse comparative
    # Plus le FID est bas, plus l'image est proche du dataset réel (réalisme élevé)
    fid_diff = fid_initial - fid_improved
    
    if fid_improved < fid_initial:
        conclusion = (
            f"Succès ! Le réalisme global s'est amélioré.\n"
            f"Le score FID a diminué de {fid_diff:.4f} points."
        )
    elif fid_improved > fid_initial:
        conclusion = (
            f"Échec. Le réalisme statistique s'est dégradé.\n"
            f"Le score FID a augmenté de {abs(fid_diff):.4f} points. Les modifications ont éloigné les images du dataset réel."
        )
    else:
        conclusion = "Aucun changement significatif du score FID."

    # 4. Écriture du fichier de résultats
    result_dir = os.path.join(RESULTS_DIR, dataset)
    os.makedirs(result_dir, exist_ok=True)
    ref_name = os.path.basename(ref_dir.rstrip("/"))
    result_file = os.path.join(result_dir, f"results_{ref_name}.txt" if args.ref else "results.txt")
    
    report_content = f"""==================================================
BILAN DE L'ÉVALUATION DU RÉALISME (FID)
==================================================
Dossier de Référence (Réel) : {ref_dir}
Dossier Initial (Synthétique) : {initial_dir}
Dossier Amélioré (Output)     : {output_dir}
--------------------------------------------------
> FID (Initial vs Référence)  : {fid_initial:.4f}
> FID (Amélioré vs Référence) : {fid_improved:.4f}
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
