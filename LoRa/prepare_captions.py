#!/usr/bin/env python3
import os
import re
import sys
import glob
from pathlib import Path

TRIGGER_WORD = "skytroie"

def normalize_tag(t: str) -> str:
    # Gère les éventuelles anomalies d'encodage de caractères (ex: cr├⌐puscule)
    if "cr" in t and "puscule" in t:
        return "crépuscule"
    return t

def parse_filename_to_caption(filename: str) -> str:
    match = re.search(r'\[(.*?)\]', filename)
    if not match:
        return f"{TRIGGER_WORD}, aerial drone photograph of military vehicles and terrain in outdoor landscape"
    
    raw_tags = set(normalize_tag(t) for t in match.group(1).split())
    
    elements = [TRIGGER_WORD]
    
    # 1. Angle et type de prise de vue
    if "nadir" in raw_tags:
        elements.append("top-down nadir drone view")
    elif "vue_horizontale" in raw_tags:
        elements.append("horizontal eye-level drone view")
    elif "vol" in raw_tags:
        elements.append("aerial drone view")
    else:
        elements.append("aerial drone view")
        
    # 2. Conditions météorologiques
    has_nuage = "Nuage" in raw_tags
    has_beau = "Beau" in raw_tags
    if has_nuage and has_beau:
        elements.append("partly cloudy sky, scattered clouds")
    elif has_beau:
        elements.append("clear sunny sky")
    elif has_nuage:
        elements.append("overcast cloudy sky")
        
    # 3. Luminosité / Moment de la journée
    if "crépuscule" in raw_tags:
        elements.append("twilight dusk lighting")
    elif "jour" in raw_tags:
        elements.append("daytime daylight")
        
    # 4. Objets : Véhicules (Dédoublonnage hiérarchique)
    has_mil_veh = "vehicule_militaire" in raw_tags
    has_civ_veh = "vehicule_civil" in raw_tags
    if has_mil_veh and has_civ_veh:
        elements.append("military vehicles, armored vehicles, civilian cars")
    elif has_mil_veh:
        elements.append("military vehicle, armored vehicle")
    elif has_civ_veh:
        elements.append("civilian vehicle, car")
    elif "vehicule" in raw_tags:
        elements.append("vehicle")
        
    # 5. Sujets : Personnes (Dédoublonnage hiérarchique)
    has_mil_hum = "humain_militaire" in raw_tags
    has_civ_hum = "humain_civil" in raw_tags
    if has_mil_hum and has_civ_hum:
        elements.append("military personnel, soldiers, civilians")
    elif has_mil_hum:
        elements.append("military soldier, tactical personnel")
    elif has_civ_hum:
        elements.append("civilian person")
    elif "humain" in raw_tags:
        elements.append("person")
        
    # 6. Environnement / Terrain
    if "non_urbain" in raw_tags:
        elements.append("non-urban terrain, rural countryside")
    elif "urbain" in raw_tags:
        elements.append("urban environment")
        
    # 7. Capteur / Type d'image
    if "RGB" in raw_tags:
        elements.append("RGB camera photograph")
        
    # 8. Métadonnées géométriques de la caméra
    pitch_match = re.search(r'pitch-(-?\d+)-(\d+)', filename)
    dist_match = re.search(r'dist-(\d+)-(\d+)', filename)
    
    if pitch_match:
        pitch = pitch_match.group(1)
        elements.append(f"camera pitch {pitch} degrees")
    if dist_match:
        dist = dist_match.group(1)
        elements.append(f"distance {dist}m")
        
    caption = ", ".join(elements)
    return caption

def main():
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "./Dataset"
    dataset_dir = Path(target_dir)
    if not dataset_dir.exists():
        print(f"Directory {dataset_dir} does not exist.")
        return
    
    images = sorted(list(dataset_dir.glob("*.jpg")) + list(dataset_dir.glob("*.png")))
    print(f"Found {len(images)} images in {dataset_dir}")
    
    for img_path in images:
        caption = parse_filename_to_caption(img_path.name)
        txt_path = img_path.with_suffix(".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(caption + "\n")
            
    print(f"Successfully generated {len(images)} clean caption .txt files with trigger word '{TRIGGER_WORD}'.")

if __name__ == "__main__":
    main()
