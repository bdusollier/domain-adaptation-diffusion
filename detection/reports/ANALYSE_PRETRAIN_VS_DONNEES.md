# Analyse Détaillée : Rôle du Pré-entraînement COCO vs Données Synthétiques/Réalistes

**Date :** 2026-08-26  
**Auteur :** Antigravity (Analyse du Benchmark de Détection YOLO)  
**Emplacement :** `dataset2/Results/`

---

## 1. La Question Centrale

> *« Les performances de chaque modèle sont les meilleures au tout début de l'entraînement, c'est-à-dire grâce au modèle pré-entraîné et pas du tout à cause de mes données ? »*

**Réponse courte : Oui, en grande partie.** 
À l'initialisation, le modèle `yolov8s.pt` bénéficie massivement de ses représentations apprises sur le jeu de données **COCO** (des millions de photos réelles contenant déjà des voitures, camions, bus et véhicules divers). 

Lors de l'entraînement sur vos données sources (synthétiques ou réalistes Flux), un phénomène de **transfert négatif** et de **surapprentissage de domaine (*domain overfitting*)** se produit, ce qui explique pourquoi la performance sur les vraies photos de test culmine au début avant de stagner ou de régresser.

---

## 2. Décryptage du Comportement Observé à l'Entraînement

Dans les fichiers `results.csv` des entraînements :

### Modèle Réaliste (Full Fine-Tuning sans Freeze) :
* **Epoch 1** : `train/cls_loss` = 3.94 | `val/cls_loss` = **2.75** | **mAP50 = 0.2868** (Meilleur score absolu)
* **Epoch 2** : `train/cls_loss` = 1.74 | `val/cls_loss` = 3.54 | mAP50 = 0.1562
* **Epoch 6** : `train/cls_loss` = 1.48 | `val/cls_loss` = 5.17 | mAP50 = 0.0363
* **Epoch 12** : `train/cls_loss` = 1.17 | `val/cls_loss` = 5.13 | **mAP50 = 0.0083** (Effondrement)

### Que s'est-il passé ?
1. **À l'Epoch 1** : Le modèle part des poids `yolov8s.pt` (COCO). Les filtres de vision du monde réel sont intacts. La seule couche qui apprend est la tête de classification, qui associe rapidement les caractéristiques visuelles des véhicules COCO à la classe `target`.
2. **Aux Epochs 2 à 20** : Pour minimiser sa perte d'entraînement sur les images Flux, le modèle modifie ses couches profondes pour devenir ultra-spécifique aux **textures et artefacts de diffusion de Flux** (*shortcut learning* / signature du générateur).
3. **Le coût de cette spécialisation** : En s'adaptant au domaine Flux, il **détruit ses filtres de vision réelle** (*catastrophic forgetting*). Quand on l'évalue sur `val_real` (de vraies photos), il ne reconnaît plus rien (`val/cls_loss` explose de 2.75 à 5.17).

---

## 3. Pourquoi le gel du Backbone (`--freeze 10`) a fait bondir le score de 0.27 à 0.53 ?

Quand nous avons gelé les 10 premières couches du backbone :
* Nous avons **interdit à YOLO de modifier ses extracteurs de caractéristiques réelles COCO**.
* Le modèle a donc été forcé de conserver sa vision "monde réel", tout en utilisant vos données pour calibrer la détection spécifique des véhicules militaires.
* **Résultat** : Le mAP50 est passé de **0.2753 à 0.5340** pour le modèle réaliste et de **0.3775 à 0.6369** pour le modèle synthétique.

Cela prouve que **le backbone pré-entraîné sur des données réelles était le facteur clé de la performance**.

---

## 4. Pourquoi les Données Synthétiques 3D restent-elles supérieures aux Données Réalistes Flux ?

Malgré le fait que le dossier `measures/` montre un meilleur réalisme photométrique pour Flux (FID de 95.6 vs 122.3) :

1. **Précision géométrique des Bounding Boxes (Label Drift)** :
   * Les boîtes annotées proviennent du moteur 3D (précision au pixel près).
   * Flux modifie subtilement la géométrie (déplacement de roues, déformation du canon, flou sur les arêtes).
   * Les boîtes d'annotation ne collent plus exactement aux véhicules générés par Flux, ce qui perturbe la fonction de perte de boîte (`box_loss` et `dfl_loss`).
2. **Netteté structurelle vs Bruit de diffusion** :
   * Les images 3D synthétiques ont des contrastes forts et des arêtes très nettes qui facilitent l'apprentissage des formes rigides des véhicules militaires.
   * La diffusion crée un effet de camouflage naturel (harmonisation des ombres et lumières) et des micro-textures qui brouillent la détection des contours.

---

## 5. Comment isoler scientifiquement la vraie valeur de vos données ?

Pour savoir exactement ce que valent vos données synthétiques et réalistes **sans l'aide de COCO**, deux expériences complémentaires sont nécessaires :

### Expérience A : Entraînement *From Scratch* (Poids aléatoires, 0% COCO)
* Entraîner YOLOv8s initialisé aléatoirement (`yolov8s.yaml`) sur :
  1. Données Synthétiques
  2. Données Réalistes (Flux)
  3. Données Combinées
* **Objectif** : Mesurer ce que le réseau apprend de zéro uniquement grâce à vos images.

### Expérience B : Évaluation Baseline COCO Pure (Zéro-Shot)
* Prendre `yolov8s.pt` sans aucun entraînement, mapper les classes de véhicules COCO (`car`, `truck`, `bus`) sur `target` et évaluer sur le test set de véhicules militaires.
* **Objectif** : Connaître le score de départ exact offert gratuitement par COCO.

---

## 6. Synthèse des Recommandations Techniques

1. **Toujours geler le backbone (`--freeze 10`)** lors de l'adaptation de domaine avec un nombre modéré d'images (< 5000 images) pour éviter l'oubli catastrophique.
2. **Utiliser un learning rate très bas (`lr0 <= 0.0001`)** pour préserver la stabilité des représentations réelles.
3. **Privilégier le modèle Combiné ou Curriculum** : Le synthétique apporte la rigueur géométrique des boîtes, et le réaliste affine la précision des textures.
4. **Pour améliorer le pipeline Flux** : Utiliser un inpainting avec masque strict ou ControlNet Canny/Depth pour empêcher Flux de déformer la silhouette du véhicule par rapport à sa bounding box d'origine.
